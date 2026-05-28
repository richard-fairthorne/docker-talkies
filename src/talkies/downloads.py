"""URL fetcher for the /v1/audio/transcriptions ``file_path`` field.

When ``file_path`` is an http(s) URL, the request is routed through this
module: download the URL once into ``${FILES_DIR}/downloads/<key>``, then
hand the cached path back to the transcription pipeline. Subsequent
requests for the same URL hit the cached file (no re-download).

Cache key: ``downloads/<sha256(url)[:16]>-<safe-basename>``. The hash
guarantees no cross-host collisions; the trailing readable basename
makes ``GET /v1/files`` listings legible.

Safety:

  - SSRF guard (off by default): when ``TALKIES_BLOCK_PRIVATE_DOWNLOADS``
    is true, every URL hop's hostname is resolved and rejected if any
    answer falls in a private / loopback / link-local / multicast /
    metadata range. Defaults to permissive because the typical
    self-hosted deployment is a LAN box fetching from another LAN box.
  - Redirects: followed manually (httpx ``follow_redirects=False``) so
    each hop gets re-checked against the SSRF guard. Capped at five
    hops.
  - Size: streamed to disk with a per-download cap from
    ``TALKIES_MAX_DOWNLOAD_BYTES`` (default 1 GiB). Exceeding the cap
    aborts the download and removes the partial file.
  - Concurrency: a per-URL ``asyncio.Lock`` prevents two simultaneous
    requests from double-fetching the same URL — the second waiter
    observes the cache hit after the first finishes.
"""

from __future__ import annotations

import asyncio
import hashlib
import ipaddress
import logging
import os
import re
import socket
from pathlib import Path
from urllib.parse import unquote, urlparse

import httpx

from . import config


_DOWNLOADS_SUBDIR = "downloads"
_HASH_LEN = 16
_MAX_REDIRECTS = 5
_CONNECT_TIMEOUT = 10.0
_READ_TIMEOUT = 300.0
_CHUNK_BYTES = 64 * 1024
_BASENAME_FALLBACK = "download.bin"
_BASENAME_MAX = 96
_SAFE_BASENAME_RE = re.compile(r"[^A-Za-z0-9._-]+")


_log = logging.getLogger("talkies.downloads")

_url_locks: dict[str, asyncio.Lock] = {}
_url_locks_guard = asyncio.Lock()


class DownloadError(Exception):
    """Raised when a URL fetch fails (DNS, SSRF, HTTP, size cap, ...)."""


def is_url(s: str) -> bool:
    return s.startswith("http://") or s.startswith("https://")


def _hash_url(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()[:_HASH_LEN]


def _safe_basename(url: str) -> str:
    parsed = urlparse(url)
    name = os.path.basename(unquote(parsed.path or ""))
    if not name:
        return _BASENAME_FALLBACK
    name = _SAFE_BASENAME_RE.sub("_", name).strip("._-")
    if not name:
        return _BASENAME_FALLBACK
    if len(name) <= _BASENAME_MAX:
        return name
    stem, dot, ext = name.rpartition(".")
    if dot and 0 < len(ext) < 16:
        keep = _BASENAME_MAX - len(ext) - 1
        return f"{stem[:keep]}.{ext}"
    return name[:_BASENAME_MAX]


def cache_path_for(url: str) -> Path:
    return config.FILES_DIR / _DOWNLOADS_SUBDIR / f"{_hash_url(url)}-{_safe_basename(url)}"


def cache_relpath_for(url: str) -> str:
    return f"{_DOWNLOADS_SUBDIR}/{_hash_url(url)}-{_safe_basename(url)}"


async def _lock_for(key: str) -> asyncio.Lock:
    async with _url_locks_guard:
        lock = _url_locks.get(key)
        if lock is None:
            lock = asyncio.Lock()
            _url_locks[key] = lock
        return lock


def _check_ssrf(host: str) -> None:
    if not config.BLOCK_PRIVATE_DOWNLOADS:
        return
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise DownloadError(f"DNS resolution failed for {host!r}: {exc}") from exc
    for info in infos:
        family = info[0]
        sockaddr = info[4]
        addr_str = sockaddr[0] if isinstance(sockaddr[0], str) else ""
        if not addr_str:
            continue
        if family == socket.AF_INET:
            ip: ipaddress.IPv4Address | ipaddress.IPv6Address = ipaddress.IPv4Address(addr_str)
        elif family == socket.AF_INET6:
            # strip scope id if present ("fe80::1%eth0" → "fe80::1")
            ip = ipaddress.IPv6Address(addr_str.split("%", 1)[0])
        else:
            continue
        if (
            ip.is_private
            or ip.is_loopback
            or ip.is_link_local
            or ip.is_multicast
            or ip.is_unspecified
            or ip.is_reserved
        ):
            raise DownloadError(
                f"refused to fetch {host!r}: resolved IP {ip} is in a "
                f"blocked range (set TALKIES_BLOCK_PRIVATE_DOWNLOADS=false "
                f"to allow private-network downloads)"
            )


async def ensure_downloaded(url: str) -> Path:
    """Return the local cache path for ``url``, downloading if missing."""
    dest = cache_path_for(url)
    if dest.is_file() and dest.stat().st_size > 0:
        _log.info("download cache hit: %s -> %s", url, dest)
        return dest
    lock = await _lock_for(_hash_url(url))
    async with lock:
        # Re-check inside the lock — another coroutine may have fetched
        # this URL while we were waiting on the lock.
        if dest.is_file() and dest.stat().st_size > 0:
            return dest
        await _download_streaming(url, dest)
    return dest


async def _download_streaming(initial_url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    timeout = httpx.Timeout(
        connect=_CONNECT_TIMEOUT, read=_READ_TIMEOUT, write=None, pool=None
    )
    url = initial_url
    async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
        for hop in range(_MAX_REDIRECTS + 1):
            parsed = urlparse(url)
            if parsed.scheme not in ("http", "https"):
                raise DownloadError(f"unsupported URL scheme: {parsed.scheme!r}")
            if not parsed.hostname:
                raise DownloadError(f"URL has no host: {url!r}")
            _check_ssrf(parsed.hostname)
            try:
                async with client.stream("GET", url) as resp:
                    if resp.is_redirect:
                        loc = resp.headers.get("Location")
                        if not loc:
                            raise DownloadError(
                                f"redirect with no Location header from {url!r}"
                            )
                        url = str(resp.url.join(loc))
                        _log.info("redirect hop %d -> %s", hop + 1, url)
                        continue
                    if resp.status_code >= 400:
                        raise DownloadError(
                            f"download failed: HTTP {resp.status_code} from {url!r}"
                        )
                    await _stream_to_disk(resp, tmp, dest, initial_url)
                    return
            except httpx.RequestError as exc:
                _safe_unlink(tmp)
                raise DownloadError(f"download failed for {url!r}: {exc}") from exc
        _safe_unlink(tmp)
        raise DownloadError(
            f"too many redirects (>{_MAX_REDIRECTS}) following {initial_url!r}"
        )


async def _stream_to_disk(
    resp: httpx.Response, tmp: Path, dest: Path, initial_url: str
) -> None:
    written = 0
    try:
        with open(tmp, "wb") as fh:
            async for chunk in resp.aiter_bytes(chunk_size=_CHUNK_BYTES):
                if not chunk:
                    continue
                written += len(chunk)
                if written > config.MAX_DOWNLOAD_BYTES:
                    raise DownloadError(
                        f"download exceeded MAX_DOWNLOAD_BYTES "
                        f"({written} > {config.MAX_DOWNLOAD_BYTES}) "
                        f"for {initial_url!r}"
                    )
                fh.write(chunk)
        os.replace(tmp, dest)
        _log.info("downloaded %s -> %s (%d bytes)", initial_url, dest, written)
    except Exception:
        _safe_unlink(tmp)
        raise


def _safe_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError:
        _log.exception("failed to unlink partial download at %s", path)
