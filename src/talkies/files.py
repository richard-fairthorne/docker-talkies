"""File-staging area for the /v1/files API.

Clients PUT files under user-supplied relative paths; this module handles
the safety boundary between the URL path string and the filesystem path
under ``FILES_DIR``.

Sanitization rules (applied before any disk access):

  - leading ``/`` is stripped (so ``/foo/bar`` and ``foo/bar`` are equivalent)
  - null bytes rejected (would terminate the C string in the syscall)
  - backslashes rejected (no Windows-style separator smuggling)
  - ``.`` and ``..`` segments rejected (traversal)
  - empty path after stripping rejected

After lexical sanitization the candidate is joined with ``FILES_DIR``,
``Path.resolve()``-ed (which collapses any symlinks in parent dirs), and
the result is required to remain under ``FILES_DIR.resolve()``. This is
the belt-and-braces check: lexical validation catches the obvious cases,
the resolve+is_relative_to check catches anything that slipped through.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Iterator


class FilePathError(ValueError):
    """Raised when a user-supplied path fails sanitization."""


def sanitize_path(raw: str) -> PurePosixPath:
    """Validate a user-supplied path and return a safe relative POSIX path."""
    if raw is None or raw == "":
        raise FilePathError("path is empty")
    if "\x00" in raw:
        raise FilePathError("path contains null byte")
    if "\\" in raw:
        raise FilePathError("path contains backslash")
    stripped = raw.lstrip("/")
    if stripped == "":
        raise FilePathError("path is empty after stripping leading slashes")
    # Inspect raw segments before PurePosixPath gets to silently collapse
    # `.` entries — we want strict validation, not normalisation.
    for seg in stripped.split("/"):
        if seg == "":
            raise FilePathError("path contains empty segment (double slash)")
        if seg in (".", ".."):
            raise FilePathError(f"path contains forbidden segment {seg!r}")
    p = PurePosixPath(stripped)
    if p.is_absolute():
        raise FilePathError("path is absolute after normalisation")
    return p


def resolve_under(base: Path, rel: PurePosixPath) -> Path:
    """Join ``rel`` under ``base``, resolve symlinks, enforce containment."""
    base_real = base.resolve()
    candidate = (base_real / Path(*rel.parts)).resolve()
    if not candidate.is_relative_to(base_real):
        raise FilePathError("path escapes base directory")
    return candidate


def ensure_base(base: Path) -> None:
    base.mkdir(parents=True, exist_ok=True)


def write_atomic(dest: Path, data: bytes) -> None:
    """Write ``data`` to ``dest`` atomically (write-then-rename)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, dest)
    except Exception:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise


def prune_empty_parents(path: Path, base: Path) -> None:
    """Remove empty parent dirs of ``path`` up to (but not including) ``base``."""
    base_real = base.resolve()
    parent = path.parent.resolve()
    while parent != base_real and parent.is_relative_to(base_real):
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


def _iter_files(base: Path) -> Iterator[Path]:
    if not base.exists():
        return
    for root, _, names in os.walk(base, followlinks=False):
        for name in names:
            full = Path(root) / name
            if full.is_symlink():
                continue
            if not full.is_file():
                continue
            yield full


def list_files(base: Path) -> list[dict]:
    """Return a flat listing of every regular file under ``base``."""
    base_real = base.resolve()
    out: list[dict] = []
    for full in _iter_files(base_real):
        try:
            st = full.stat()
        except OSError:
            continue
        rel = full.relative_to(base_real)
        out.append(
            {
                "path": str(rel).replace(os.sep, "/"),
                "size": st.st_size,
                "modified": _utc_iso(st.st_mtime),
            }
        )
    out.sort(key=lambda x: x["path"])
    return out


def _utc_iso(epoch_seconds: float) -> str:
    return (
        datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )
