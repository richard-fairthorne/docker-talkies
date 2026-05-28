"""Shared helper for finding the .nemo archive inside a flat model dir.

snapshot_download(repo, local_dir=...) drops the full repo into one
directory. NeMo backends restore from a single .nemo archive inside it
(EncDecMultiTaskModel.restore_from, ASRModel.restore_from, SALM.restore_from).
This finds the archive — raises with a clear message if missing so the
failure points at the prefetch step, not at NeMo's internals.
"""

from __future__ import annotations

from pathlib import Path


def find_nemo_file(model_path: Path) -> str:
    if not model_path.is_dir():
        raise FileNotFoundError(
            f"model directory {model_path} not found — entrypoint "
            "prefetch did not populate it (check "
            "TALKIES_ENABLED_MODELS + container logs)"
        )
    matches = sorted(model_path.glob("*.nemo"))
    if not matches:
        raise FileNotFoundError(
            f"no .nemo file in {model_path} — expected exactly one, "
            f"found: {[p.name for p in model_path.iterdir()]}"
        )
    if len(matches) > 1:
        names = [p.name for p in matches]
        raise RuntimeError(
            f"multiple .nemo files in {model_path}: {names} — "
            "expected exactly one"
        )
    return str(matches[0])
