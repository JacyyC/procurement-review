"""Shared SHA-256 helpers — the single implementation behind every digest (V1-06).

Historically ``_sha256`` had separate copies in ``gold_set`` (bytes) and
``eval_audit``/``eval_rescore`` (path); AC-04 deduplicates them here.  The chunked
streaming variant in ``public_corpus`` stays separate on purpose: it reads large
downloads in blocks instead of whole-file ``read_bytes``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


__all__ = ["sha256_bytes", "sha256_file"]
