"""Private exclusive H artifacts and a process-safe transition lock."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import tempfile
from contextlib import contextmanager
from pathlib import Path

from .model_wire_v2 import strict_json


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path: Path):
    return strict_json(Path(path).read_text(encoding="utf-8"))


def private(path: Path, *, directory: bool = False) -> None:
    mode = path.lstat().st_mode
    if (
        stat.S_ISLNK(mode)
        or (directory and not stat.S_ISDIR(mode))
        or (not directory and not stat.S_ISREG(mode))
    ):
        raise ValueError("h_artifact_file_type")
    if mode & 0o077:
        raise ValueError("h_artifact_permissions")


def json_bytes(value) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        + "\n"
    ).encode()


def write(path: Path, value, *, replace: bool = False) -> None:
    write_bytes(path, json_bytes(value), replace=replace)


def write_bytes(path: Path, body: bytes, *, replace: bool = False) -> None:
    if not replace:
        descriptor = os.open(
            path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW, 0o600
        )
        with os.fdopen(descriptor, "wb") as output:
            output.write(body)
            output.flush()
            os.fsync(output.fileno())
    else:
        private(path)
        descriptor, temporary = tempfile.mkstemp(
            prefix=".h-transaction-", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(body)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
    descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


@contextmanager
def locked(directory: Path, *, blocking: bool = True):
    private(directory, directory=True)
    path = directory / ".transition.lock"
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        private(path)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError:
            raise ValueError("h_concurrent_operation_in_progress") from None
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
