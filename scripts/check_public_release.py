#!/usr/bin/env python3
"""Check files intended for publication, without printing matched secret values.

Uses tracked + untracked, nonignored files when Git exists; otherwise checks the
directory tree. Pair with Gitleaks for generic assignments and service-specific
credential formats. Deliberately does not contact any external service.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
LOCAL_PARTS = {
    ".idea", ".vscode", ".claude", ".codex", ".trae", ".local", ".run",
    ".venv", "venv", "node_modules", "__pycache__", ".pytest_cache",
    ".ruff_cache", ".mypy_cache", "dist", "coverage", "test-results",
    "playwright-report", "agent-workspace", "workspace", "ui-02-evidence",
    "ui-03-evidence",
}
PRIVATE_SUFFIXES = {".pem", ".key", ".token", ".secret", ".db", ".sqlite",
                    ".sqlite3", ".log", ".har", ".dump", ".pyc", ".zip"}
PRIVATE_NAMES = {".npmrc", ".pypirc", ".DS_Store", "id_rsa", "id_ed25519"}
PRIVATE_KEY = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")
URL = re.compile(r"(?:https?|postgresql(?:\+\w+)?|mysql(?:\+\w+)?|redis)://[^\s\"'<>]+")
KEY = re.compile(
    r"(?<![A-Za-z0-9_-])(?:sk-(?:proj-)?[A-Za-z0-9_-]{20,}"
    r"|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}"
    r"|AKIA[A-Z0-9]{16})(?![A-Za-z0-9_-])"
)
AUTHOR_PATH = re.compile(r"/(?:Users|Volumes|home)/[^\s\"'<>]+")
REQUIRED = ("README.md", "LICENSE", "SECURITY.md", "THIRD_PARTY_NOTICES.md",
            ".gitignore", ".gitleaks.toml", "docs/release-preparation.md")


def publication_files(root: Path) -> list[Path]:
    if (root / ".git").exists():
        result = subprocess.run(
            ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            cwd=root, capture_output=True, check=True,
        )
        return sorted({root / p.decode() for p in result.stdout.split(b"\0") if p})
    return sorted(p for p in root.rglob("*") if p.is_file() or p.is_symlink())


def inspect_file(path: Path, root: Path) -> list[str]:
    relative = path.relative_to(root)
    errors: list[str] = []
    if ".git" in relative.parts:
        return errors
    if path.is_symlink():
        return ["symlink must be reviewed before publication"]
    if not path.is_file():
        return ["listed file is missing"]
    if LOCAL_PARTS.intersection(relative.parts):
        errors.append("local state or internal evidence is in the publication set")
    if path.name.startswith(".env") and not path.name.endswith(".example"):
        errors.append("real environment file is in the publication set")
    if path.name in PRIVATE_NAMES or path.suffix.lower() in PRIVATE_SUFFIXES:
        errors.append("private configuration or generated artifact")
    raw = path.read_bytes()
    if b"\0" in raw[:8192]:
        return errors
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return errors
    # Approve only the exact, reviewed synthetic values in the exact test files.
    # A new credential added to the same file remains a finding.
    fixture_path = root / "scripts" / "public_test_credentials.json"
    reviewed = json.loads(fixture_path.read_text()) if fixture_path.is_file() else []
    allowed = {(item["file"], item["sha256"]) for item in reviewed}
    unreviewed_keys = [m for m in KEY.finditer(text) if (
        relative.as_posix(), hashlib.sha256(m.group().encode()).hexdigest()
    ) not in allowed]
    if PRIVATE_KEY.search(text) or unreviewed_keys:
        errors.append("credential-shaped content requires review")
    for match in URL.finditer(text):
        try:
            url = urlsplit(match.group())
            host = url.hostname or ""
            if url.password and host not in {"localhost", "127.0.0.1", "db", "postgres"}:
                if not host.endswith((".invalid", ".test", ".example")):
                    errors.append("nonlocal URL contains an embedded password")
            if re.fullmatch(r"(?:10\.\d+\.\d+\.\d+|192\.168\.\d+\.\d+)", host):
                errors.append("private network endpoint requires review")
        except ValueError:
            pass  # String templates are checked as code, not parsed as live URLs.
    if path.suffix in {".md", ".example", ".sh"} and AUTHOR_PATH.search(text):
        errors.append("machine-specific path requires review")
    if path.suffix == ".py" and sys.version_info >= (3, 12):
        try:
            ast.parse(text, filename=str(relative))
        except SyntaxError:
            errors.append("Python syntax is invalid")
    elif path.suffix == ".json":
        try:
            json.loads(text)
        except json.JSONDecodeError:
            errors.append("JSON syntax is invalid")
    return sorted(set(errors))


def main() -> int:
    if sys.version_info < (3, 12):
        print("Note: Python syntax validation requires Python 3.12+; privacy checks still run")
    findings = [(name, "required public file is missing") for name in REQUIRED
                if not (ROOT / name).is_file()]
    files = publication_files(ROOT)
    for path in files:
        findings.extend((path.relative_to(ROOT).as_posix(), message)
                        for message in inspect_file(path, ROOT))
    for name, message in findings:
        print(f"FAIL {name}: {message}")
    print(f"Public release check: {len(files)} files, {len(findings)} findings")
    return int(bool(findings))


if __name__ == "__main__":
    sys.exit(main())
