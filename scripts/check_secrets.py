#!/usr/bin/env python3
"""Check that no real credential appears in Git history, staged content or worktree.

The script reads candidate values from local, untracked configuration files
(``.env`` in the repository root and ``provider.txt`` next to the repository),
keeps them only in memory, and reports matches by a short SHA-256 tag plus the
object/path where the match occurred. It never prints a secret value, a fragment
of a secret, a full URL or a hostname.

Usage::

    python scripts/check_secrets.py                 # full history + worktree
    python scripts/check_secrets.py --staged-only   # fast pre-commit check

Exit codes: 0 = clean, 1 = credential leak found, 2 = configuration error.

Security notes:
* Only values from keys whose name suggests a credential (KEY / TOKEN /
  SECRET / PASSWORD / CREDENTIAL) are treated as hard failures. Model names and
  endpoint URLs are public identifiers and are reported separately as INFO.
* TLS verification is not touched by this script; it performs no network calls.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
# Match credential *files* by name only. Deliberately avoid a loose "secret"
# substring, which would flag this very script (check_secrets.py).
SENSITIVE_NAME = re.compile(
    r"(^|/)\.env($|\.)|"
    r"(^|/)provider\.txt$|"
    r"(^|/)(credentials?|secrets?)\.[A-Za-z0-9]+$|"
    r"\.(pem|key|p12|pfx)$|"
    r"(^|/)id_(rsa|ed25519)$|"
    r"client_secret|service[-_]account",
    re.IGNORECASE,
)
SENSITIVE_NAME_EXEMPT = re.compile(r"(^|/)\.env\.example$", re.IGNORECASE)
CREDENTIAL_KEY = re.compile(r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL", re.IGNORECASE)
IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_.\-]*$")
# Deliberately conservative: no bare long-hex pattern, which would match Git
# object ids everywhere and produce false positives.
GENERIC_PATTERNS = (
    (b"sk-prefixed token", re.compile(rb"sk-[A-Za-z0-9_\-]{20,}", re.IGNORECASE)),
    (b"GitHub token", re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}")),
    (b"JWT-like", re.compile(rb"[A-Za-z0-9_\-]{30,}\.[A-Za-z0-9_\-]{20,}\.[A-Za-z0-9_\-]{10,}")),
    (b"Bearer literal", re.compile(rb"Bearer\s+[A-Za-z0-9._\-]{24,}", re.IGNORECASE)),
    (b"assigned api key", re.compile(
        rb"(?:api[_-]?key|access[_-]?token|secret)\s*[:=]\s*['\"]?[A-Za-z0-9_\-]{24,}",
        re.IGNORECASE)),
)


@dataclass(frozen=True)
class Candidate:
    tag: str
    key: str
    value: bytes
    is_credential: bool


def _add(candidates: list[Candidate], seen: set[str], value: str, key: str) -> None:
    value = value.strip().strip('"').strip("'").strip()
    if len(value) < 8 or not re.search(r"[A-Za-z0-9]", value):
        return
    tag = hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]
    if tag in seen:
        return
    seen.add(tag)
    candidates.append(Candidate(tag, key, value.encode("utf-8"),
                                bool(CREDENTIAL_KEY.search(key))))


def load_candidates() -> list[Candidate]:
    """Collect values from local config files without ever emitting them."""
    candidates: list[Candidate] = []
    seen: set[str] = set()
    sources = (REPO / ".env", REPO.parent / "provider.txt")
    for path in sources:
        if not path.is_file():
            continue
        for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, value = line.split("=", 1)
            elif ":" in line and not line.lower().startswith(("http://", "https://")):
                key, value = line.split(":", 1)
                if "://" in value:
                    # "base_url: https://..." is fine; splitting the URL itself is not.
                    pass
            else:
                continue
            key = key.strip()
            if not IDENTIFIER.match(key) or key.lower() in {"http", "https"}:
                continue
            if line.endswith(":") and "://" not in line:
                continue  # section header, e.g. "llm:"
            _add(candidates, seen, value, key)
    return candidates


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", "-C", str(REPO), *args],
                          capture_output=True, text=True, errors="replace")


def _object_paths() -> dict[str, str]:
    result = _git("rev-list", "--objects", "--all")
    mapping: dict[str, str] = {}
    for line in (result.stdout or "").splitlines():
        sha, _, path = line.partition(" ")
        if sha:
            mapping[sha] = path.strip()
    return mapping


def _iter_objects():
    """Yield (sha, type, content) for every object in the object database."""
    process = subprocess.Popen(
        ["git", "-C", str(REPO), "cat-file", "--batch-all-objects",
         "--batch", "--unordered"],
        stdout=subprocess.PIPE,
    )
    assert process.stdout is not None
    stream = process.stdout
    try:
        while True:
            header = stream.readline()
            if not header:
                break
            parts = header.split()
            if len(parts) < 3:
                continue
            sha = parts[0].decode("ascii", "replace")
            kind = parts[1]
            size = int(parts[2])
            content = stream.read(size)
            stream.read(1)  # trailing newline
            yield sha, kind, content
    finally:
        stream.close()
        process.wait()


def scan_history(candidates: list[Candidate]) -> tuple[list[tuple], list[tuple], int]:
    paths = _object_paths()
    credential_hits: list[tuple] = []
    info_hits: list[tuple] = []
    pattern_hits: list[tuple] = []
    for sha, kind, content in _iter_objects():
        if kind != b"blob":
            # commit/tag messages can also carry pasted credentials
            for candidate in candidates:
                if candidate.is_credential and candidate.value in content:
                    credential_hits.append((candidate.tag, candidate.key,
                                            kind.decode(), sha[:10]))
            continue
        location = paths.get(sha, "?") or sha[:10]
        for candidate in candidates:
            if candidate.value in content:
                record = (candidate.tag, candidate.key, sha[:10], location)
                (credential_hits if candidate.is_credential else info_hits).append(record)
        for label, pattern in GENERIC_PATTERNS:
            match = pattern.search(content)
            if match:
                # Never print the matched text, only the pattern class.
                pattern_hits.append((label.decode(), sha[:10], location))
    return credential_hits + pattern_hits, info_hits, len(pattern_hits)


def scan_worktree(candidates: list[Candidate]) -> list[tuple]:
    hits: list[tuple] = []
    exclude_dirs = {".git", ".venv", "__pycache__", "node_modules"}
    for path in REPO.rglob("*"):
        if not path.is_file() or any(part in exclude_dirs for part in path.parts):
            continue
        if path.name in {".env", "provider.txt"}:
            continue  # these are the credential sources themselves
        try:
            content = path.read_bytes()
        except OSError:
            continue
        relative = str(path.relative_to(REPO))
        for candidate in candidates:
            if candidate.is_credential and candidate.value in content:
                hits.append((candidate.tag, candidate.key, relative))
    return hits


def scan_staged(candidates: list[Candidate]) -> list[tuple]:
    """Check staged content and reject staged credential-looking filenames."""
    hits: list[tuple] = []
    names = _git("diff", "--cached", "--name-only", "-z")
    for raw in (names.stdout or "").split("\0"):
        if not raw:
            continue
        if SENSITIVE_NAME.search(raw) and not SENSITIVE_NAME_EXEMPT.search(raw):
            hits.append(("filename", raw, "credential-like file staged"))
    raw = _git("diff", "--cached", "--raw", "-z")
    tokens = (raw.stdout or "").split("\0")
    for entry in tokens:
        if not entry or not entry.startswith(":"):
            continue
        fields = entry.split()
        # ":oldmode newmode oldsha newsha status" -> new blob is fields[3]
        if len(fields) < 5:
            continue
        new_sha = fields[3]
        if set(new_sha) == {"0"}:
            continue
        blob = subprocess.run(
            ["git", "-C", str(REPO), "cat-file", "blob", new_sha],
            capture_output=True,
        )
        if blob.returncode != 0:
            continue
        for candidate in candidates:
            if candidate.is_credential and candidate.value in blob.stdout:
                hits.append((candidate.tag, candidate.key, new_sha[:10]))
        for label, pattern in GENERIC_PATTERNS:
            if pattern.search(blob.stdout):
                hits.append((label.decode(), new_sha[:10], "staged blob"))
    return hits


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--staged-only", action="store_true",
                        help="only inspect staged content (pre-commit mode)")
    args = parser.parse_args()

    candidates = load_candidates()
    credentials = [c for c in candidates if c.is_credential]
    if not candidates:
        print("ERROR: no configuration values found; cannot verify credentials.")
        return 2
    print(f"Loaded {len(candidates)} candidate values "
          f"({len(credentials)} credential-type); values are never printed.")

    bad: list[tuple] = []
    info: list[tuple] = []
    if args.staged_only:
        bad = scan_staged(candidates)
    else:
        bad, info, _ = scan_history(candidates)
        bad += scan_worktree(candidates)

    if info:
        print(f"INFO: {len(info)} non-credential value(s) appear in history "
              f"(public model names / identifiers), tags: "
              f"{sorted({row[0] for row in info})}")
    if bad:
        print(f"FAIL: {len(bad)} credential leak(s) detected:")
        for row in bad:
            print("  !!", " | ".join(row))
        return 1
    scope = "staged content" if args.staged_only else "full Git history, objects and worktree"
    print(f"OK: no credential found in {scope}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
