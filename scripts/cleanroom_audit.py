#!/usr/bin/env python3
"""Mechanical clean-room anonymization gate for tsjit.

Scans repository source, documentation, configuration, scripts, and tests and
exits with status 1 if any local user path, private RFC1918 IP, internal
hostname, credential token, local OS username, or caller-supplied forbidden
identifier appears.
"""

from __future__ import annotations

import argparse
import getpass
import json
import os
import re
import socket
from collections.abc import Iterable
from pathlib import Path

SKIP_DIRS = frozenset(
    {
        ".git",
        ".hg",
        ".svn",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        ".mypy_cache",
        "build",
        "dist",
        "node_modules",
        ".venv",
        "venv",
    }
)

SKIP_SUFFIXES = frozenset(
    {
        ".so",
        ".dylib",
        ".dll",
        ".o",
        ".a",
        ".pyc",
        ".pyo",
        ".whl",
        ".tar",
        ".gz",
        ".zip",
        ".png",
        ".jpg",
        ".pdf",
        ".bin",
    }
)

GENERIC_USERNAMES = frozenset(
    {
        "root",
        "user",
        "runner",
        "ci",
        "admin",
        "nobody",
        "builder",
        "ubuntu",
        "debian",
        "ec2-user",
        "github",
        "vscode",
        "coder",
        "worker",
        "test",
    }
)

STRUCTURAL_RULES: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "local_unix_home_path",
        re.compile(r"(?<![A-Za-z0-9_])/(?:Users|home)/[A-Za-z0-9._-]+"),
    ),
    (
        "local_macos_var_folders",
        re.compile(r"(?<![A-Za-z0-9_])/var/folders/[A-Za-z0-9_]+/[A-Za-z0-9_-]+"),
    ),
    (
        "local_win_home_path",
        re.compile(r"[A-Za-z]:\\Users\\[A-Za-z0-9._-]+"),
    ),
    (
        "private_rfc1918_ip",
        re.compile(
            r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|"
            r"192\.168\.\d{1,3}\.\d{1,3}|"
            r"172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|"
            r"169\.254\.\d{1,3}\.\d{1,3}|"
            r"100\.(?:6[4-9]|[7-9]\d|1[01]\d|12[0-7])\.\d{1,3}\.\d{1,3})\b"
        ),
    ),
    (
        "private_hostname",
        re.compile(
            r"\b(?:dhcp-[0-9-]+\.[A-Za-z0-9.-]+|"
            r"[A-Za-z0-9-]+\.(?:internal|corp|lan|localdomain))\b",
            re.IGNORECASE,
        ),
    ),
    (
        "internal_email_or_ssh",
        re.compile(
            r"\b[A-Za-z0-9._-]+@[A-Za-z0-9.-]+\.(?:edu|corp|internal|local)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "credential_prefix",
        re.compile(r"\b(?:AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{30,}|sk-[A-Za-z0-9]{20,})\b"),
    ),
)


def _iter_repo_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(
            d for d in dirnames if d not in SKIP_DIRS and not d.endswith(".egg-info")
        )
        for fname in sorted(filenames):
            p = Path(dirpath) / fname
            if p.suffix.lower() in SKIP_SUFFIXES:
                continue
            yield p


def _runtime_forbidden_rules(
    extra_tokens: Iterable[str],
) -> list[tuple[str, re.Pattern[str]]]:
    rules: list[tuple[str, re.Pattern[str]]] = list(STRUCTURAL_RULES)

    # Dynamically detect the current local OS username without hardcoding any personal name
    local_users: set[str] = set()
    try:
        u = getpass.getuser()
        if u:
            local_users.add(u)
    except Exception:
        pass
    env_u = os.environ.get("USER") or os.environ.get("USERNAME")
    if env_u:
        local_users.add(env_u)

    for u in sorted(local_users):
        if len(u) >= 4 and u.lower() not in GENERIC_USERNAMES:
            rules.append(
                (
                    "local_os_username",
                    re.compile(rf"\b{re.escape(u)}\b", re.IGNORECASE),
                )
            )

    # Dynamically detect the current machine hostname (if non-generic) without hardcoding it
    try:
        host = socket.gethostname().strip()
    except Exception:
        host = ""
    generic_hosts = GENERIC_USERNAMES | {
        "localhost",
        "macbook",
        "linux",
        "darwin",
        "docker",
        "container",
    }
    base_host = host.split(".", 1)[0].lower() if host else ""
    if host and len(host) >= 5 and base_host not in generic_hosts:
        rules.append(
            (
                "local_machine_hostname",
                re.compile(rf"\b{re.escape(host)}\b", re.IGNORECASE),
            )
        )

    env_extra = os.environ.get("TSJIT_AUDIT_FORBIDDEN", "")
    all_extra = [t.strip() for t in env_extra.split(",") if t.strip()]
    all_extra.extend(t.strip() for t in extra_tokens if t.strip())
    for idx, tok in enumerate(all_extra):
        rules.append(
            (
                f"extra_forbidden_{idx}",
                re.compile(rf"\b{re.escape(tok)}\b", re.IGNORECASE),
            )
        )
    return rules


def scan_repository(
    root: Path,
    *,
    extra_forbidden: Iterable[str] = (),
) -> dict:
    root = root.resolve()
    rules = _runtime_forbidden_rules(extra_forbidden)
    files_scanned = 0
    findings: list[dict] = []

    for path in _iter_repo_files(root):
        try:
            raw = path.read_bytes()
        except OSError:
            continue
        if b"\x00" in raw[:4096]:
            continue
        text = raw.decode("utf-8", errors="replace")
        files_scanned += 1
        rel = str(path.relative_to(root))
        for lineno, line in enumerate(text.splitlines(), start=1):
            for rule_name, pattern in rules:
                if pattern.search(line):
                    findings.append(
                        {
                            "file": rel,
                            "line": lineno,
                            "rule": rule_name,
                        }
                    )

    return {
        "ok": len(findings) == 0,
        "root": root.name,
        "files_scanned": files_scanned,
        "finding_count": len(findings),
        "findings": findings,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Clean-room anonymization gate for tsjit.")
    parser.add_argument(
        "root",
        nargs="?",
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root directory to scan (default: repo root)",
    )
    parser.add_argument(
        "--extra-forbidden",
        action="append",
        default=[],
        help="Additional forbidden token (may be passed multiple times)",
    )
    parser.add_argument("--json", action="store_true", help="Emit JSON output")
    args = parser.parse_args(argv)

    report = scan_repository(Path(args.root), extra_forbidden=args.extra_forbidden)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "PASS" if report["ok"] else "FAIL"
        print(
            f"cleanroom_audit: {status} "
            f"(files_scanned={report['files_scanned']}, findings={report['finding_count']})"
        )
        for item in report["findings"]:
            print(f"  {item['file']}:{item['line']}: [{item['rule']}]")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
