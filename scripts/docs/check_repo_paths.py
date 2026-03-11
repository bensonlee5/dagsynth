#!/usr/bin/env python3
"""Check repo-root Markdown path references for stale or missing targets."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOTS = (
    "README.md",
    "docs",
    "scripts/README.md",
)
REPO_PATH_PREFIXES = (
    "src/",
    "docs/",
    "scripts/",
    "configs/",
    "reference/",
    "site/",
)
REPO_PATH_EXACT = frozenset(
    {
        "README.md",
        "CHANGELOG.md",
        "AGENTS.md",
        "pyproject.toml",
        ".pre-commit-config.yaml",
    }
)
LEGACY_PATH_DENYLIST = frozenset(
    {
        "src/dagzoo/cli.py",
        "src/dagzoo/config.py",
        "src/dagzoo/meta_targets.py",
        "src/dagzoo/core/parallel_generation.py",
        "src/dagzoo/core/worker_partition.py",
        "src/dagzoo/core/generation_engine.py",
    }
)
INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")


def _iter_markdown_files(root: Path) -> Iterable[Path]:
    if root.is_file():
        if root.suffix.lower() == ".md":
            yield root
        return
    if not root.exists():
        return
    for path in root.rglob("*.md"):
        if path.is_file():
            yield path


def _iter_inline_code_spans(path: Path) -> Iterable[tuple[int, str]]:
    in_fence = False
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        for match in INLINE_CODE_RE.finditer(line):
            yield lineno, match.group(1).strip()


def _is_repo_path(candidate: str) -> bool:
    return candidate in REPO_PATH_EXACT or candidate.startswith(REPO_PATH_PREFIXES)


def _normalize_repo_path_candidate(candidate: str) -> str | None:
    stripped = candidate.strip().rstrip(".,:;")
    if any(char in stripped for char in "*?[]{}"):
        return None
    if (REPO_ROOT / stripped).exists():
        return stripped
    token = stripped.split()[0]
    if any(char in token for char in "*?[]{}"):
        return None
    return token


def _scan_file(path: Path) -> list[tuple[int, str]]:
    errors: list[tuple[int, str]] = []
    for lineno, candidate in _iter_inline_code_spans(path):
        normalized = _normalize_repo_path_candidate(candidate)
        if normalized is None or not _is_repo_path(normalized):
            continue
        if normalized in LEGACY_PATH_DENYLIST:
            errors.append((lineno, f"{normalized} (legacy path)"))
            continue
        if not (REPO_ROOT / normalized).exists():
            errors.append((lineno, f"{normalized} (missing path)"))
    return errors


def parse_args(argv: Iterable[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "roots",
        nargs="*",
        default=DEFAULT_ROOTS,
        help="Repo-relative Markdown file or directory roots to scan.",
    )
    return parser.parse_args(list(argv))


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    all_errors: list[tuple[Path, int, str]] = []
    for root_rel in args.roots:
        root = REPO_ROOT / root_rel
        for path in _iter_markdown_files(root):
            for lineno, target in _scan_file(path):
                all_errors.append((path, lineno, target))

    if all_errors:
        print("Broken repo path references found:")
        for path, lineno, target in all_errors:
            print(f"- {path.relative_to(REPO_ROOT)}:{lineno} -> {target}")
        return 1

    print("Repo path check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
