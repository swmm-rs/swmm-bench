from __future__ import annotations

from fnmatch import fnmatch
from pathlib import Path


def discover_inp_files(paths: list[str], recursive: bool, pattern: str) -> list[Path]:
    discovered: set[Path] = set()

    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if path.is_file():
            if fnmatch(path.name, pattern):
                discovered.add(path)
            continue

        if path.is_dir():
            matches = path.rglob(pattern) if recursive else path.glob(pattern)
            for match in matches:
                if match.is_file():
                    discovered.add(match.resolve())

    return sorted(discovered)
