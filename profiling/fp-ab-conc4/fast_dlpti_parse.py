#!/usr/bin/env python3
"""Line-oriented adapter for DLPTI native exports from dlpti_tools 0.10."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
from typing import Iterator


PARSER_PATH = Path("/home/qinyiqun/workspace/qygpu-refactor-integration/tools/dlpti_parse.py")


def iter_json_objects_in_array(path: Path, array_key: str = "activities") -> Iterator[dict]:
    if array_key != "activities":
        raise ValueError("fast adapter only supports activities")
    inside = False
    with path.open() as stream:
        for line in stream:
            if not inside:
                if '"activities"' in line and "[" in line:
                    inside = True
                continue
            stripped = line.strip()
            if stripped.startswith("]"):
                return
            if stripped.endswith(","):
                stripped = stripped[:-1]
            if not stripped:
                continue
            yield json.loads(stripped)


def main() -> int:
    spec = importlib.util.spec_from_file_location("dlpti_parse", PARSER_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {PARSER_PATH}")
    parser = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(parser)
    parser.iter_json_objects_in_array = iter_json_objects_in_array
    return parser.main()


if __name__ == "__main__":
    raise SystemExit(main())
