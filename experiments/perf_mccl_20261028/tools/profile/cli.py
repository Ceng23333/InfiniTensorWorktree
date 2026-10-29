"""CLI entrypoints for profile package."""

from __future__ import annotations

import sys
from typing import List, Optional


def main(argv: Optional[List[str]] = None) -> int:
	"""Dispatch: default to parse_mctracer; `compare` subcommand supported."""
	args = list(sys.argv[1:] if argv is None else argv)
	if args and args[0] in ("compare", "cmp"):
		from .compare import main as compare_main

		return compare_main(args[1:])
	from .parse_mctracer import main as parse_main

	return parse_main(args)


if __name__ == "__main__":
	raise SystemExit(main())
