"""Allow `python -m profile` to dispatch to cli."""

from .cli import main

raise SystemExit(main())
