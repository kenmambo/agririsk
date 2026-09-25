"""Allow ``python -m agrik`` to run the full build pipeline."""

from .pipeline import main

if __name__ == "__main__":
    raise SystemExit(main())
