"""`python -m rosenna ...`: the same CLI as the `rosenna` entry point."""
import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
