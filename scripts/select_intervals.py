"""Run the three interval-selection strategies on cached clean scores."""

import sys

from isars.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["select", *sys.argv[1:]]))
