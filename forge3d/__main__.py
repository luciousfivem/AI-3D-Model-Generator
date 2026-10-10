"""``python -m forge3d`` opens the desktop app; ``python -m forge3d <command>`` runs the CLI."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["gui"]))
