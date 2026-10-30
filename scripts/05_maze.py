"""Figure 5 · Figures S7–S8: run the complete maze workflow; use --smoke for a short check."""

import sys

from flycontrol.cli import main


if __name__ == "__main__":
    main(["run", *sys.argv[1:], "--config", "maze"])
