"""Figure 6 · Figure S9: run the complete tripod workflow; use --smoke for a short check."""

import sys

from flycontrol.cli import main


if __name__ == "__main__":
    main(["run", *sys.argv[1:], "--config", "tripod"])
