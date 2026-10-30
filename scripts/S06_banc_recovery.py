"""Figure S6: run the complete banc-recovery workflow; use --smoke for a short check."""

import sys

from flycontrol.cli import main


if __name__ == "__main__":
    main(["run", *sys.argv[1:], "--config", "banc-recovery"])
