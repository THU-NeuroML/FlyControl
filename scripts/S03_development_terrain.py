"""Figure S3 · terrain comparison family: run the complete development-terrain workflow; use --smoke for a short check."""

import sys

from flycontrol.cli import main


if __name__ == "__main__":
    main(["run", *sys.argv[1:], "--config", "development-terrain"])
