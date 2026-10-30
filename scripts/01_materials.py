"""Figure 1 controller overview and Tables S1–S4 configuration records."""

import sys

from flycontrol.cli import main


if __name__ == "__main__":
    main(["materials", *sys.argv[1:]])
