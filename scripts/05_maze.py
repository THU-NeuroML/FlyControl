"""Figure 5e–j and S8: complete maze workflow; flat capacity control uses 05_ant_flat.py."""

import sys

from flycontrol.cli import main


if __name__ == "__main__":
    main(["run", *sys.argv[1:], "--config", "maze"])
