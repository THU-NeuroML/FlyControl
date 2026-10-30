"""Figure 5b–d and S7: capacity-matched Ant flat-ground comparison."""

import sys
from flycontrol.cli import main

if __name__ == "__main__":
    main(["run", *sys.argv[1:], "--config", "ant-flat"])
