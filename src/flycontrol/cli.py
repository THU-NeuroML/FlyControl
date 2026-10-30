"""Public entry point: prepare -> train -> evaluate -> probe -> analyze -> plot."""

import argparse
from pathlib import Path

from .paths import CONFIGS, read_json
from .pipeline import STAGES, initialize, execute
from .runtime import run_lock


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("list", help="List fixed experiment configurations")
    combined = commands.add_parser("combine-bodies", help="Combine four full body runs and correct the four-body family")
    combined.add_argument("--run-dirs", type=Path, nargs=4, required=True)
    combined.add_argument("--output", type=Path, required=True)
    dev = commands.add_parser("combine-development",help="Joint 22-comparison Adult/Larva family")
    dev.add_argument("--terrain",type=Path,required=True)
    dev.add_argument("--turning",type=Path,required=True)
    dev.add_argument("--output",type=Path,required=True)
    replay=commands.add_parser("replay",help="Render newly generated native trajectories using Supplementary Video numbering")
    replay.add_argument("--run-dir",type=Path,required=True)
    replay.add_argument("--video",type=int,choices=range(1,21),required=True)
    replay.add_argument("--output",type=Path,required=True)
    materials = commands.add_parser("materials",help="Export architecture SVG and actual Tables S1–S4 settings")
    materials.add_argument("--output",type=Path,required=True)
    for name in (*STAGES, "run"):
        sub = commands.add_parser(name)
        sub.add_argument("--run-dir", type=Path, required=True)
        sub.add_argument("--device", default="cpu")
        sub.add_argument("--workers", type=int, default=1)
        if name in {"prepare", "run"}:
            sub.add_argument("--config", required=True, help="Experiment name or JSON configuration")
            sub.add_argument("--smoke", action="store_true", help="Short training; not the paper protocol")
            sub.add_argument("--methods", nargs="+")
            sub.add_argument("--seeds", nargs="+", type=int)
    args = parser.parse_args(argv)
    if args.command == "list":
        for path in sorted(CONFIGS.glob("*.json")):
            spec = read_json(path)
            if not isinstance(spec,dict) or "experiment" not in spec:continue
            print(f"{spec['experiment']:12} {spec['paper']:20} {len(spec['seeds'])} seeds × {len(spec['methods'])} methods")
        return
    if args.command == "combine-bodies":
        from .bodies import combine
        with run_lock(args.output):
            combine(args.run_dirs, args.output)
        return
    if args.command == "combine-development":
        from .supplemental_analysis import combine_development
        with run_lock(args.output):combine_development(args.terrain.resolve(),args.turning.resolve(),args.output.resolve())
        return
    if args.command == "replay":
        from .media import render
        with run_lock(args.run_dir):render(args.run_dir.resolve(),args.video,args.output.resolve())
        return
    if args.command == "materials":
        from .materials import export
        export(args.output);return
    if args.workers < 1:
        parser.error("--workers must be positive")
    try:
        with run_lock(args.run_dir):
            if args.command in {"prepare", "run"}:
                initialize(args.run_dir, args.config, args.smoke, args.methods, args.seeds)
            for stage in STAGES if args.command == "run" else [args.command]:
                execute(args.run_dir, stage, device=args.device, workers=args.workers)
    except (ValueError, FileNotFoundError, RuntimeError) as error:
        parser.exit(2, f"flycontrol: {error}\n")
