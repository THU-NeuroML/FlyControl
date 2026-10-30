#!/usr/bin/env bash
# Full fixed cohorts. This is a substantial training campaign, not a smoke test.
set -euo pipefail
reproduction_device="${1:-cuda:0}"
reproduction_workers="${2:-4}"
reproduction_output="${3:-runs/paper}"
for experiment in terrain turning speed historical-terrain development-terrain development-turning development-speed banc-sensitivity banc-recovery maze cassie go1 tripod hexapod; do
    flycontrol run --config "$experiment" --run-dir "$reproduction_output/$experiment" --device "$reproduction_device" --workers "$reproduction_workers"
done
flycontrol combine-development --terrain "$reproduction_output/development-terrain" --turning "$reproduction_output/development-turning" --output "$reproduction_output/development-combined"
flycontrol combine-bodies --run-dirs "$reproduction_output/cassie" "$reproduction_output/tripod" "$reproduction_output/go1" "$reproduction_output/hexapod" --output "$reproduction_output/bodies-combined"
flycontrol materials --output "$reproduction_output/materials"
