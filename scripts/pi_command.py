#!/usr/bin/env python3
"""Write local command heartbeats for pi_robot.py; Ctrl-C requests a stop."""
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spot_micro.hardware.config import load_hardware, save_json_atomic
from spot_micro.hardware.supervision import request_stop, clear_stop, stop_path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--hardware", type=Path, default=ROOT / "configs/hardware.example.json")
parser.add_argument("--file", type=Path, default=ROOT / "work/command.json")
parser.add_argument("--command", type=float, nargs=3, default=[0, 0, 0], metavar=("VX", "VY", "WZ"))
parser.add_argument("--seconds", type=float, default=10)
parser.add_argument("--stop", action="store_true")
parser.add_argument("--clear-stop", action="store_true", help="Explicitly clear the stop marker; does not start any motor")
args = parser.parse_args()
if args.stop:
    request_stop(args.file)
    raise SystemExit(0)
if args.clear_stop:
    clear_stop(args.file)
    raise SystemExit(0)
cfg = load_hardware(args.hardware)
if stop_path(args.file).exists():
    parser.error("Stop is latched. Clear it explicitly with --clear-stop before starting a new controller")
if not math.isfinite(args.seconds) or args.seconds <= 0:
    parser.error("seconds must be positive and finite")
for value, lo, hi in zip(args.command, cfg["limits"]["command_min"], cfg["limits"]["command_max"]):
    if not math.isfinite(value) or not lo <= value <= hi:
        parser.error("Command exceeds configured limits")
end = time.monotonic() + args.seconds
try:
    while time.monotonic() < end:
        save_json_atomic(args.file, {"command": args.command})
        time.sleep(0.1)
finally:
    request_stop(args.file)
