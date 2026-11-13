#!/usr/bin/env python3
"""Package Pi tools, optionally with an exported policy; no simulator dependency."""
import argparse
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("export_dir", type=Path, nargs="?")
parser.add_argument("--output", type=Path)
parser.add_argument("--validation", type=Path, help="Include the selected policy's validation report")
args = parser.parse_args()
if args.output is None:
    args.output = ROOT / "outputs" / ("spot_micro_pi.zip" if args.export_dir else "spot_micro_pi_software.zip")
if args.export_dir:
    info = json.loads((args.export_dir / "metadata.json").read_text())
    if info.get("simulation_only") or info.get("robot",{}).get("policy_role") == "teacher":
        raise ValueError("A privileged-state teacher is simulation-only. Package the sensor-only student export.")
    if info["training"].get("smoke_test"):
        print("NOTE: Packaging a smoke-test network. It is not a trained walking policy.")
args.output.parent.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(args.output, "w", zipfile.ZIP_DEFLATED) as archive:
    if args.validation:
        archive.write(args.validation, "VALIDATION.json")
    if args.export_dir:
        for name in ("policy.onnx", "metadata.json", "verification.json"):
            archive.write(args.export_dir / name, "policy/" + name)
    files = ["spot_micro/__init__.py", "spot_micro/contract.py", "spot_micro/runtime.py", "spot_micro/commands.py",
             "scripts/pi_inference.py", "scripts/pi_robot.py", "scripts/pi_command.py",
             "requirements-pi.txt", "configs/hardware.example.json", "PI_SETUP.md"]
    files.extend(str(p.relative_to(ROOT)) for p in sorted((ROOT / "spot_micro/hardware").glob("*.py")))
    for name in files:
        archive.write(ROOT / name, name)
    archive.writestr("README.txt", "Spot Micro Pi software. Start with PI_SETUP.md.\n\nNo automatic start or motor activation.\nThe example configuration is uncalibrated and leaves the servo driver unselected.\n\npython3 -m venv .venv\n.venv/bin/pip install -r requirements-pi.txt\n.venv/bin/python scripts/pi_robot.py doctor\n\nA policy is included only when this archive was built with an export directory.\n")
print(args.output.resolve())
