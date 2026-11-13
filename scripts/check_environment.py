#!/usr/bin/env python3
"""Check the selected Python environment without launching Isaac Sim."""
import importlib.metadata
import importlib.util
import json
import platform
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from spot_micro.contract import load_robot, metadata

result = {"python": sys.version, "platform": platform.platform(), "packages": {}}
for package in ("torch", "isaacsim", "isaaclab", "rsl-rl-lib", "onnx", "onnxruntime"):
    try:
        result["packages"][package] = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        result["packages"][package] = None
try:
    import torch
    result["cuda_available"] = torch.cuda.is_available()
    result["gpu"] = torch.cuda.get_device_name(0) if result["cuda_available"] else None
except ImportError:
    result["cuda_available"] = False
robot = load_robot(ROOT / "configs/spotmicro_teacher_dynamic.json")
result["robot_assets_ready"] = (ROOT / robot["urdf_asset_path"]).is_file()
result["default_model_family"] = robot["model_family"]
result["teacher_input_size"] = metadata(robot)["observation_size"]
result["sensor_student_input_size"] = metadata(load_robot(ROOT / "configs/spotmicro_sensor_dynamic.json"))["observation_size"]
result["legacy_sensor_input_size"] = 23 * robot["history_length"]
print(json.dumps(result, indent=2))
if not result["cuda_available"] or not result["robot_assets_ready"] or any(v is None for v in result["packages"].values()):
    raise SystemExit(1)
