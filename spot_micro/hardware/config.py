"""Measured mounting and joint-to-PWM calibration, separate from the policy."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile

import numpy as np


def load_hardware(path):
    cfg = json.loads(Path(path).read_text())
    if cfg.get("schema_version") != 1:
        raise ValueError("Unsupported hardware configuration version")
    if not isinstance(cfg["i2c_bus"], int) or cfg["i2c_bus"] < 0:
        raise ValueError("Invalid I2C bus")
    addresses = [cfg["imu"]["address"], cfg["servo_driver"]["address"]]
    if any(not isinstance(a, int) or not 0x08 <= a <= 0x77 for a in addresses):
        raise ValueError("I2C addresses must be valid integer 7-bit addresses")
    if addresses[0] == addresses[1]:
        raise ValueError("IMU and servo driver must have different addresses")
    rotation = np.asarray(cfg["imu"]["rotation_sensor_to_body"], dtype=float)
    if rotation.shape != (3, 3) or not np.isfinite(rotation).all():
        raise ValueError("Sensor mounting rotation must be a finite 3x3 matrix")
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6) or not np.isclose(np.linalg.det(rotation), 1, atol=1e-6):
        raise ValueError("Sensor mounting must be a proper rotation, without reflection")
    for key in ("gyro_bias_rad_s", "accel_bias_m_s2"):
        value = np.asarray(cfg["imu"][key], dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f"Invalid IMU {key}")
    channels, names = set(), set()
    if len(cfg["joints"]) != 12:
        raise ValueError("Exactly twelve joint configurations are required")
    for joint in cfg["joints"]:
        channel = joint["channel"]
        if not isinstance(channel, int) or not 0 <= channel < 16 or channel in channels:
            raise ValueError("Servo channels must be unique integers in 0..15")
        if joint["name"] in names:
            raise ValueError("Joint names must be unique")
        channels.add(channel)
        names.add(joint["name"])
        limits = np.asarray(joint["pulse_limits_us"], dtype=float)
        if limits.shape != (2,) or not np.isfinite(limits).all() or not 500 <= limits[0] < limits[1] <= 2500:
            raise ValueError("Pulse limits must be ordered and within 500..2500 us")
        if joint.get("points") is not None:
            JointMapping(joint)
    limits = cfg["limits"]
    for key in ("command_min", "command_max"):
        vector = np.asarray(limits[key], dtype=float)
        if vector.shape != (3,) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid {key}")
    if np.any(np.asarray(limits["command_min"]) > 0) or np.any(np.asarray(limits["command_max"]) < 0):
        raise ValueError("Command ranges must contain zero")
    for key in ("command_timeout_s", "max_cycle_s", "max_imu_age_s"):
        if not np.isfinite(limits[key]) or limits[key] <= 0:
            raise ValueError(f"Invalid {key}")
    if not 0 < limits["max_tilt_deg"] < 90:
        raise ValueError("max_tilt_deg must be in (0,90)")
    return cfg


class JointMapping:
    """Piecewise-linear calibration; direction is encoded by measured points."""
    def __init__(self, cfg):
        self.name = cfg["name"]
        self.channel = cfg["channel"]
        self.limits = np.asarray(cfg["pulse_limits_us"], dtype=float)
        points = np.asarray(cfg.get("points"), dtype=float)
        if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 2 or not np.isfinite(points).all():
            raise ValueError(f"{self.name}: at least two measured [model_angle_rad, pulse_us] points are required")
        if np.any(np.diff(points[:, 0]) <= 0):
            raise ValueError(f"{self.name}: model angles must be strictly increasing")
        pulse_delta = np.diff(points[:, 1])
        if not (np.all(pulse_delta > 0) or np.all(pulse_delta < 0)):
            raise ValueError(f"{self.name}: pulse mapping must be strictly monotonic")
        if np.any(points[:, 1] < self.limits[0]) or np.any(points[:, 1] > self.limits[1]):
            raise ValueError(f"{self.name}: calibration exceeds pulse limits")
        self.points = points

    def pulse(self, angle):
        if not np.isfinite(angle) or angle < self.points[0, 0] - 1e-6 or angle > self.points[-1, 0] + 1e-6:
            raise ValueError(f"{self.name}: angle {angle} outside measured range")
        return float(np.interp(angle, self.points[:, 0], self.points[:, 1]))


class RobotMapping:
    def __init__(self, hardware, robot):
        by_name = {j["name"]: j for j in hardware["joints"]}
        if set(by_name) != set(robot["joint_names"]):
            raise ValueError("Hardware joint names differ from policy joint names")
        self.joints = []
        for i, name in enumerate(robot["joint_names"]):
            if by_name[name].get("calibrated") is not True:
                raise ValueError(f"Joint has not been calibrated: {name}")
            mapping = JointMapping(by_name[name])
            reachable_min = max(robot["joint_lower_rad"][i], robot["rest_positions_rad"][i] - robot["action_scale_rad"][i])
            reachable_max = min(robot["joint_upper_rad"][i], robot["rest_positions_rad"][i] + robot["action_scale_rad"][i])
            if mapping.points[0, 0] > reachable_min + 1e-6 or mapping.points[-1, 0] < reachable_max - 1e-6:
                raise ValueError(f"{name}: measured calibration does not cover the policy's output range")
            self.joints.append(mapping)

    def pulses(self, angles):
        angles = np.asarray(angles, dtype=float)
        if angles.shape != (12,) or not np.isfinite(angles).all():
            raise ValueError("Expected twelve finite joint targets")
        # Validate every joint before handing anything to an output driver.
        return {mapping.channel: mapping.pulse(angle) for mapping, angle in zip(self.joints, angles)}


def save_json_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps(value, indent=2) + "\n")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
