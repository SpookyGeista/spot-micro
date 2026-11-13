"""Deployment contract shared by the trainer and the Raspberry Pi runtime."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FRAME_FIELDS = (
    ("command_vx_vy_wz_scaled", 3),
    ("gyro_xyz_scaled", 3),
    ("estimated_gravity_xyz", 3),
    ("previous_filtered_joint_targets_normalized", 12),
    ("phase_sin_cos", 2),
)
FRAME_SIZE = sum(size for _, size in FRAME_FIELDS)


def frame_fields(cfg):
    return FRAME_FIELDS + ((("accel_xyz_scaled", 3),) if cfg.get("observe_acceleration") else ())


def frame_size(cfg):
    return sum(size for _, size in frame_fields(cfg))

PRIVILEGED_FIELDS = (
    ("base_linear_velocity_body", 3),
    ("base_angular_velocity_body_scaled_025", 3),
    ("projected_gravity_body", 3),
    ("joint_positions_relative_to_rest", 12),
    ("joint_velocities_scaled_01", 12),
    ("base_height", 1),
    ("lower_leg_contacts", 4),
)


def load_robot(path: str | Path | None = None) -> dict:
    with open(path or ROOT / "configs/robot.json", encoding="utf-8") as f:
        cfg = json.load(f)
    for key in ("joint_names", "rest_positions_rad", "action_scale_rad", "joint_lower_rad", "joint_upper_rad"):
        if len(cfg[key]) != 12:
            raise ValueError(f"{key} must have exactly twelve entries")
    if len(set(cfg["joint_names"])) != 12:
        raise ValueError("Joint names must be unique")
    for lo, rest, hi, scale in zip(cfg["joint_lower_rad"], cfg["rest_positions_rad"], cfg["joint_upper_rad"], cfg["action_scale_rad"]):
        if not lo < rest < hi or scale <= 0:
            raise ValueError("Invalid joint bounds, rest position or action scale")
    ratio = cfg["control_dt"] / cfg["physics_dt"]
    if cfg["physics_dt"] <= 0 or ratio < 1 or abs(ratio - round(ratio)) > 1e-6:
        raise ValueError("control_dt must be a positive integer multiple of physics_dt")
    if cfg["history_length"] < 1 or cfg["max_target_rate_rad_s"] <= 0:
        raise ValueError("Invalid history length or target rate")
    return cfg


def metadata(cfg: dict) -> dict:
    teacher = cfg.get("policy_role") == "teacher"
    return {
        "format_version": 1,
        "robot": cfg,
        "policy_role": "teacher" if teacher else "sensor_actor",
        "simulation_only": teacher,
        "observation_size": frame_size(cfg) * cfg["history_length"] + (sum(size for _, size in PRIVILEGED_FIELDS) if teacher else 0),
        "observation_frame_fields": [{"name": name, "size": size} for name, size in frame_fields(cfg)],
        "privileged_fields_after_history": [{"name": name, "size": size} for name, size in PRIVILEGED_FIELDS] if teacher else [],
        "history_order": "oldest_to_newest; first frame repeated at reset",
        "coordinates": "body x forward, y left, z up; gyro rad/s; accel m/s^2, stationary upright reads +9.81 on z",
        "action": "12 normalized offsets; clamp [-1,1], map to radians, low-pass and slew-limit using robot config",
        "phase": "phase starts at zero; increment AFTER applying a policy action by dt*f when command is nonzero; reset to zero at standstill",
        "hardware_calibration_required": True,
    }
