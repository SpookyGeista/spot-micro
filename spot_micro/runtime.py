"""NumPy-only input/output processing; identical conventions to the simulator.

This module never writes to hardware. A board-specific driver must map model
joint angles to calibrated servo pulse widths.
"""
from __future__ import annotations

import math
import numpy as np

from .contract import frame_size
from .commands import gait_frequency


class PolicyRuntime:
    def __init__(self, cfg: dict):
        if cfg.get("policy_role") == "teacher":
            raise ValueError("Privileged-state teacher is simulation-only; train a sensor-only student for the Pi")
        self.cfg = cfg
        self.rest = np.asarray(cfg["rest_positions_rad"], dtype=np.float32)
        self.scale = np.asarray(cfg["action_scale_rad"], dtype=np.float32)
        self.lower = np.asarray(cfg["joint_lower_rad"], dtype=np.float32)
        self.upper = np.asarray(cfg["joint_upper_rad"], dtype=np.float32)
        self.reset()

    def reset(self):
        self.target = self.rest.copy()
        self.gravity = np.array([0, 0, -1], dtype=np.float32)
        self.phase = 0.0
        self.history = np.zeros((self.cfg["history_length"], frame_size(self.cfg)), dtype=np.float32)
        self.first = True

    def observe(self, gyro_rad_s, accel_m_s2, command) -> np.ndarray:
        gyro = np.asarray(gyro_rad_s, dtype=np.float32)
        accel = np.asarray(accel_m_s2, dtype=np.float32)
        cmd = np.asarray(command, dtype=np.float32)
        if any(x.shape != (3,) or not np.isfinite(x).all() for x in (gyro, accel, cmd)):
            raise ValueError("gyro, accel and command must be finite 3-vectors")
        dt = self.cfg["control_dt"]
        predicted = self.gravity - np.cross(gyro, self.gravity) * dt
        norm = float(np.linalg.norm(accel))
        weight = min(dt / self.cfg["gravity_filter_tau_s"], 1.0) if abs(norm - 9.81) < 2.0 else 0.0
        self.gravity = (1 - weight) * predicted - weight * accel / max(norm, 1e-6)
        self.gravity /= max(float(np.linalg.norm(self.gravity)), 1e-6)
        frame = np.concatenate((
            cmd * np.asarray(self.cfg["command_scale"], dtype=np.float32),
            np.clip(gyro, -20, 20) * self.cfg["gyro_scale"],
            self.gravity,
            (self.target - self.rest) / self.scale,
            np.asarray([math.sin(2 * math.pi * self.phase), math.cos(2 * math.pi * self.phase)], dtype=np.float32),
        )).astype(np.float32)
        if self.cfg.get("observe_acceleration"):
            limit=float(self.cfg.get("accel_clip_m_s2",100.))
            frame = np.concatenate((frame,np.clip(accel,-limit,limit)*float(self.cfg.get("accel_scale",.1)))).astype(np.float32)
        if self.first:
            self.history[:] = frame
            self.first = False
        else:
            self.history[:-1] = self.history[1:].copy()
            self.history[-1] = frame
        return np.clip(self.history.reshape(-1), -10, 10).copy()

    def apply(self, action, command) -> np.ndarray:
        action = np.asarray(action, dtype=np.float32)
        command = np.asarray(command, dtype=np.float32)
        if action.shape != (12,) or not np.isfinite(action).all():
            raise ValueError("Policy action must be a finite 12-vector")
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError("Command must be a finite 3-vector")
        dt = self.cfg["control_dt"]
        desired = np.clip(self.rest + self.scale * np.clip(action, -1, 1), self.lower, self.upper)
        alpha = 1 - math.exp(-dt / self.cfg["action_filter_tau_s"])
        max_delta = self.cfg["max_target_rate_rad_s"] * dt
        self.target = np.clip(self.target + np.clip(alpha * (desired - self.target), -max_delta, max_delta), self.lower, self.upper)
        moving = np.linalg.norm(command[:2]) > 0.025 or abs(command[2]) > 0.05
        self.phase = (self.phase + dt * gait_frequency(self.cfg,np.linalg.norm(command[:2]))) % 1.0 if moving else 0.0
        return self.target.copy()
