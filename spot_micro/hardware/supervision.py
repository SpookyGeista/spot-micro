"""Command freshness and cycle checks for the Pi control loop."""
from __future__ import annotations

import json
import math
from pathlib import Path
import time

import numpy as np

from .config import save_json_atomic


class EmergencyStop(RuntimeError):
    pass


def stop_path(command_path):
    command_path = Path(command_path)
    return command_path.with_name(command_path.name + ".stop")


def request_stop(command_path):
    # Separate marker cannot be overwritten by an existing heartbeat producer.
    save_json_atomic(stop_path(command_path), {"stop": True})


def clear_stop(command_path):
    save_json_atomic(command_path, {"command": [0, 0, 0]})
    stop_path(command_path).unlink(missing_ok=True)


class CommandReader:
    """Local file IPC. Stale commands become zero; explicit stop is latched."""
    def __init__(self, path, limits, *, clock=time.monotonic):
        self.path = Path(path)
        self.limits = limits
        self.clock = clock
        self.last_sequence = None
        self.last_seen = None
        self.last_mtime = None
        self.cached = np.zeros(3, dtype=float)
        self.stopped = False

    def read(self, now_wall=None):
        if self.stopped or stop_path(self.path).exists():
            self.stopped = True
            raise EmergencyStop("Operator stop is latched")
        now_wall = time.time() if now_wall is None else now_wall
        try:
            with self.path.open() as stream:
                # fstat refers to the same file version even during atomic replacement.
                import os
                stat = os.fstat(stream.fileno())
                payload = json.load(stream)
        except FileNotFoundError:
            return np.zeros(3)
        if payload.get("stop") is True:
            self.stopped = True
            raise EmergencyStop("Operator stop requested")
        command = np.asarray(payload.get("command"), dtype=float)
        if command.shape != (3,) or not np.isfinite(command).all():
            raise ValueError("Command file must contain a finite command [vx,vy,wz]")
        if np.any(command < np.asarray(self.limits["command_min"])) or np.any(command > np.asarray(self.limits["command_max"])):
            raise ValueError("Command exceeds configured velocity limits")
        identity = (stat.st_ino, stat.st_mtime_ns)
        if identity != self.last_mtime:
            self.last_mtime = identity
            self.last_seen = self.clock()
            self.cached = command
        wall_age = now_wall - stat.st_mtime
        monotonic_age = self.clock() - self.last_seen
        timeout = self.limits["command_timeout_s"]
        if wall_age < -1 or wall_age > timeout or monotonic_age > timeout:
            return np.zeros(3)
        return self.cached.copy()


class ControlGuard:
    def __init__(self, limits):
        self.limits = limits
        self.fault = None

    def check(self, now, sample_timestamp, cycle_start, gravity=None):
        if self.fault:
            raise EmergencyStop(self.fault)
        reason = None
        if not all(math.isfinite(v) for v in (now, sample_timestamp, cycle_start)):
            reason = "Invalid control timing"
        elif sample_timestamp > now + 0.001 or now - sample_timestamp > self.limits["max_imu_age_s"]:
            reason = "IMU measurement is stale"
        elif now - cycle_start > self.limits["max_cycle_s"]:
            reason = "Control cycle exceeded its deadline"
        elif gravity is not None:
            g = np.asarray(gravity, dtype=float)
            if g.shape != (3,) or not np.isfinite(g).all() or g[2] > -math.cos(math.radians(self.limits["max_tilt_deg"])):
                reason = "Body inclination exceeded configured limit"
        if reason:
            self.fault = reason
            raise EmergencyStop(reason)
