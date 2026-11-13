"""MPU-6050: 200 Hz sampling, +/-4g and +/-500 deg/s, SI-unit output."""
from __future__ import annotations

from dataclasses import dataclass
import math
import struct
import time

import numpy as np


@dataclass
class ImuSample:
    timestamp: float
    gyro_rad_s: np.ndarray
    accel_m_s2: np.ndarray
    temperature_c: float


class MPU6050:
    def __init__(self, bus, cfg, *, sleep=time.sleep, clock=time.monotonic):
        self.bus = bus
        self.cfg = cfg
        self.address = cfg["address"]
        self.rotation = np.asarray(cfg["rotation_sensor_to_body"], dtype=float)
        self.gyro_bias = np.asarray(cfg["gyro_bias_rad_s"], dtype=float)
        self.accel_bias = np.asarray(cfg["accel_bias_m_s2"], dtype=float)
        self.sleep = sleep
        self.clock = clock

    def identity(self):
        return self.bus.read_byte_data(self.address, 0x75)

    def initialize(self):
        if self.identity() & 0x7E != 0x68:
            raise RuntimeError("Unexpected MPU-6050 WHO_AM_I value")
        # PLL gyro reference, all axes enabled; no servo driver accesses here.
        self.bus.write_byte_data(self.address, 0x6B, 0x01)
        self.bus.write_byte_data(self.address, 0x6C, 0x00)
        self.sleep(0.1)
        self.bus.write_byte_data(self.address, 0x1A, 0x03)  # gyro DLPF 42 Hz
        self.bus.write_byte_data(self.address, 0x19, 0x04)  # 1kHz / (4+1)
        self.bus.write_byte_data(self.address, 0x1B, 0x08)  # gyro +/-500 deg/s
        self.bus.write_byte_data(self.address, 0x1C, 0x08)  # accel +/-4g
        self.bus.write_byte_data(self.address, 0x38, 0x01)  # DATA_RDY

    def read_raw(self, timeout=0.05):
        deadline = self.clock() + timeout
        while not self.bus.read_byte_data(self.address, 0x3A) & 1:
            if self.clock() >= deadline:
                raise TimeoutError("MPU-6050 supplied no fresh measurement")
            self.sleep(0.0005)
        data = self.bus.read_i2c_block_data(self.address, 0x3B, 14)
        if len(data) != 14:
            raise OSError("Incomplete MPU-6050 measurement")
        ax, ay, az, temperature, gx, gy, gz = struct.unpack(">7h", bytes(data))
        accel = np.asarray([ax, ay, az], dtype=float) * (9.81 / 8192.0)
        gyro = np.asarray([gx, gy, gz], dtype=float) * (math.pi / (180.0 * 65.5))
        return self.clock(), gyro, accel, temperature / 340.0 + 36.53

    def read(self):
        timestamp, gyro, accel, temperature = self.read_raw()
        return ImuSample(timestamp, self.rotation @ (gyro - self.gyro_bias),
                         self.rotation @ (accel - self.accel_bias), temperature)


def stationary_calibration(sensor, count=600):
    """Bias at a known, level mounting pose; not a six-position scale calibration."""
    if count < 200:
        raise ValueError("Use at least 200 fresh measurements")
    gyros, accelerations = [], []
    for _ in range(count):
        _, gyro, accel, _ = sensor.read_raw()
        gyros.append(gyro)
        accelerations.append(accel)
    gyros, accelerations = np.asarray(gyros), np.asarray(accelerations)
    gyro_std = gyros.std(axis=0)
    accel_std = accelerations.std(axis=0)
    if np.linalg.norm(gyro_std) > 0.03 or np.linalg.norm(accel_std) > 0.35:
        raise ValueError("IMU moved or vibrated during calibration")
    mean_accel = accelerations.mean(axis=0)
    if not 8.5 < np.linalg.norm(mean_accel) < 11.0:
        raise ValueError("Stationary acceleration is inconsistent with gravity")
    expected_sensor_gravity = sensor.rotation.T @ np.asarray([0.0, 0.0, 9.81])
    if np.linalg.norm(mean_accel - expected_sensor_gravity) > 2.0:
        raise ValueError("Place the body level and verify sensor-axis mapping before calibration")
    return {"gyro_bias_rad_s": gyros.mean(axis=0).tolist(),
            "accel_bias_m_s2": (mean_accel - expected_sensor_gravity).tolist(),
            "calibrated": True,
            "calibration_method": "stationary_level_single_pose",
            "calibration_samples": count,
            "gyro_std_rad_s": gyro_std.tolist(), "accel_std_m_s2": accel_std.tolist()}
