import copy
import json
import math
import os
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

from spot_micro.contract import ROOT, load_robot
from spot_micro.hardware.config import JointMapping, RobotMapping, load_hardware, save_json_atomic
from spot_micro.hardware.mpu6050 import MPU6050, stationary_calibration
from spot_micro.hardware.pca9685 import PCA9685
from spot_micro.hardware.supervision import CommandReader, ControlGuard, EmergencyStop, request_stop, clear_stop


class FakeBus:
    def __init__(self):
        self.registers = {(0x68, 0x75): 0x68, (0x68, 0x3A): 1}
        self.sample = list(struct.pack(">7h", 0, 0, 8192, 0, 655, -655, 0))
        self.writes = []

    def read_byte_data(self, address, register):
        return self.registers.get((address, register), 0)

    def write_byte_data(self, address, register, value):
        self.writes.append((address, register, value))
        self.registers[address, register] = value
        if 0xFA <= register <= 0xFD:
            # Datasheet: ALL_LED is a broadcast to channel registers, not an
            # independent enable mask. This distinction matters for unused pins.
            for channel in range(16):
                self.registers[address, 0x06 + channel * 4 + register - 0xFA] = value

    def read_i2c_block_data(self, address, register, length):
        return self.sample.copy()

    def write_i2c_block_data(self, address, register, values):
        self.writes.append((address, register, list(values)))
        for offset, value in enumerate(values):
            self.registers[address, register + offset] = value


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


def example():
    return load_hardware(ROOT / "configs/hardware.example.json")


class ImuDriverTests(unittest.TestCase):
    def test_register_setup_units_and_mounting(self):
        cfg = example()["imu"]
        cfg["rotation_sensor_to_body"] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        bus, clock = FakeBus(), FakeClock()
        sensor = MPU6050(bus, cfg, clock=clock, sleep=clock.sleep)
        sensor.initialize()
        sample = sensor.read()
        np.testing.assert_allclose(sample.accel_m_s2, [0, 0, 9.81])
        np.testing.assert_allclose(sample.gyro_rad_s, [math.radians(10), math.radians(10), 0])
        self.assertAlmostEqual(sample.temperature_c, 36.53)
        self.assertEqual(bus.registers[0x68, 0x1B], 8)
        self.assertEqual(bus.registers[0x68, 0x1C], 8)
        bus.registers[0x68, 0x3A] = 0
        with self.assertRaises(TimeoutError):
            sensor.read()

    def test_level_calibration_rejects_tilt(self):
        bus = FakeBus()
        bus.sample = list(struct.pack(">7h", 0, 0, 8192, 0, 0, 0, 0))
        sensor = MPU6050(bus, example()["imu"])
        result = stationary_calibration(sensor, 200)
        self.assertTrue(result["calibrated"])
        np.testing.assert_allclose(result["accel_bias_m_s2"], 0, atol=1e-10)
        bus.sample = list(struct.pack(">7h", 8192, 0, 0, 0, 0, 0, 0))
        with self.assertRaises(ValueError):
            stationary_calibration(sensor, 200)


class PwmDriverTests(unittest.TestCase):
    def test_pulse_staging_and_broadcast_disable(self):
        bus = FakeBus()
        driver = PCA9685(bus, example()["servo_driver"], sleep=lambda _: None)
        driver.initialize()
        self.assertEqual(driver.prescale, 121)
        driver.write_pulses({3: 1500})
        # Merely staging pulses must not activate hardware.
        self.assertEqual(bus.registers[0x40, 0x09 + 3 * 4] & 0x10, 0x10)
        driver.enable()
        for channel in range(16):
            off = bus.registers[0x40, 0x09 + channel * 4]
            self.assertEqual(bool(off & 0x10), channel != 3)
        ticks = bus.registers[0x40, 0x08 + 3 * 4] + ((bus.registers[0x40, 0x09 + 3 * 4] & 0xF) << 8)
        represented_us = ticks * (driver.prescale + 1) * 1e6 / driver.oscillator_hz
        self.assertAlmostEqual(represented_us, 1500, delta=3)
        before = len(bus.writes)
        with self.assertRaises(ValueError):
            driver.write_pulses({3: 1500, 4: float("nan")})
        self.assertEqual(len(bus.writes), before)
        driver.disable_all()
        self.assertTrue(all(bus.registers[0x40, 0x09 + ch * 4] & 0x10 for ch in range(16)))
        with self.assertRaises(RuntimeError):
            driver.enable()


class MappingAndSupervisionTests(unittest.TestCase):
    def test_missing_calibration_and_reversed_servo(self):
        cfg, robot = example(), load_robot()
        with self.assertRaises(ValueError):
            RobotMapping(cfg, robot)
        for i, joint in enumerate(cfg["joints"]):
            joint["calibrated"] = True
            joint["points"] = [[robot["joint_lower_rad"][i], 2000], [robot["joint_upper_rad"][i], 1000]]
        mapping = RobotMapping(cfg, robot)
        pulses = mapping.pulses(robot["rest_positions_rad"])
        self.assertEqual(len(pulses), 12)
        first = JointMapping(cfg["joints"][0])
        self.assertGreater(first.pulse(-0.1), first.pulse(0.1))
        with self.assertRaises(ValueError):
            first.pulse(10)
        cfg["joints"][0]["points"][0][0] = -0.05
        with self.assertRaises(ValueError):
            RobotMapping(cfg, robot)

    def test_stale_commands_and_latched_stop(self):
        clock = FakeClock()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "command.json"
            reader = CommandReader(path, example()["limits"], clock=clock)
            np.testing.assert_array_equal(reader.read(), 0)
            save_json_atomic(path, {"command": [0.04, 0, 0]})
            wall = path.stat().st_mtime
            np.testing.assert_allclose(reader.read(now_wall=wall), [0.04, 0, 0])
            clock.now += 1
            # Monotonic timeout still applies if the wall clock stops or changes.
            np.testing.assert_array_equal(reader.read(now_wall=wall), 0)
            save_json_atomic(path, {"stop": True})
            with self.assertRaises(EmergencyStop):
                reader.read()
            save_json_atomic(path, {"command": [0, 0, 0]})
            with self.assertRaises(EmergencyStop):
                reader.read()

    def test_stop_marker_survives_heartbeat_and_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "command.json"
            request_stop(path)
            save_json_atomic(path, {"command": [0.04, 0, 0]})
            with self.assertRaises(EmergencyStop):
                CommandReader(path, example()["limits"]).read()
            clear_stop(path)
            np.testing.assert_array_equal(CommandReader(path, example()["limits"]).read(), 0)

    def test_cycle_and_sensor_faults_latch(self):
        guard = ControlGuard(example()["limits"])
        guard.check(1, 0.99, 0.98, [0, 0, -1])
        with self.assertRaises(EmergencyStop):
            guard.check(1, 0.8, 0.98, [0, 0, -1])
        with self.assertRaises(EmergencyStop):
            guard.check(2, 1.99, 1.98, [0, 0, -1])
        with self.assertRaises(EmergencyStop):
            ControlGuard(example()["limits"]).check(1, 1, 0.8, [0, 0, -1])
        with self.assertRaises(EmergencyStop):
            ControlGuard(example()["limits"]).check(1, 1, 1, [1, 0, 0])

    def test_reflected_imu_axes_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "hardware.json"
            cfg = example()
            cfg["imu"]["rotation_sensor_to_body"][0][0] = -1
            save_json_atomic(path, cfg)
            with self.assertRaises(ValueError):
                load_hardware(path)


if __name__ == "__main__":
    unittest.main()
