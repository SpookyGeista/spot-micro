#!/usr/bin/env python3
"""Sensor bring-up, calibration and explicit dry-run/hardware policy execution."""
from __future__ import annotations

import argparse
import contextlib
import json
import math
from pathlib import Path
import signal
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from spot_micro.contract import metadata as make_metadata
from spot_micro.hardware.config import JointMapping, RobotMapping, load_hardware, save_json_atomic
from spot_micro.hardware.mpu6050 import ImuSample, MPU6050, stationary_calibration
from spot_micro.hardware.supervision import CommandReader, ControlGuard, EmergencyStop
from spot_micro.runtime import PolicyRuntime


def open_bus(cfg):
    from smbus2 import SMBus
    return SMBus(cfg["i2c_bus"])


def open_servo_driver(bus, cfg):
    if cfg["servo_driver"]["type"] != "pca9685":
        raise ValueError("Confirm the servo driver in hardware.json; currently supported backend: pca9685")
    from spot_micro.hardware.pca9685 import PCA9685
    return PCA9685(bus, cfg["servo_driver"])


def doctor(args, cfg):
    report = {"i2c_bus": cfg["i2c_bus"], "servo_driver_type": cfg["servo_driver"]["type"],
              "imu_calibrated": cfg["imu"]["calibrated"],
              "calibrated_joints": sum(j["calibrated"] is True for j in cfg["joints"]),
              "motor_output": "not accessed"}
    if args.probe_i2c:
        # Read only the configured identities/registers; no indiscriminate bus scan.
        with open_bus(cfg) as bus:
            report["mpu6050_who_am_i"] = hex(bus.read_byte_data(cfg["imu"]["address"], 0x75))
            if cfg["servo_driver"]["type"] == "pca9685":
                report["pca9685_mode1"] = hex(bus.read_byte_data(cfg["servo_driver"]["address"], 0x00))
    print(json.dumps(report, indent=2))


def read_imu(args, cfg):
    deadline = time.monotonic() + args.seconds
    stream = args.output.open("w") if args.output else sys.stdout
    try:
        with open_bus(cfg) as bus:
            sensor = MPU6050(bus, cfg["imu"])
            sensor.initialize()
            while time.monotonic() < deadline:
                sample = sensor.read()
                stream.write(json.dumps({"timestamp_monotonic": sample.timestamp,
                                         "gyro_rad_s": sample.gyro_rad_s.tolist(),
                                         "accel_m_s2": sample.accel_m_s2.tolist(),
                                         "temperature_c": sample.temperature_c}) + "\n")
                stream.flush()
                time.sleep(0.02)
    finally:
        if args.output:
            stream.close()


def calibrate_imu(args, cfg):
    with open_bus(cfg) as bus:
        sensor = MPU6050(bus, cfg["imu"])
        sensor.initialize()
        result = stationary_calibration(sensor, args.samples)
    cfg["imu"].update(result)
    save_json_atomic(args.output, cfg)
    print(json.dumps({"saved": str(args.output), "imu": result}, indent=2))


def add_joint_points(args, cfg):
    joint = next((j for j in cfg["joints"] if j["name"] == args.joint), None)
    if joint is None:
        raise ValueError("Unknown joint name")
    # Replace this joint's complete calibration; never infer endpoints from a name.
    joint["points"] = sorted(args.point, key=lambda p: p[0])
    JointMapping(joint)
    joint["calibrated"] = True
    save_json_atomic(args.output, cfg)
    print(json.dumps({"saved": str(args.output), "joint": joint}, indent=2))


def pulse(args, cfg):
    if not args.enable_servos:
        raise ValueError("pulse requires --enable-servos; it is an actual single-servo movement")
    joint = next((j for j in cfg["joints"] if j["name"] == args.joint), None)
    if joint is None:
        raise ValueError("Unknown joint name")
    if not math.isfinite(args.pulse_us) or not joint["pulse_limits_us"][0] <= args.pulse_us <= joint["pulse_limits_us"][1]:
        raise ValueError("Pulse exceeds this joint's configured pulse limits")
    if not 0 < args.seconds <= 10:
        raise ValueError("Single-servo pulse duration must be in (0,10] seconds")
    with open_bus(cfg) as bus:
        driver = open_servo_driver(bus, cfg)
        try:
            driver.initialize()
            driver.write_pulses({joint["channel"]: args.pulse_us})
            driver.enable()
            time.sleep(args.seconds)
        finally:
            driver.disable_all()


def run_policy(args, cfg):
    import onnxruntime as ort

    info = json.loads((args.export_dir / "metadata.json").read_text())
    robot = info["robot"]
    expected = make_metadata(robot)
    for key in ("format_version", "observation_size", "observation_frame_fields", "history_order", "phase"):
        if info.get(key) != expected[key]:
            raise ValueError(f"Unsupported policy deployment contract: {key}")
    runtime = PolicyRuntime(robot)
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    session = ort.InferenceSession(str(args.export_dir / "policy.onnx"), sess_options=options,
                                  providers=["CPUExecutionProvider"])
    if session.get_inputs()[0].shape[-1] != info["observation_size"] or session.get_outputs()[0].shape[-1] != 12:
        raise ValueError("Policy ONNX dimensions differ from metadata")
    # Warm up inference before any actuator is initialized.
    for _ in range(20):
        session.run(None, {"observation": np.zeros((1, info["observation_size"]), dtype=np.float32)})
    hardware = args.enable_servos
    mapping = RobotMapping(cfg, robot) if hardware else None
    if hardware and cfg["imu"].get("calibrated") is not True:
        raise ValueError("IMU must be calibrated before motor output")
    if hardware and cfg["servo_driver"]["type"] != "pca9685":
        raise ValueError("Servo driver has not been confirmed/configured")
    guard = ControlGuard(cfg["limits"])
    commands = CommandReader(args.command_file, cfg["limits"]) if args.command_file else None
    driver = None
    count, overruns, max_inference_ms = 0, 0, 0.0
    dt = robot["control_dt"]
    deadline = time.monotonic()
    with contextlib.ExitStack() as stack:
        stream = stack.enter_context(args.output.open("w")) if args.output else sys.stdout
        bus = stack.enter_context(open_bus(cfg)) if hardware or args.real_imu else None
        sensor = MPU6050(bus, cfg["imu"]) if bus is not None else None
        if sensor:
            sensor.initialize()
        try:
            if commands:
                commands.read()  # Respect a pre-existing stop before enabling PWM.
            if hardware:
                initial_sample = sensor.read()
                norm = float(np.linalg.norm(initial_sample.accel_m_s2))
                if not 8.5 < norm < 11.0 or np.linalg.norm(initial_sample.gyro_rad_s) > 0.5:
                    raise EmergencyStop("Support the body and keep it still before enabling outputs")
                initial_gravity = -initial_sample.accel_m_s2 / norm
                if initial_gravity[2] > -math.cos(math.radians(10)):
                    raise EmergencyStop("Body must be approximately level before enabling outputs")
                driver = open_servo_driver(bus, cfg)
                driver.initialize()
                # MG996R does not report its initial position. The first rest target
                # is an explicit operation to perform with the robot supported.
                driver.write_pulses(mapping.pulses(runtime.target))
                driver.enable()
            # Device/inference initialization is outside the timed control loop.
            deadline = time.monotonic()
            end = deadline + args.seconds
            while time.monotonic() < end:
                start = time.monotonic()
                if hardware and start - deadline > cfg["limits"]["max_cycle_s"]:
                    raise EmergencyStop("Control loop was not scheduled in time")
                command = commands.read() if commands else np.zeros(3)
                sample = sensor.read() if sensor else ImuSample(time.monotonic(), np.zeros(3), np.array([0, 0, 9.81]), 25.0)
                guard.check(time.monotonic(), sample.timestamp, start)
                observation = runtime.observe(sample.gyro_rad_s, sample.accel_m_s2, command)
                guard.check(time.monotonic(), sample.timestamp, start, runtime.gravity)
                inference_start = time.monotonic()
                action = session.run(None, {"observation": observation[None, :]})[0][0]
                max_inference_ms = max(max_inference_ms, (time.monotonic() - inference_start) * 1000)
                targets = runtime.apply(action, command)
                guard.check(time.monotonic(), sample.timestamp, start, runtime.gravity)
                if driver:
                    driver.write_pulses(mapping.pulses(targets))
                stream.write(json.dumps({"timestamp_monotonic": sample.timestamp,
                                         "gyro_rad_s": sample.gyro_rad_s.tolist(),
                                         "accel_m_s2": sample.accel_m_s2.tolist(),
                                         "command": command.tolist(), "target_rad": targets.tolist(),
                                         "motor_output": hardware}) + "\n")
                guard.check(time.monotonic(), sample.timestamp, start, runtime.gravity)
                count += 1
                deadline += dt
                remaining = deadline - time.monotonic()
                if remaining < 0:
                    overruns += 1
                else:
                    time.sleep(remaining)
        finally:
            if driver:
                driver.disable_all()
            print(json.dumps({"cycles": count, "overruns": overruns,
                              "max_inference_ms": max_inference_ms, "motor_output": hardware}), file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware", type=Path, default=ROOT / "configs/hardware.example.json")
    commands = parser.add_subparsers(dest="action", required=True)
    p = commands.add_parser("doctor", help="Validate config; optional read-only I2C probing")
    p.add_argument("--probe-i2c", action="store_true")
    p = commands.add_parser("imu", help="Read/record MPU-6050; never accesses servo driver")
    p.add_argument("--seconds", type=float, default=5)
    p.add_argument("--output", type=Path)
    p = commands.add_parser("calibrate-imu", help="Known level, stationary body pose; saves a new config")
    p.add_argument("--samples", type=int, default=600)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("joint-points", help="Save measured angle/pulse points; no device access")
    p.add_argument("--joint", required=True)
    p.add_argument("--point", type=float, nargs=2, action="append", required=True, metavar=("MODEL_RAD", "PULSE_US"))
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("pulse", help="Drive only the selected servo for a short calibration test")
    p.add_argument("--joint", required=True)
    p.add_argument("--pulse-us", type=float, required=True)
    p.add_argument("--seconds", type=float, default=2)
    p.add_argument("--enable-servos", action="store_true")
    p = commands.add_parser("run", help="Execute policy; dry-run is independent of I2C hardware")
    p.add_argument("--export-dir", type=Path, required=True)
    output = p.add_mutually_exclusive_group(required=True)
    output.add_argument("--dry-run", action="store_true")
    output.add_argument("--enable-servos", action="store_true")
    p.add_argument("--real-imu", action="store_true", help="Use MPU-6050 even in dry-run")
    p.add_argument("--command-file", type=Path)
    p.add_argument("--seconds", type=float, default=10)
    p.add_argument("--output", type=Path)
    args = parser.parse_args()
    if hasattr(args, "seconds") and (not math.isfinite(args.seconds) or args.seconds <= 0):
        parser.error("seconds must be positive and finite")
    if hasattr(args, "output") and args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    cfg = load_hardware(args.hardware)
    # A service stop or Ctrl-C must execute the output driver's finally block.
    def terminate(signum, frame):
        raise EmergencyStop(f"Signal {signum}")
    signal.signal(signal.SIGTERM, terminate)
    actions = {"doctor": doctor, "imu": read_imu, "calibrate-imu": calibrate_imu,
               "joint-points": add_joint_points, "pulse": pulse, "run": run_policy}
    try:
        actions[args.action](args, cfg)
    except (EmergencyStop, KeyboardInterrupt) as error:
        print(f"Stopped: {error}", file=sys.stderr)
        raise SystemExit(2)
    except (ValueError, OSError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
