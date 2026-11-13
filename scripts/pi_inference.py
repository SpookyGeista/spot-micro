#!/usr/bin/env python3
"""Replay IMU samples through an exported policy on a Raspberry Pi. No motor I/O.

Input JSONL (one sample per 20ms):
  {"gyro_rad_s":[0,0,0],"accel_m_s2":[0,0,9.81],"command":[0.05,0,0]}
Output JSONL: model joint targets in radians, in metadata joint order.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spot_micro.runtime import PolicyRuntime

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("export_dir", type=Path)
parser.add_argument("--samples", type=Path, help="JSONL file; default reads stdin")
parser.add_argument("--benchmark", type=int, default=0, metavar="N", help="Measure inference, without any hardware access")
args = parser.parse_args()
meta = json.loads((args.export_dir / "metadata.json").read_text())
runtime = PolicyRuntime(meta["robot"])
options = ort.SessionOptions()
options.intra_op_num_threads = 1
options.inter_op_num_threads = 1
session = ort.InferenceSession(str(args.export_dir / "policy.onnx"), sess_options=options,
                              providers=["CPUExecutionProvider"])
if session.get_inputs()[0].shape[-1] != meta["observation_size"]:
    raise ValueError("ONNX input shape does not match metadata")
if args.benchmark:
    observation = runtime.observe([0, 0, 0], [0, 0, 9.81], [0, 0, 0])[None, :]
    for _ in range(50):
        session.run(None, {"observation": observation})
    times = []
    for _ in range(args.benchmark):
        start = time.perf_counter_ns()
        session.run(None, {"observation": observation})
        times.append((time.perf_counter_ns() - start) / 1e6)
    print(json.dumps({"samples": len(times), "median_ms": float(np.median(times)),
                      "p99_ms": float(np.percentile(times, 99)),
                      "control_period_ms": meta["robot"]["control_dt"] * 1000}))
else:
    stream = args.samples.open() if args.samples else sys.stdin
    try:
        for line in stream:
            if not line.strip():
                continue
            sample = json.loads(line)
            observation = runtime.observe(sample["gyro_rad_s"], sample["accel_m_s2"], sample["command"])
            action = session.run(None, {"observation": observation[None, :]})[0][0]
            targets = runtime.apply(action, sample["command"])
            print(json.dumps({"joint_names": meta["robot"]["joint_names"],
                              "target_rad": targets.tolist()}), flush=True)
    finally:
        if args.samples:
            stream.close()
