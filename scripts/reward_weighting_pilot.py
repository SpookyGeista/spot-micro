#!/usr/bin/env python3
"""Run a matched, short reward-only fine-tuning comparison and fixed-command evaluations."""
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from datetime import datetime

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "logs/2026-11-11_06-24-18_walk_original_fast_tracking8_yaw1p4/model_final.pt"
STAMP = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
OUT = ROOT / "outputs" / f"reward_weighting_pilot_{STAMP}"
OUT.mkdir(parents=True)
WORK = ROOT / "work" / OUT.name
WORK.mkdir(parents=True)
CONFIG = ROOT / "configs/experiments" / OUT.name
CONFIG.mkdir(parents=True)
STATUS = ROOT / "work/reward_weighting_pilot_status.json"
base = json.loads((SOURCE.parent / "robot.json").read_text())
commands = {"forward_030": [.3, 0, 0], "forward_040": [.4, 0, 0],
            "yaw_060": [0, 0, .6], "stand": [0, 0, 0]}
record = {"status": "running", "pid": os.getpid(), "source": str(SOURCE),
          "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
          "out": str(OUT), "work": str(WORK), "updates_per_arm": 100,
          "num_envs": 2048, "transitions_per_arm": 100*2048*32,
          "training_seed": 42, "evaluation_seed": 2027, "commands": commands,
          "conditions": "Warm-start actor, critic and observation normalizers from identical checkpoint. Fresh optimizer in every arm. Identical PPO/entropy, domain randomization and completed command curriculum. Single training seed; pilot, not from-scratch convergence evidence.",
          "arms": {}}

def save(phase):
    record["phase"] = phase
    record["updated"] = datetime.now().isoformat()
    (OUT / "experiment.json").write_text(json.dumps(record, indent=2)+"\n")
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=2)+"\n")
    tmp.replace(STATUS)
    print(phase, flush=True)

def run(arguments, log):
    subprocess.run([sys.executable, "-u", *arguments, "--headless", "--log-file", str(log)],
                   cwd=ROOT, check=True)

def evaluate(label, checkpoint, names=None):
    results = {}
    for name in names or commands:
        save(f"evaluate:{label}:{name}")
        output = OUT / f"{label}_{name}.json"
        run(["scripts/play.py", "--checkpoint", str(checkpoint), "--num-envs", "32",
             "--steps", "1000", "--command", *map(str, commands[name]),
             "--output", str(output)], WORK / f"{label}_{name}.log")
        results[name] = json.loads(output.read_text())
    return results

try:
    save("source_evaluation")
    record["source_results"] = evaluate("source", SOURCE)
    for label, changes in [
        ("control", {}),
        ("smooth_quarter", {"target_rate": .005, "target_rate_change": .0025}),
        ("smooth_quarter_yaw3x", {"target_rate": .005, "target_rate_change": .0025, "track_yaw": 4.2}),
    ]:
        cfg = json.loads(json.dumps(base))
        cfg["reward_weights"].update(changes)
        config_path = CONFIG / f"{label}.json"
        config_path.write_text(json.dumps(cfg, indent=2)+"\n")
        console = WORK / f"train_{label}.log"
        record["arms"][label] = {"reward_weights": cfg["reward_weights"],
                                "config": str(config_path), "training_log": str(console)}
        save(f"train:{label}")
        run(["scripts/train.py", "--robot", str(config_path), "--initialize-from", str(SOURCE),
             "--num-envs", "2048", "--iterations", "100", "--seed", "42",
             "--run-name", f"reward_pilot_{STAMP}_{label}"], console)
        matches = re.findall(r"\[SpotMicro\] Training run: (.+)", console.read_text())
        if not matches:
            raise RuntimeError(f"No training run found in {console}")
        checkpoint = Path(matches[-1].strip()) / "model_final.pt"
        record["arms"][label]["checkpoint"] = str(checkpoint)
        record["arms"][label]["results"] = evaluate(label, checkpoint)
    record["status"] = "complete"
    save("complete")
except BaseException as exc:
    record["status"] = "failed"
    record["error"] = repr(exc)
    save("failed")
    raise
