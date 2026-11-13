#!/usr/bin/env python3
"""Start the long training job independently of the chat/terminal session."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=2048)
parser.add_argument("--iterations", type=int, default=3000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--no-randomization", action="store_true")
parser.add_argument("--robot", type=Path, default=ROOT / "configs/spotmicro_teacher_dynamic.json")
parser.add_argument("--resume", type=Path)
parser.add_argument("--initialize-from", type=Path)
parser.add_argument("--reset-command-curriculum", action="store_true")
parser.add_argument("--adapt-environment", action="store_true")
parser.add_argument("--evaluate-after", action="store_true", help="Run the managed training/evaluation job")
parser.add_argument("--run-name", default="full_training")
args = parser.parse_args()
if args.resume and args.initialize_from:
    parser.error("Choose resume or initialize-from")
if args.num_envs < 1 or args.iterations < 1:
    parser.error("num-envs and iterations must be positive")
work = ROOT / "work"
work.mkdir(exist_ok=True)
record_path = work / "active_training.json"
if record_path.exists():
    old = json.loads(record_path.read_text())
    proc = Path(f"/proc/{old['pid']}")
    try:
        command_line = (proc / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        if any(str(ROOT / f"scripts/{name}.py") in command_line for name in ("train", "training_job")) and (proc / "cwd").resolve() == ROOT:
            if old.get("status") != "evaluating":
                raise SystemExit(f"Training already running with PID {old['pid']}")
            print(f"Previous training finished; its evaluation continues with PID {old['pid']}", flush=True)
    except (FileNotFoundError, PermissionError):
        pass
stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
console = work / f"training_{stamp}.log"
bootstrap = work / f"training_{stamp}_launcher.log"
command = [sys.executable, "-u", str(ROOT / "scripts/train.py"),
           "--headless", "--num-envs", str(args.num_envs),
           "--iterations", str(args.iterations), "--seed", str(args.seed), "--mode", "walk",
           "--run-name", args.run_name, "--log-file", str(console)]
if args.robot:
    command.extend(["--robot", str(args.robot.resolve())])
if args.no_randomization:
    command.append("--no-randomization")
if args.resume:
    command.extend(["--resume", str(args.resume.resolve())])
if args.initialize_from:
    command.extend(["--initialize-from", str(args.initialize_from.resolve())])
if args.reset_command_curriculum:
    if not args.initialize_from:
        parser.error("--reset-command-curriculum requires --initialize-from")
    command.append("--reset-command-curriculum")
if args.adapt_environment:
    if not args.initialize_from:
        parser.error("--adapt-environment requires --initialize-from")
    command.append("--adapt-environment")
if args.evaluate_after:
    command[2] = str(ROOT / "scripts/training_job.py")
if record_path.exists():
    archive = work / "training_records"
    archive.mkdir(exist_ok=True)
    (archive / (Path(old["console_log"]).stem + ".json")).write_text(json.dumps(old, indent=2) + "\n")
with bootstrap.open("w") as output:
    process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                               stdout=output, stderr=subprocess.STDOUT,
                               start_new_session=True)
record = {"pid": process.pid, "started_local": stamp, "command": command,
          "console_log": str(console), "launcher_log": str(bootstrap),
          "num_envs": args.num_envs, "iterations": args.iterations,
          "status": "launched; verify progress in console_log"}
if args.resume or args.initialize_from:
    source = args.resume or args.initialize_from
    record["resume_from" if args.resume else "initialize_from"] = str(source.resolve())
    ppo_path = source.resolve().parent / "ppo.json"
    if ppo_path.exists():
        policy_config = json.loads(ppo_path.read_text())["policy"]
        bounded = policy_config.get("class_name") in ("SquashedActorCritic", "DirectSquashedActorCritic")
        record["action_distribution"] = "squashed_normal" if bounded else "normal"
        record["noise_std_units"] = "latent_before_tanh" if bounded else "raw_action"
        if bounded:
            record["noise_std_bounds_latent"] = [policy_config["min_noise_std"], policy_config["max_noise_std"]]
record["evaluate_after"] = args.evaluate_after
if args.robot:
    model_config = json.loads(args.robot.read_text())
    record.update(robot_config=str(args.robot.resolve()),
                  model_family=model_config.get("model_family"),
                  reward_weights=model_config.get("reward_weights"))
record_path.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
