#!/usr/bin/env python3
"""Run training independently, record progress, then evaluate saved checkpoints."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--log-file", type=Path, required=True)
args, train_arguments = parser.parse_known_args()
console = args.log_file.resolve()
state_path = console.with_suffix(".job.json")
active_path = ROOT / "work/active_training.json"
state = {"pid": os.getpid(), "status": "starting", "console_log": str(console),
         "started_local": datetime.now().isoformat(), "hardware_validated": False}

def save(**changes):
    state.update(changes, updated_local=datetime.now().isoformat())
    tmp = state_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2)+"\n")
    tmp.replace(state_path)
    if active_path.exists():
        active = json.loads(active_path.read_text())
        if active.get("pid") == os.getpid():
            active.update(state, job_status_file=str(state_path))
            temp = active_path.with_suffix(".tmp")
            temp.write_text(json.dumps(active, indent=2)+"\n")
            temp.replace(active_path)

try:
    child = subprocess.Popen([sys.executable, "-u", str(ROOT / "scripts/train.py"),
                              *train_arguments, "--log-file", str(console)], cwd=ROOT)
    save(status="training", training_pid=child.pid)
    first_iteration = None
    while child.poll() is None:
        text = console.read_text(errors="replace") if console.exists() else ""
        run = re.findall(r"\[SpotMicro\] Training run: (.+)", text)
        progress = re.findall(r"Learning iteration (\d+)/(\d+)", text)
        changes = {}
        if run:
            changes["run_dir"] = run[-1].strip()
        if progress:
            first_iteration = int(progress[0][0]) if first_iteration is None else first_iteration
            changes.update(latest_iteration=int(progress[-1][0]),
                           additional_updates_completed=int(progress[-1][0])-first_iteration+1,
                           additional_updates_requested=int(progress[-1][1])-first_iteration)
        noise = re.findall(r"Mean action noise std:\s*([\d.]+)", text)
        if noise:
            changes["latest_mean_action_noise_std"] = float(noise[-1])
        save(**changes)
        time.sleep(5)
    if child.returncode:
        raise RuntimeError(f"Training exited with {child.returncode}; see {console}")
    matches = re.findall(r"\[SpotMicro\] Training run: (.+)", console.read_text(errors="replace"))
    if not matches:
        raise RuntimeError("Training did not report a run directory")
    run_dir = Path(matches[-1].strip())
    final_log = console.read_text(errors="replace")
    final_progress = re.findall(r"Learning iteration (\d+)/(\d+)", final_log)
    final_noise = re.findall(r"Mean action noise std:\s*([\d.]+)", final_log)
    if final_progress:
        save(latest_iteration=int(final_progress[-1][0]),
             additional_updates_completed=int(final_progress[-1][0])-int(final_progress[0][0])+1)
    if final_noise:
        save(latest_mean_action_noise_std=float(final_noise[-1]))
    output = ROOT / "outputs" / f"{run_dir.name}_evaluation"
    output.mkdir(parents=True, exist_ok=True)
    commands = {"forward_030": [.3, 0, 0], "forward_040": [.4, 0, 0],
                "yaw_positive_060": [0, 0, .6], "yaw_negative_060": [0, 0, -.6],
                "stand": [0, 0, 0]}
    robot_parameters = json.loads((run_dir/"robot.json").read_text())
    max_forward = robot_parameters.get("command_curriculum", {}).get("final_vx_vy_wz", [[0,.4]])[0][1]
    high_speed_commands = {f"forward_{speed:.1f}": [speed,0,0] for speed in (.8,1.2,1.6,2.) if speed <= max_forward}
    checkpoints = [p for p in (run_dir/"model_500.pt", run_dir/"model_1000.pt", run_dir/"model_final.pt") if p.exists()]
    if not checkpoints or checkpoints[-1].name != "model_final.pt":
        raise RuntimeError("Final checkpoint missing")
    summary = {"run_dir": str(run_dir), "conditions": "32 environments, 1000 control steps / 20 s, seed 2027; regular episode resets at 15 s",
               "results": {}, "randomized_final_results": {}}
    save(status="evaluating", run_dir=str(run_dir), evaluation_output=str(output),
         additional_updates_completed=state.get("additional_updates_requested"),
         evaluation_summary=str(output / "evaluation_summary.json"))
    for checkpoint in checkpoints:
        summary["results"][checkpoint.name] = {}
        cases = [(name, command, False) for name, command in commands.items()]
        if checkpoint.name == "model_final.pt":
            cases += [(name, command, False) for name, command in high_speed_commands.items()]
            cases += [(name, commands[name], True) for name in ("forward_040", "yaw_positive_060")]
        for name, command, randomized in cases:
            label = f"{checkpoint.stem}_{name}" + ("_randomized" if randomized else "")
            save(evaluation_case=label)
            target = output / f"{label}.json"
            arguments = [sys.executable, "-u", str(ROOT / "scripts/play.py"), "--checkpoint", str(checkpoint),
                         "--num-envs", "32", "--steps", "1000", "--seed", "2027", "--headless",
                         "--command", *map(str, command), "--output", str(target),
                         "--log-file", str(output / f"{label}.log")]
            if randomized:
                arguments.append("--randomize")
            subprocess.run(arguments, cwd=ROOT, check=True)
            metrics = json.loads(target.read_text())
            group = summary["randomized_final_results"] if randomized else summary["results"][checkpoint.name]
            group[name] = {k: v for k,v in metrics.items() if k != "trace_env_0"}
            (output / "evaluation_summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    robot_parameters = json.loads((run_dir/"robot.json").read_text())
    if robot_parameters.get("reward_parameters",{}).get("gait_style"):
        run_info = json.loads((run_dir/"run.json").read_text())
        reference = run_info.get("initialize_from") or run_info.get("resume_from")
        videos = [("new_calm_trot",run_dir/"model_final.pt")]
        if reference:
            videos.insert(0,("reference_gait",Path(reference)))
        summary["videos"] = {}
        for label,checkpoint in videos:
            save(evaluation_case=f"video:{label}")
            video = output/f"{label}_030.mp4"
            subprocess.run([sys.executable,"-u",str(ROOT/"scripts/play.py"),
                "--checkpoint",str(checkpoint),"--num-envs","1","--steps","500",
                "--command","0.3","0","0","--headless","--follow-camera",
                "--video",str(video),"--output",str(output/f"{label}_video_metrics.json"),
                "--log-file",str(output/f"{label}_video.log")],cwd=ROOT,check=True)
            summary["videos"][label] = str(video)
            (output/"evaluation_summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    save(status="completed", evaluation_case=None, finished_local=datetime.now().isoformat())
except BaseException as exc:
    save(status="failed", error=repr(exc))
    raise
