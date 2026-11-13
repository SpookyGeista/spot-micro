#!/usr/bin/env python3
"""Collect real teacher rollouts for action-space conversion, without hardware."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--checkpoint", type=Path, required=True)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--log-file", type=Path, required=True)
p.add_argument("--start-case", type=int, default=0)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.log_file.parent.mkdir(parents=True, exist_ok=True)
console = a.log_file.open("w", buffering=1)
os.dup2(console.fileno(),1); os.dup2(console.fileno(),2)
app = AppLauncher(a).app
import torch
import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from spot_micro.runner import SpotOnPolicyRunner
from spot_micro.env import register
from spot_micro.env_cfg import make_env_cfg

def main():
    register()
    cfg = make_env_cfg(a.checkpoint.parent/"robot.json", 128, "cuda:0", 2029, randomize=True)
    env = gym.make("SpotMicro-Velocity-Flat-v0", cfg=cfg)
    try:
        wrapper = RslRlVecEnvWrapper(env, clip_actions=1.)
        runner = SpotOnPolicyRunner(wrapper, copy.deepcopy(json.loads((a.checkpoint.parent/"ppo.json").read_text())), device="cuda:0")
        runner.load(str(a.checkpoint), load_optimizer=False, map_location="cuda:0")
        policy = runner.get_inference_policy(device="cuda:0")
        raw = env.unwrapped
        raw.curriculum_offset = cfg.curriculum_steps
        cases = [[.4,0,0],[.3,0,0],[0,0,.6],[0,0,-.6],[0,0,0],None]
        observations, targets, holdout = [], [], []
        if a.start_case:
            prior = torch.load(a.output, weights_only=False)
            if len(prior["observations"]) != a.start_case*128*125:
                raise ValueError("Partial dataset does not match requested starting case")
            observations.append(prior["observations"])
            targets.append(prior["targets"])
            holdout.append(prior["holdout"])
        split = torch.arange(128) >= 96
        def save_data():
            a.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"observations":torch.cat(observations), "targets":torch.cat(targets),
                        "holdout":torch.cat(holdout), "source":str(a.checkpoint.resolve()),
                        "seed":2029,"cases":cases,"target_clip":.99,
                        "completed_case_count":len(torch.cat(observations))//(128*125)}, a.output)
        with torch.inference_mode():
            for command in cases[a.start_case:]:
                print("Starting case", command, flush=True)
                raw.command_override = tuple(command) if command is not None else None
                obs,_ = wrapper.reset()
                for step in range(250):
                    action = policy(obs)
                    if step % 2 == 0:
                        observations.append(obs["policy"].cpu().clone())
                        targets.append(action.clamp(-.99,.99).cpu())
                        holdout.append(split)
                    obs,_,_,_ = wrapper.step(action)
                print("Collected", command, flush=True)
                save_data()
        print("Dataset saved:",a.output,flush=True)
    finally:
        env.close()
try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    raise
finally:
    app.close()
