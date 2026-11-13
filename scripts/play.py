#!/usr/bin/env python3
"""Evaluate a checkpoint with fixed commands, record video, or probe the rest pose."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path)
parser.add_argument("--zero", action="store_true", help="Use zero policy offsets to inspect the untrained rest pose")
parser.add_argument("--num-envs", type=int, default=16)
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--command", type=float, nargs=3, default=[0.08, 0.0, 0.0], metavar=("VX", "VY", "WZ"))
parser.add_argument("--randomize", action="store_true", help="Evaluate with training domain randomization")
parser.add_argument("--seed", type=int, default=2027)
parser.add_argument("--robot", type=Path)
parser.add_argument("--video", type=Path)
parser.add_argument("--follow-camera", action="store_true", help="Follow environment zero with a world-fixed camera angle")
parser.add_argument("--overview", action="store_true", help="Frame all environment origins for a classroom demonstration")
parser.add_argument("--snapshot", type=Path)
parser.add_argument("--export-dir", type=Path)
parser.add_argument("--output", type=Path, default=ROOT / "outputs/evaluation.json")
parser.add_argument("--log-file", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.zero and args.checkpoint is None:
    parser.error("Pass --checkpoint or --zero")
if args.steps < 1 or args.num_envs < 1:
    parser.error("steps and num-envs must be positive")
if args.video or args.snapshot:
    args.enable_cameras = True
if args.log_file:
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"[SpotMicro] Console log: {args.log_file.resolve()}", flush=True)
    console = open(args.log_file, "w", buffering=1)
    os.dup2(console.fileno(), 1)
    os.dup2(console.fileno(), 2)
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
from spot_micro.runner import SpotOnPolicyRunner as OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from spot_micro.env import register
from spot_micro.env_cfg import make_env_cfg
from spot_micro.export import export_actor
from spot_micro.ppo import runner_config


def main():
    register()
    device = args.device or "cuda:0"
    robot_path = args.robot
    if args.checkpoint and robot_path is None:
        robot_path = args.checkpoint.resolve().parent / "robot.json"
    if robot_path is None:
        robot_path = ROOT / "configs/spotmicro_teacher_dynamic.json"
    cfg = make_env_cfg(robot_path, args.num_envs, device, args.seed,
                       randomize=args.randomize)
    env = gym.make("SpotMicro-Velocity-Flat-v0", cfg=cfg,
                   render_mode="rgb_array" if args.enable_cameras else None)
    writer = None
    try:
        wrapper = RslRlVecEnvWrapper(env, clip_actions=1.0)
        raw = env.unwrapped
        if args.overview:
            origins = raw.env_origins
            center = origins.mean(dim=0).cpu().tolist()
            span = float((origins.max(dim=0).values-origins.min(dim=0).values)[:2].max())
            distance = max(3.0, span * 1.2)
            raw.sim.set_camera_view((center[0]+distance*.65, center[1]+distance*.65, center[2]+distance),
                                    (center[0], center[1], center[2]))
        raw.command_override = tuple(args.command)
        observations, _ = wrapper.reset()
        if args.zero:
            policy = lambda obs: torch.zeros(args.num_envs, 12, device=device)
        else:
            ppo_path = args.checkpoint.resolve().parent / "ppo.json"
            train_cfg = json.loads(ppo_path.read_text()) if ppo_path.is_file() else runner_config()
            runner = OnPolicyRunner(wrapper, copy.deepcopy(train_cfg), log_dir=None, device=device)
            info = runner.load(str(args.checkpoint.resolve()), load_optimizer=False, map_location=device)
            if raw.style_parameters and not (info and "style_curriculum_transitions" in info):
                run_path = args.checkpoint.resolve().parent / "run.json"
                run = json.loads(run_path.read_text()) if run_path.is_file() else {}
                completed = max(0, runner.current_learning_iteration+1-run.get("initial_learning_iteration",0))
                raw.style_curriculum_offset = run.get("style_curriculum_offset",0) + completed*train_cfg["num_steps_per_env"]*run.get("num_envs",args.num_envs)
            policy = runner.get_inference_policy(device=device)
            if args.export_dir:
                export_actor(runner.alg.policy, cfg.robot_parameters, args.export_dir,
                             {"checkpoint": str(args.checkpoint.resolve()), "validated_on_hardware": False})
        if args.video:
            import imageio.v2 as imageio
            args.video.parent.mkdir(parents=True, exist_ok=True)
            writer = imageio.get_writer(str(args.video), fps=round(1 / cfg.robot_parameters["control_dt"]), macro_block_size=1)
        squared_error = torch.zeros(3, device=device)
        velocity_sum = torch.zeros(3, device=device)
        height_sum = 0.0
        fall_resets = 0
        timeouts = 0
        min_height = float("inf")
        max_torque = 0.0
        max_joint_speed = 0.0
        trace = []
        diagnostics = torch.zeros(4, device=device)
        reward_sums = {}
        style_sums = {}
        bounded = False if args.zero else getattr(runner.alg.policy, "bounded_actions", False)
        actor_std = None if args.zero else (runner.alg.policy.latent_std if bounded else runner.alg.policy.log_std.exp()).detach()
        with torch.inference_mode():
            for step in range(args.steps):
                action = policy(observations)
                clipped = action.clamp(-1, 1)
                proposed = (raw.controller.rest + raw.controller.scale * clipped).clamp(raw.controller.lower, raw.controller.upper)
                alpha = 1 - np.exp(-raw.step_dt / raw.params["action_filter_tau_s"])
                requested_delta = alpha * (proposed - raw.controller.target)
                diagnostics[0] += (action.abs() > 1).float().mean()
                diagnostics[3] += (clipped.abs() >= .98).float().mean()
                diagnostics[1] += (requested_delta.abs() > raw.params["max_target_rate_rad_s"] * raw.step_dt + 1e-7).float().mean()
                if actor_std is not None and not bounded:
                    probability = torch.special.ndtr((-1 - action) / actor_std) + torch.special.ndtr((action - 1) / actor_std)
                    diagnostics[2] += probability.mean()
                observations, rewards, dones, extras = wrapper.step(action)
                for key, value in raw.reward_terms.items():
                    reward_sums[key] = reward_sums.get(key, 0.0) + value.sum()
                for key,value in raw.style_metrics.items():
                    style_sums[key] = style_sums.get(key,0.0)+value.sum()
                if not torch.isfinite(rewards).all() or not all(torch.isfinite(x).all() for x in observations.values()):
                    raise RuntimeError("Non-finite evaluation values")
                # Read measurements before auto-reset via the environment's cached metrics.
                measured = raw.step_metrics["velocity"]
                error = measured - torch.tensor(args.command, device=device)
                squared_error += error.square().sum(dim=0)
                velocity_sum += measured.sum(dim=0)
                heights = raw.step_metrics["height"]
                height_sum += float(heights.sum())
                min_height = min(min_height, float(heights.min()))
                fall_resets += int(raw.reset_terminated.sum())
                timeouts += int(raw.reset_time_outs.sum())
                max_torque = max(max_torque, float(raw.step_metrics["torque"].abs().max()))
                max_joint_speed = max(max_joint_speed, float(raw.step_metrics["joint_speed"].abs().max()))
                if step % 10 == 0:
                    trace.append({"time_s": (step + 1) * raw.step_dt,
                                  "base_height_m": float(heights[0]),
                                  "joint_positions_rad": raw.step_metrics["joint_position"][0].cpu().tolist(),
                                  "torques_nm": raw.step_metrics["torque"][0].cpu().tolist(),
                                  "measured_velocity": measured[0].cpu().tolist()})
                if writer is not None or (args.snapshot and step == min(50, args.steps - 1)):
                    if args.follow_camera:
                        position = raw.robot.data.root_pos_w[0].cpu().tolist()
                        raw.sim.set_camera_view((position[0] + .85, position[1] + .85, .65),
                                                (position[0], position[1], .12))
                    frame = env.render()
                    # The first Replicator request can return a black warm-up frame.
                    for _ in range(16):
                        if frame.max() > 0:
                            break
                        frame = env.render()
                    if writer is not None:
                        writer.append_data(frame)
                    if args.snapshot and step == min(50, args.steps - 1):
                        import imageio.v2 as imageio
                        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
                        imageio.imwrite(args.snapshot, frame)
        count = args.steps * args.num_envs
        metrics = {"checkpoint": str(args.checkpoint) if args.checkpoint else None,
                   "policy_role": raw.params.get("policy_role", "sensor_actor"),
                   "style_curriculum_offset": raw.style_curriculum_offset,
                   "zero_policy": args.zero, "seed": args.seed, "randomization": args.randomize,
                   "steps": args.steps, "num_envs": args.num_envs, "duration_per_env_s": args.steps * raw.step_dt,
                   "command_vx_vy_wz": args.command,
                   "velocity_rmse_vx_vy_wz": torch.sqrt(squared_error / count).cpu().tolist(),
                   "mean_velocity_vx_vy_wz": (velocity_sum / count).cpu().tolist(),
                   "mean_base_height_m": height_sum / count, "min_base_height_m": min_height,
                   "fall_resets": fall_resets, "time_limit_resets": timeouts,
                   "max_torque_nm": max_torque, "max_joint_speed_rad_s": max_joint_speed,
                   "diagnostics": {"deterministic_action_clip_fraction": float(diagnostics[0] / args.steps),
                                   "near_boundary_action_fraction_abs_ge_098": float(diagnostics[3] / args.steps),
                                   "distribution": "squashed_normal" if bounded else "normal",
                                   "noise_units": "latent_before_tanh" if bounded else "raw_action",
                                   "target_rate_limited_joint_fraction": float(diagnostics[1] / args.steps),
                                   "gaussian_clip_probability_on_deterministic_eval_states": float(diagnostics[2] / args.steps) if actor_std is not None else None,
                                   "actor_std_per_joint": actor_std.cpu().tolist() if actor_std is not None else None},
                   "mean_reward_terms_per_second": {k: float(v / count) for k,v in reward_sums.items()},
                   "gait_style_metrics_mean": {k:float(v/count) for k,v in style_sums.items()},
                   "trace_env_0": trace}
        if style_sums:
            averaged = metrics["gait_style_metrics_mean"]
            metrics["gait_style_metrics"] = {
                "stance_point_speed_proxy_m_s": averaged["stance_point_speed_sum_m_s"]/max(averaged["stance_count"],1e-9),
                "swing_clearance_rmse_proxy_m": (averaged["swing_clearance_error_sum_m2"]/max(averaged["swing_count"],1e-9))**.5,
                "body_roll_pitch_rate_rms_rad_s": averaged["body_roll_pitch_rate_squared"]**.5,
                "body_vertical_speed_rms_m_s": averaged["body_vertical_speed_squared"]**.5,
                "touchdown_downward_speed_proxy_m_s": averaged["touchdown_downward_speed_sum_m_s"]/max(averaged["touchdown_count"],1e-9),
            }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(metrics, indent=2) + "\n")
        print("[SpotMicro] Evaluation:", json.dumps({k: v for k, v in metrics.items() if k != "trace_env_0"}), flush=True)
        print(f"[SpotMicro] Saved {args.output.resolve()}", flush=True)
    finally:
        if writer is not None:
            writer.close()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
