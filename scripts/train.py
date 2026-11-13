#!/usr/bin/env python3
"""Train or smoke-test Spot Micro in the existing Isaac Lab environment."""
from __future__ import annotations

import argparse
import copy
import hashlib
from datetime import datetime
import importlib.metadata
import json
import os
import shutil
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num-envs", type=int, default=2048)
parser.add_argument("--iterations", type=int, default=3000)
parser.add_argument("--mode", choices=("stand", "walk"), default="walk")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run-name", default="")
parser.add_argument("--resume", type=Path)
parser.add_argument("--initialize-from", type=Path, help="Warm-start weights with a fresh optimizer; only reward settings may differ")
parser.add_argument("--reset-command-curriculum", action="store_true", help="Allow new command ranges on warm start and restart their difficulty ramp")
parser.add_argument("--adapt-environment", action="store_true", help="Warm start with new terrain, disturbances and domain randomization; preserve robot, actuator and observation contracts")
parser.add_argument("--robot", type=Path, default=ROOT / "configs/spotmicro_teacher_dynamic.json")
parser.add_argument("--no-randomization", action="store_true")
parser.add_argument("--log-file", type=Path, help="Capture simulator stdout/stderr in this file")
parser.add_argument("--smoke", action="store_true", help="Short integration test, not a walking policy")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.resume and args.initialize_from:
    parser.error("Choose resume or initialize-from")
if args.reset_command_curriculum and not args.initialize_from:
    parser.error("--reset-command-curriculum requires --initialize-from")
if args.adapt_environment and not args.initialize_from:
    parser.error("--adapt-environment requires --initialize-from")
if args.num_envs < 1 or args.iterations < 1:
    parser.error("num-envs and iterations must be positive")
if args.smoke:
    args.num_envs = min(args.num_envs, 32)
    args.iterations = 2
if args.log_file:
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"[SpotMicro] Console log: {args.log_file.resolve()}", flush=True)
    console = open(args.log_file, "w", buffering=1)
    os.dup2(console.fileno(), 1)
    os.dup2(console.fileno(), 2)
app = AppLauncher(args).app

import gymnasium as gym
import torch
from spot_micro.runner import SpotOnPolicyRunner as OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from spot_micro.contract import metadata
from spot_micro.commands import command_ranges
from spot_micro.env import register
from spot_micro.env_cfg import make_env_cfg
from spot_micro.export import export_actor
from spot_micro.ppo import runner_config, teacher_config


def main():
    register()
    device = args.device or "cuda:0"
    cfg = make_env_cfg(args.robot, args.num_envs, device, args.seed,
                       args.mode, not args.no_randomization)
    if args.resume:
        saved_robot = args.resume.resolve().parent / "robot.json"
        if not saved_robot.is_file():
            raise ValueError("Resume checkpoint needs its run's robot.json")
        if json.loads(saved_robot.read_text()) != cfg.robot_parameters:
            raise ValueError("Robot config differs from checkpoint. Use --robot PATH_TO_RUN/robot.json")
    if args.initialize_from:
        saved = json.loads((args.initialize_from.resolve().parent / "robot.json").read_text())
        current = copy.deepcopy(cfg.robot_parameters)
        allowed_changes = ["reward_weights", "reward_parameters"]
        if args.reset_command_curriculum:
            allowed_changes.append("command_curriculum")
        if args.adapt_environment:
            allowed_changes += ["terrain", "domain_randomization", "disturbances", "robustness_curriculum",
                                "description", "performance_curriculum", "teleop_speeds"]
        for key in allowed_changes:
            saved.pop(key, None)
            current.pop(key, None)
        if saved != current:
            raise ValueError("Warm start may change only reward settings and explicitly reset command ranges; model, sensors and control must match")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    label = "smoke" if args.smoke else args.mode
    if args.run_name:
        label += "_" + "".join(c for c in args.run_name if c.isalnum() or c in "-_")
    log_dir = ROOT / "logs" / f"{timestamp}_{label}"
    log_dir.mkdir(parents=True)
    config_factory = teacher_config if cfg.robot_parameters.get("policy_role") == "teacher" else runner_config
    train_cfg = config_factory(args.seed, steps=16 if args.smoke else 32,
                              save_interval=1 if args.smoke else 100)
    if args.resume or args.initialize_from:
        saved_ppo = (args.resume or args.initialize_from).resolve().parent / "ppo.json"
        if not saved_ppo.is_file():
            raise ValueError("Resume checkpoint needs its run's ppo.json")
        train_cfg = json.loads(saved_ppo.read_text())
        train_cfg["seed"] = args.seed
        train_cfg["save_interval"] = 1 if args.smoke else 100
    config_record = copy.deepcopy(train_cfg)
    versions = {name: importlib.metadata.version(name) for name in ("torch", "isaacsim", "rsl-rl-lib", "onnxruntime")}
    run_info = {"seed": args.seed, "mode": args.mode, "smoke_test": args.smoke,
                "num_envs": args.num_envs, "iterations_requested": args.iterations,
                "randomization": cfg.randomize, "versions": versions,
                "resume_from": str(args.resume.resolve()) if args.resume else None,
                "initialize_from": str(args.initialize_from.resolve()) if args.initialize_from else None,
                "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else "CPU",
                "curriculum_offset": 0,
                "reward_weights": {**cfg.robot_parameters.get("reward_weights", {}), "track_velocity": cfg.lin_tracking_weight, "track_yaw": cfg.yaw_tracking_weight},
                "reward_parameters": cfg.robot_parameters.get("reward_parameters", {}),
                "tracking_tolerances": {"linear_m_s": cfg.lin_tracking_sigma, "yaw_rad_s": cfg.yaw_tracking_sigma},
                "command_curriculum": {"initial_vx_vy_wz": command_ranges(cfg.robot_parameters, 0),
                                       "final_vx_vy_wz": command_ranges(cfg.robot_parameters, 1)},
                "policy_role": cfg.robot_parameters.get("policy_role", "sensor_actor"),
                "action_distribution": "squashed_normal" if train_cfg["policy"]["class_name"] in ("SquashedActorCritic", "DirectSquashedActorCritic") else "normal",
                "noise_std_bounds_latent": [train_cfg["policy"].get("min_noise_std"), train_cfg["policy"].get("max_noise_std")],
                "robot_asset_sha256": hashlib.sha256(Path(cfg.robot.spawn.asset_path).read_bytes()).hexdigest()}
    (log_dir / "robot.json").write_text(json.dumps(cfg.robot_parameters, indent=2) + "\n")
    (log_dir / "ppo.json").write_text(json.dumps(config_record, indent=2) + "\n")
    source_manifest = {}
    for source in [*sorted((ROOT / "spot_micro").glob("*.py")), Path(__file__).resolve(), ROOT / "scripts/build_spotmicro_original.py"]:
        relative = source.relative_to(ROOT)
        destination = log_dir / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        source_manifest[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    (log_dir / "source_manifest.json").write_text(json.dumps(source_manifest, indent=2) + "\n")
    model_manifest = Path(cfg.robot.spawn.asset_path).parent / "manifest.json"
    if model_manifest.exists():
        shutil.copyfile(model_manifest, log_dir / "model_manifest.json")
    env = gym.make("SpotMicro-Velocity-Flat-v0", cfg=cfg)
    try:
        wrapper = RslRlVecEnvWrapper(env, clip_actions=1.0)
        raw = env.unwrapped
        imported_mass = raw.robot.root_physx_view.get_masses().sum(dim=1)
        run_info["imported_total_mass_kg"] = {"mean": float(imported_mass.mean()),
                                              "min": float(imported_mass.min()),
                                              "max": float(imported_mass.max())}
        print("[SpotMicro] Imported mass:", json.dumps(run_info["imported_total_mass_kg"]), flush=True)
        if cfg.robot_parameters.get("model_family") == "spotmicro_original":
            expected_mass = json.loads(model_manifest.read_text())["mass_kg"]
            # Randomized samples need not have an exactly nominal sample mean.
            tolerance = .25 if cfg.randomize else .05
            if abs(float(imported_mass.mean()) - expected_mass) > tolerance:
                raise ValueError("Imported mean mass differs from the original SpotMicro model manifest")
        if args.smoke:
            zero = torch.zeros(args.num_envs, 12, device=device)
            probe = {"control_steps": 150, "fall_resets": 0}
            heights, speeds = [], []
            raw.command_override = (0.0, 0.0, 0.0)
            wrapper.reset()
            for _ in range(probe["control_steps"]):
                obs, reward, done, extras = wrapper.step(zero)
                if not torch.isfinite(reward).all() or not all(torch.isfinite(v).all() for v in obs.values()):
                    raise RuntimeError("Non-finite values during environment smoke test")
                probe["fall_resets"] += int(raw.reset_terminated.sum().item())
                heights.append(float(raw.robot.data.root_pos_w[:, 2].mean()))
                speeds.append(float(raw.robot.data.joint_vel[:, raw.joint_ids].abs().max()))
            probe.update({"mean_base_height_m": sum(heights) / len(heights),
                          "final_base_height_m": heights[-1], "max_joint_speed_rad_s": max(speeds)})
            print("[SpotMicro] Stand probe:", json.dumps(probe), flush=True)
            (log_dir / "stand_probe.json").write_text(json.dumps(probe, indent=2) + "\n")
            raw.command_override = None
            wrapper.reset()
        runner = OnPolicyRunner(wrapper, train_cfg, log_dir=str(log_dir), device=device)
        if args.initialize_from:
            source = args.initialize_from.resolve()
            source_ppo = json.loads((source.parent / "ppo.json").read_text())
            source_ppo["seed"] = args.seed
            source_ppo["save_interval"] = 1 if args.smoke else 100
            # This controlled experiment keeps all learner settings identical.
            if source_ppo != config_record:
                raise ValueError("Warm-start PPO settings must match the source run")
            info = runner.load(str(source), load_optimizer=False, map_location=device)
            old_run = json.loads((source.parent / "run.json").read_text())
            raw.curriculum_offset = (info or {}).get("curriculum_transitions",
                old_run.get("curriculum_offset", 0) + (runner.current_learning_iteration + 1) * train_cfg["num_steps_per_env"] * old_run["num_envs"])
            if args.reset_command_curriculum:
                raw.curriculum_offset = 0
                raw.style_curriculum_offset = (info or {}).get("style_curriculum_transitions", 10_000_000)
            runner.current_learning_iteration = 0
            run_info["curriculum_offset"] = raw.curriculum_offset
            run_info["optimizer_reset"] = True
            run_info["command_curriculum_reset"] = args.reset_command_curriculum
            run_info["critic_initialization"] = "source actor/critic weights and normalizers retained; fresh optimizer"
            # Sample the first command at the restored curriculum difficulty too.
            wrapper.reset()
        if args.resume:
            checkpoint_info = runner.load(str(args.resume.resolve()), map_location=device)
            old_run_info = args.resume.resolve().parent / "run.json"
            if checkpoint_info and "curriculum_transitions" in checkpoint_info:
                raw.curriculum_offset = checkpoint_info["curriculum_transitions"]
                run_info["curriculum_offset"] = raw.curriculum_offset
            elif old_run_info.is_file():
                old = json.loads(old_run_info.read_text())
                completed = runner.current_learning_iteration * train_cfg["num_steps_per_env"] * old["num_envs"]
                raw.curriculum_offset = old.get("curriculum_offset", 0) + completed
                run_info["curriculum_offset"] = raw.curriculum_offset
            if cfg.robot_parameters.get("reward_parameters", {}).get("gait_style"):
                old = json.loads(old_run_info.read_text()) if old_run_info.is_file() else {}
                raw.style_curriculum_offset = (checkpoint_info or {}).get("style_curriculum_transitions",
                    old.get("style_curriculum_offset",0) + max(0,runner.current_learning_iteration+1-old.get("initial_learning_iteration",0))*train_cfg["num_steps_per_env"]*old.get("num_envs",args.num_envs))
            wrapper.reset()
        run_info["initial_learning_iteration"] = runner.current_learning_iteration
        run_info["style_curriculum_offset"] = raw.style_curriculum_offset
        (log_dir / "run.json").write_text(json.dumps(run_info, indent=2) + "\n")
        runner.add_git_repo_to_log(str(ROOT / "spot_micro/env.py"))
        print(f"[SpotMicro] Training run: {log_dir}", flush=True)
        torch.backends.cuda.matmul.allow_tf32 = True
        runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
        checkpoint = log_dir / "model_final.pt"
        runner.save(str(checkpoint), infos={"curriculum_transitions": raw.curriculum_offset + raw.common_step_counter * raw.num_envs,
                                           "style_curriculum_transitions": raw.style_curriculum_offset + raw.common_step_counter * raw.num_envs})
        export_info = export_actor(runner.alg.policy, cfg.robot_parameters, log_dir / "export",
                                  {**run_info, "checkpoint": str(checkpoint),
                                   "validated_on_hardware": False})
        print("[SpotMicro] Export verified:", json.dumps(export_info), flush=True)
        print(f"[SpotMicro] Done. Checkpoint: {checkpoint}", flush=True)
        if runner.writer is not None:
            runner.writer.flush()
            runner.writer.close()
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        app.close()
