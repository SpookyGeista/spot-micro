#!/usr/bin/env python3
"""Read-only policy/actuator audit on fresh simulated trajectories; no training."""
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
parser.add_argument("--output", type=Path, required=True)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--num-envs", type=int, default=32)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
console = (args.output / "console.log").open("w", buffering=1)
os.dup2(console.fileno(), 1)
os.dup2(console.fileno(), 2)
app = AppLauncher(args).app

import gymnasium as gym
import numpy as np
import torch
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from spot_micro.env import register
from spot_micro.env_cfg import make_env_cfg
from spot_micro.runner import SpotOnPolicyRunner

CALM = ROOT / "logs/2026-11-11_14-52-14_walk_calm_trot_style/model_final.pt"
FAST = ROOT / "logs/2026-11-11_16-05-20_walk_speed_2mps_continue/model_final.pt"


def quantiles(x):
    return dict(zip(("p10", "p50", "p90", "p99"),
                    torch.quantile(x.flatten().float(), torch.tensor([.1,.5,.9,.99])).tolist()))


def main():
    register()
    cfg = make_env_cfg(FAST.parent / "robot.json", args.num_envs, "cuda:0", 2027, randomize=False)
    env = gym.make("SpotMicro-Velocity-Flat-v0", cfg=cfg)
    raw = env.unwrapped
    wrapper = RslRlVecEnvWrapper(env, clip_actions=1.)
    runner = SpotOnPolicyRunner(wrapper, copy.deepcopy(json.loads((FAST.parent / "ppo.json").read_text())), device="cuda:0")
    # Evaluation must use fully active style weights, independent of env count.
    raw.style_curriculum_offset = 100_000_000
    results = {"conditions": {"num_envs": args.num_envs, "steps": args.steps,
                              "randomization": False, "seed": 2027,
                              "note": "Fixed-policy sensitivity tests; no retraining, no hardware."}, "cases": {}}
    if (args.output / "summary.json").exists():
        previous = json.loads((args.output / "summary.json").read_text())
        if previous["conditions"] != results["conditions"]:
            raise ValueError("Existing audit has different conditions")
        results = previous
    material = raw.robot.root_physx_view.get_material_properties().cpu()
    results["physics"] = {
        "body_names": raw.robot.body_names,
        "body_masses_kg_env0": raw.robot.root_physx_view.get_masses()[0].cpu().tolist(),
        "joint_names": raw.robot.joint_names,
        "joint_limits_rad_env0": raw.robot.data.joint_pos_limits[0].cpu().tolist(),
        "material_static_dynamic_restitution_min": material.flatten(0,1).min(0).values.tolist(),
        "material_static_dynamic_restitution_max": material.flatten(0,1).max(0).values.tolist(),
    }
    (args.output/"summary.json").write_text(json.dumps(results, indent=2)+"\n")
    cases = [("calm_040", CALM, .4, 2.5), ("fast_040", FAST, .4, 2.5),
             ("fast_080", FAST, .8, 2.5), ("fast_200", FAST, 2., 2.5),
             ("fast_080_target_rate_7", FAST, .8, 7.),
             ("calm_040_target_rate_7", CALM, .4, 7.)]
    try:
        for label, checkpoint, command, rate in cases:
            if label in results["cases"]:
                continue
            print("AUDIT CASE", label, flush=True)
            runner.load(str(checkpoint), load_optimizer=False, map_location="cuda:0")
            policy = runner.get_inference_policy(device="cuda:0")
            actor = runner.alg.policy
            raw.params["max_target_rate_rad_s"] = rate
            raw.command_override = (command, 0., 0.)
            with torch.inference_mode():
                obs, _ = wrapper.reset()
            samples, trace, velocity, contacts = [], [], [], []
            rates, target_errors, torques, joint_rates = [], [], [], []
            fallen, limited_sum = 0, 0.
            with torch.inference_mode():
                for step in range(args.steps):
                    action = policy(obs)
                    if step >= 50 and step % 5 == 0:
                        samples.append(actor.actor(actor.actor_obs_normalizer(obs["policy"])).cpu())
                    target_before = raw.controller.target.clone()
                    desired = raw.controller.rest + raw.controller.scale * action
                    delta = (1-np.exp(-raw.step_dt/raw.params["action_filter_tau_s"]))*(desired-target_before)
                    limited_sum += float((delta.abs() > rate*raw.step_dt).float().mean())
                    obs, _, _, _ = wrapper.step(action)
                    fallen += int(raw.reset_terminated.sum())
                    velocity.append(raw.step_metrics["velocity"].cpu())
                    contacts.append(raw.feet_contact.cpu().clone())
                    if step >= 50:
                        valid = ~raw.reset_buf
                        rates.append((raw.delta[valid]/raw.step_dt).cpu())
                        target_errors.append((raw.controller.target[valid]-raw.step_metrics["joint_position"][valid]).cpu())
                        torques.append(raw.step_metrics["torque"][valid].cpu())
                        joint_rates.append(raw.step_metrics["joint_speed"][valid].cpu())
                    trace.append({"time_s": (step+1)*raw.step_dt, "velocity": raw.step_metrics["velocity"][0].cpu().tolist(),
                                  "targets": raw.controller.target[0].cpu().tolist(),
                                  "positions": raw.step_metrics["joint_position"][0].cpu().tolist(),
                                  "contact": raw.feet_contact[0].cpu().tolist(), "phase": float(raw.controller.phase[0]),
                                  "reset": bool(raw.reset_buf[0])})
            logits = torch.cat(samples).clone().requires_grad_()
            processor = actor.action_postprocessor
            latent = processor.latent_mean(logits)
            dlatent = torch.autograd.grad(latent.sum(), logits, retain_graph=True)[0]
            mean_action = processor(logits)
            daction = torch.autograd.grad(mean_action.sum(), logits)[0]
            with torch.no_grad():
                torch.manual_seed(2027)
                std = actor.latent_std.cpu()
                draws = processor.action_bound*torch.tanh(latent.detach()[None] + torch.randn(64,*latent.shape)*std)
                action_noise = draws.std(0)
                desired_angle_noise = action_noise*raw.controller.scale.cpu()
                latent_score_gradient_rms = dlatent.abs()/std
            vel = torch.stack(velocity)
            contact = torch.stack(contacts).float()
            result = {"checkpoint": str(checkpoint), "command": [command,0,0], "max_target_rate_rad_s": rate,
                      "mean_velocity": vel.mean((0,1)).tolist(), "fall_resets": fallen,
                      "near_action_boundary_fraction": float((mean_action.abs() >= .98).float().mean()),
                      "raw_actor_abs": quantiles(logits.abs()), "latent_mean_jacobian_abs": quantiles(dlatent.abs()),
                      "latent_mean_jacobian_below_1e4_fraction": float((dlatent.abs()<1e-4).float().mean()),
                      "deterministic_action_jacobian_abs": quantiles(daction.abs()),
                      "score_gradient_rms_wrt_raw_mean": quantiles(latent_score_gradient_rms),
                      "latent_noise_std": std.tolist(), "action_noise_std": quantiles(action_noise),
                      "desired_angle_noise_std_rad": quantiles(desired_angle_noise),
                      "desired_angle_noise_below_deadband_fraction": float((desired_angle_noise < raw.params["servo"]["deadband_rad"]).float().mean()),
                      "target_rate_limited_fraction": limited_sum/args.steps,
                      "target_rate_abs_rad_s": quantiles(torch.cat(rates).abs()),
                      "joint_rate_abs_rad_s": quantiles(torch.cat(joint_rates).abs()),
                      "target_error_abs_rad": quantiles(torch.cat(target_errors).abs()),
                      "torque_abs_nm": quantiles(torch.cat(torques).abs()),
                      "contact_fraction_by_leg": contact.mean((0,1)).tolist(),
                      "touchdown_events_per_leg_per_s": ((contact[1:]>contact[:-1]).float().sum(0)/(args.steps*raw.step_dt)).mean(0).tolist()}
            results["cases"][label] = result
            (args.output/"summary.json").write_text(json.dumps(results, indent=2)+"\n")
            (args.output/f"{label}_trace.json").write_text(json.dumps(trace)+"\n")
            print("AUDIT RESULT", label, json.dumps(result), flush=True)
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
