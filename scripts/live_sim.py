#!/usr/bin/env python3
"""Display a trained policy with live keyboard commands in Isaac Sim. Simulation only."""
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--checkpoint", type=Path, required=True)
parser.add_argument("--steps", type=int, default=0, help="0: run until the window is closed")
parser.add_argument("--snapshot", type=Path)
parser.add_argument("--log-file", type=Path)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.log_file:
    args.log_file.parent.mkdir(parents=True, exist_ok=True)
    console = args.log_file.open("w", buffering=1)
    os.dup2(console.fileno(), 1)
    os.dup2(console.fileno(), 2)
app = AppLauncher(args).app

import carb.input
import carb.windowing
import gymnasium as gym
import omni.appwindow
import omni.ui as ui
import torch
from spot_micro.runner import SpotOnPolicyRunner as OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from spot_micro.env import register
from spot_micro.env_cfg import make_env_cfg
from spot_micro.hardware.config import save_json_atomic
from spot_micro.teleop import KeyboardCommand, set_live_command


def main():
    register()
    checkpoint = args.checkpoint.resolve()
    cfg = make_env_cfg(checkpoint.parent / "robot.json", 1, args.device or "cuda:0", 2027,
                       randomize=False)
    cfg.episode_length_s = 86400  # No periodic teleport during an interactive demonstration.
    env = gym.make("SpotMicro-Velocity-Flat-v0", cfg=cfg)
    controls = KeyboardCommand(cfg.robot_parameters.get("teleop_speeds"))
    input_iface = carb.input.acquire_input_interface()
    appwindow = omni.appwindow.get_default_app_window()
    keyboard = appwindow.get_keyboard() if appwindow else None
    subscription = None
    panel = None
    capture = None
    try:
        wrapper = RslRlVecEnvWrapper(env, clip_actions=1.0)
        raw = env.unwrapped
        raw.command_override = (0.0, 0.0, 0.0)
        observations, _ = wrapper.reset()
        train_cfg = json.loads((checkpoint.parent / "ppo.json").read_text())
        runner = OnPolicyRunner(wrapper, copy.deepcopy(train_cfg), log_dir=None, device=raw.device)
        runner.load(str(checkpoint), load_optimizer=False, map_location=raw.device)
        policy = runner.get_inference_policy(device=raw.device)
        label = None
        if keyboard is not None:
            def on_key(event, *unused):
                if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                    controls.press(event.input.name)
                elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
                    controls.release(event.input.name)
                return True
            subscription = input_iface.subscribe_to_keyboard_events(keyboard, on_key)
            panel = ui.Window("Spot Micro - Tastatur", width=420, height=235)
            with panel.frame:
                with ui.VStack(spacing=5):
                    ui.Label("W/S: vor/zurueck | A/D: seitwaerts | Q/E: drehen")
                    ui.Label("Leertaste: Nullbefehl | R: Aufstellen | Esc: Beenden")
                    ui.Label("Fenster anklicken. Tasten zum Fahren gedrueckt halten.")
                    ui.Label("Tasten loslassen: stehen bleiben.")
                    label = ui.Label("Lade Messwerte ...", word_wrap=True)
        windowing = carb.windowing.acquire_windowing_interface() if appwindow else None
        print("[SpotMicro] LIVE READY: W/S, A/D, Q/E; Space zero command; R reset; Escape quit.", flush=True)
        print("[SpotMicro] Simulation only. No Raspberry Pi or servo connection.", flush=True)
        step, falls = 0, 0
        last_report = 0.0
        previous_command = None
        status_path = ROOT / "work/live_sim_status.json"
        with torch.inference_mode():
            while app.is_running() and not controls.quit_requested and (not args.steps or step < args.steps):
                start = time.monotonic()
                focused = bool(appwindow and windowing.is_window_focused(appwindow.get_window()))
                if not focused:
                    controls.clear()
                if controls.reset_requested:
                    controls.reset_requested = False
                    raw.command_override = (0.0, 0.0, 0.0)
                    observations, _ = wrapper.reset()
                command = controls.command()
                set_live_command(raw, observations, command)
                action = policy(observations)
                observations, _, _, _ = wrapper.step(action)
                if not torch.isfinite(action).all():
                    raise RuntimeError("Non-finite policy output")
                falls += int(raw.reset_terminated.sum())
                step += 1
                if start - last_report >= 1.0 or command != previous_command:
                    velocity = raw.step_metrics["velocity"][0].cpu().tolist()
                    height = float(raw.step_metrics["height"][0])
                    state = {"pid": os.getpid(), "checkpoint": str(checkpoint), "step": step,
                             "command_vx_vy_wz": command, "measured_vx_vy_wz": velocity,
                             "height_m": height, "fall_resets": falls,
                             "keyboard_focused": focused, "motor_output": False,
                             "updated_unix_s": time.time()}
                    save_json_atomic(status_path, state)
                    if label:
                        label.text = (f"Soll: vx {command[0]:+.2f} m/s, vy {command[1]:+.2f} m/s, "
                                      f"Drehen {command[2]:+.2f} rad/s\n"
                                      f"Ist: vx {velocity[0]:+.3f} m/s | Hoehe {height:.3f} m\n"
                                      f"Stuerze: {falls} | Tastatur aktiv: {focused}")
                    if command != previous_command:
                        print("[SpotMicro] Live command:", command, flush=True)
                    last_report, previous_command = start, command
                if step % 10 == 0:
                    position = raw.robot.data.root_pos_w[0].cpu().tolist()
                    raw.sim.set_camera_view((position[0] + .85, position[1] + .85, .65),
                                            (position[0], position[1], .12))
                if args.snapshot and step == 100 and not args.headless:
                    from omni.kit.viewport.utility import get_active_viewport, capture_viewport_to_file
                    args.snapshot.parent.mkdir(parents=True, exist_ok=True)
                    capture = capture_viewport_to_file(get_active_viewport(), file_path=str(args.snapshot.resolve()))
                remaining = raw.step_dt - (time.monotonic() - start)
                if remaining > 0:
                    time.sleep(remaining)
    finally:
        controls.clear()
        if subscription is not None:
            input_iface.unsubscribe_to_keyboard_events(keyboard, subscription)
        if panel:
            panel.destroy()
        env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        app.close()
