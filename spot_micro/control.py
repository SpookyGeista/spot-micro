"""Batched Torch equivalents of the Pi contract and a provisional servo model."""
from __future__ import annotations

import math
import torch

from .contract import frame_size
from .commands import gait_frequency


class BatchedController:
    def __init__(self, cfg: dict, count: int, device: str):
        self.cfg = cfg
        self.device = device
        self.count = count
        self.rest = torch.tensor(cfg["rest_positions_rad"], device=device)
        self.scale = torch.tensor(cfg["action_scale_rad"], device=device)
        self.lower = torch.tensor(cfg["joint_lower_rad"], device=device)
        self.upper = torch.tensor(cfg["joint_upper_rad"], device=device)
        self.target = self.rest.repeat(count, 1)
        self.gravity = torch.zeros(count, 3, device=device)
        self.phase = torch.zeros(count, device=device)
        self.history = torch.zeros(count, cfg["history_length"], frame_size(cfg), device=device)
        self.first = torch.ones(count, dtype=torch.bool, device=device)
        self.reset(torch.arange(count, device=device))

    def reset(self, ids):
        self.target[ids] = self.rest
        self.gravity[ids] = torch.tensor([0.0, 0.0, -1.0], device=self.device)
        self.phase[ids] = 0
        self.history[ids] = 0
        self.first[ids] = True

    def observe(self, gyro, accel, command):
        dt = self.cfg["control_dt"]
        predicted = self.gravity - torch.linalg.cross(gyro, self.gravity, dim=-1) * dt
        norm = accel.norm(dim=-1, keepdim=True)
        weight = (torch.abs(norm - 9.81) < 2.0).float() * min(dt / self.cfg["gravity_filter_tau_s"], 1.0)
        self.gravity[:] = (1 - weight) * predicted - weight * accel / norm.clamp(min=1e-6)
        self.gravity[:] /= self.gravity.norm(dim=-1, keepdim=True).clamp(min=1e-6)
        frame = torch.cat((
            command * torch.tensor(self.cfg["command_scale"], device=self.device),
            gyro.clamp(-20, 20) * self.cfg["gyro_scale"],
            self.gravity,
            (self.target - self.rest) / self.scale,
            torch.stack((torch.sin(2 * math.pi * self.phase), torch.cos(2 * math.pi * self.phase)), dim=-1),
        ), dim=-1)
        if self.cfg.get("observe_acceleration"):
            limit=float(self.cfg.get("accel_clip_m_s2",100.))
            frame = torch.cat((frame,accel.clamp(-limit,limit)*float(self.cfg.get("accel_scale",.1))),dim=-1)
        self.history[:, :-1] = self.history[:, 1:].clone()
        self.history[:, -1] = frame
        self.history[self.first] = frame[self.first, None, :]
        self.first[:] = False
        return self.history.flatten(1).clamp(-10, 10).clone()

    def apply(self, action, command):
        dt = self.cfg["control_dt"]
        desired = (self.rest + self.scale * action.clamp(-1, 1)).clamp(self.lower, self.upper)
        alpha = 1 - math.exp(-dt / self.cfg["action_filter_tau_s"])
        max_delta = self.cfg["max_target_rate_rad_s"] * dt
        self.target[:] = (self.target + (alpha * (desired - self.target)).clamp(-max_delta, max_delta)).clamp(self.lower, self.upper)
        moving = (command[:, :2].norm(dim=-1) > 0.025) | (command[:, 2].abs() > 0.05)
        frequency = gait_frequency(self.cfg, command[:,:2].norm(dim=-1))
        self.phase[:] = torch.where(moving, (self.phase + dt * frequency) % 1.0, 0.0)
        return self.target


class ServoDynamics:
    """Delayed position servo with deadband, bias and torque/speed envelope.

    MG996R nominal stall torque/speed at 6V are datasheet values. All other
    parameters are hypotheses to identify on the actual robot. No thermal model.
    """
    def __init__(self, cfg: dict, count: int, device: str):
        self.cfg = cfg
        self.device = device
        self.count = count
        self.length = cfg["max_delay_physics_steps"] + 1
        self.buffer = torch.zeros(self.length, count, 12, device=device)
        self.cursor = 0
        self.env_ids = torch.arange(count, device=device)
        self.delay = torch.zeros(count, dtype=torch.long, device=device)
        self.strength = torch.ones(count, 12, device=device)
        self.speed = torch.ones_like(self.strength)
        self.gain = torch.ones_like(self.strength)
        self.bias = torch.zeros_like(self.strength)
        self.last_torque = torch.zeros_like(self.strength)

    def reset(self, ids, position, randomize=True):
        self.buffer[:, ids] = position.unsqueeze(0)
        self.last_torque[ids] = 0
        self.delay[ids] = torch.randint(self.length, (len(ids),), device=self.device) if randomize else 0
        for name, key in (("strength", "strength_range"), ("speed", "speed_range"), ("gain", "gain_range")):
            tensor = getattr(self, name)
            tensor[ids] = torch.empty(len(ids), 12, device=self.device).uniform_(*self.cfg[key]) if randomize else 1.0
        self.bias[ids] = torch.empty(len(ids), 12, device=self.device).uniform_(-self.cfg["position_bias_rad"], self.cfg["position_bias_rad"]) if randomize else 0

    def compute(self, target, position, velocity):
        self.buffer[self.cursor] = target
        delayed = self.buffer[(self.cursor - self.delay) % self.length, self.env_ids]
        self.cursor = (self.cursor + 1) % self.length
        error = delayed + self.bias - position
        # Approximate backlash/deadband without injecting discontinuous full-P effort.
        error = torch.sign(error) * (error.abs() - self.cfg["deadband_rad"]).clamp(min=0)
        effort = self.cfg["kp_nm_rad"] * self.gain * error - self.cfg["kd_nm_s_rad"] * velocity
        stall = self.cfg["stall_torque_nm"] * self.strength
        cap = self.cfg["torque_cap_nm"] * self.strength
        max_speed = self.cfg["no_load_speed_rad_s"] * self.speed
        top = torch.minimum(stall * (1 - velocity / max_speed), cap).clamp(min=0)
        bottom = torch.maximum(stall * (-1 - velocity / max_speed), -cap).clamp(max=0)
        self.last_torque[:] = effort.clamp(bottom, top)
        return self.last_torque
