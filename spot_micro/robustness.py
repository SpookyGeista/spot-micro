"""Physical push pulses and a checkpointable, performance-gated curriculum."""
import math
import torch


class RobustnessCurriculum:
    def __init__(self, settings):
        self.settings = settings
        self.strength = float(settings.get("initial_strength", .2))
        self.last_check = 0
        self.episodes = self.survived = self.samples = self.error_sum = 0.
        self.last_error = self.last_survival = None

    def observe(self, episodes, survived, samples, error_sum, transitions):
        self.episodes += episodes
        self.survived += survived
        self.samples += samples
        self.error_sum += error_sum
        if (transitions-self.last_check < self.settings.get("min_transitions", 2_000_000)
                or self.episodes < 128 or self.samples < 5000):
            return False
        self.last_error = self.error_sum/self.samples
        self.last_survival = self.survived/self.episodes
        advance = (self.last_error <= self.settings.get("tracking_error_m_s", .17)
                   and self.last_survival >= self.settings.get("survival_fraction", .85))
        if advance:
            self.strength = min(1., self.strength+self.settings.get("increment", .2))
        self.last_check = transitions
        self.episodes = self.survived = self.samples = self.error_sum = 0.
        return advance

    def state_dict(self):
        return {k: v for k, v in self.__dict__.items() if k != "settings"}

    def load_state_dict(self, state):
        for key in self.state_dict():
            if key in state:
                setattr(self, key, state[key])


class PushPulses:
    """World-frame forces in N and torques in Nm, held for a finite duration.

    Force magnitude scales with each simulated robot's actual weight. No velocity
    is overwritten. The returned buffers must be applied at every physics step.
    """
    def __init__(self, settings, masses, dt):
        self.settings, self.masses, self.dt = settings, masses, dt
        self.device = masses.device
        self.force = torch.zeros(len(masses), 3, device=self.device)
        self.torque = torch.zeros_like(self.force)
        self.timer = torch.zeros_like(masses)
        self.remaining = torch.zeros_like(masses)
        self.count = torch.zeros_like(masses, dtype=torch.long)
        self.reset(torch.arange(len(masses), device=self.device))

    def reset(self, ids):
        self.force[ids] = 0
        self.torque[ids] = 0
        self.remaining[ids] = 0
        self.timer[ids] = torch.empty(len(ids), device=self.device).uniform_(*self.settings.get("interval_s", [2., 5.]))

    def inject(self, ids, force, torque, duration_s):
        self.force[ids] = force
        self.torque[ids] = torque
        self.remaining[ids] = max(self.dt, duration_s)
        self.count[ids] += 1

    def step(self, strength=1., random_events=True):
        expired = self.remaining <= 1e-6
        self.force[expired] = 0
        self.torque[expired] = 0
        self.timer -= self.dt
        ids = ((self.timer <= 0) & expired).nonzero(as_tuple=False).flatten()
        if random_events and strength > 0 and len(ids):
            n = len(ids)
            angle = torch.rand(n, device=self.device)*2*math.pi
            magnitude = ((.25+.75*torch.rand(n, device=self.device))*self.masses[ids]*9.81
                         *float(self.settings.get("max_force_body_weights", .75))*strength)
            self.force[ids] = torch.stack((angle.cos()*magnitude, angle.sin()*magnitude,
                (2*torch.rand(n, device=self.device)-1)*magnitude*self.settings.get("vertical_fraction", .15)), dim=-1)
            self.torque[ids] = (2*torch.rand(n, 3, device=self.device)-1)*float(self.settings.get("max_torque_nm", .35))*strength
            self.remaining[ids] = torch.empty(n, device=self.device).uniform_(*self.settings.get("duration_s", [.08, .18]))
            self.timer[ids] = torch.empty(n, device=self.device).uniform_(*self.settings.get("interval_s", [2., 5.]))
            self.count[ids] += 1
        self.remaining -= self.dt
        return self.force, self.torque


def gait_quality_terms(air_time, peak_clearance, landed, moving, weights, target=.03, width=.015, dt=.02):
    """Contact-event shaping; no fixed phase schedule or moving rest-pose cost."""
    quality = torch.exp(-((peak_clearance-target)/width).square())
    peak_reward = float(weights.get("swing_peak", 0.))*(quality*landed).sum(-1)*moving/dt
    pair_error = (air_time[:, 0]-air_time[:, 3]).square()+(air_time[:, 1]-air_time[:, 2]).square()
    pair_cost = -float(weights.get("diagonal_timing", 0.))*(pair_error/.1**2).clamp(max=4)*moving
    return {"swing_peak": peak_reward, "diagonal_timing": pair_cost}
