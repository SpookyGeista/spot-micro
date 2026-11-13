"""Velocity-commanded locomotion with sensor-only and privileged observations."""
from __future__ import annotations

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply, quat_apply_inverse, quat_from_euler_xyz

from .control import BatchedController, ServoDynamics
from .commands import command_ranges
from .env_cfg import SpotMicroEnvCfg
from .gait_style import style_terms
from .curriculum import PerformanceCurriculum
from .robustness import PushPulses, RobustnessCurriculum, gait_quality_terms


class SpotMicroEnv(DirectRLEnv):
    cfg: SpotMicroEnvCfg

    def __init__(self, cfg, render_mode=None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self.params = cfg.robot_parameters
        self.joint_ids, names = self.robot.find_joints(self.params["joint_names"], preserve_order=True)
        if names != self.params["joint_names"]:
            raise RuntimeError(f"Unexpected model joints: {names}")
        self.legs = ["front_left", "front_right", "back_left", "back_right"]
        feet = [f"{leg}_lower_leg" for leg in self.legs]
        self.foot_ids, _ = self.robot.find_bodies(feet, preserve_order=True)
        self.contact_foot_ids, _ = self.contact_sensor.find_bodies(feet, preserve_order=True)
        self.base_ids, _ = self.contact_sensor.find_bodies("base_link")
        self.imu_id = self.robot.find_bodies("base_link")[0][0]
        self.controller = BatchedController(self.params, self.num_envs, self.device)
        self.servo = ServoDynamics(self.params["servo"], self.num_envs, self.device)
        self.commands = torch.zeros(self.num_envs, 3, device=self.device)
        self.command_timer = torch.zeros(self.num_envs, device=self.device)
        self.gyro_bias = torch.zeros_like(self.commands)
        self.accel_bias = torch.zeros_like(self.commands)
        self.previous_target = self.controller.target.clone()
        self.previous_delta = torch.zeros_like(self.previous_target)
        self.delta = torch.zeros_like(self.previous_target)
        self.feet_contact = torch.zeros(self.num_envs, 4, dtype=torch.bool, device=self.device)
        self.gait_offsets = torch.tensor([0.0, 0.5, 0.5, 0.0], device=self.device)
        self.gravity_w = torch.tensor([0, 0, -9.81], device=self.device)
        self._observation_step = -1
        self._cached_observations = None
        self.command_override = None
        self.curriculum_offset = 0
        self.total_resets = 0
        self.step_metrics = {}
        self._control_diagnostics = torch.zeros(self.num_envs, 4, device=self.device)
        self.style_parameters = self.params.get("reward_parameters", {}).get("gait_style")
        self.style_curriculum_offset = 0
        self.performance_curriculum = (PerformanceCurriculum(self.params["performance_curriculum"])
                                       if self.params.get("performance_curriculum") else None)
        self._frontier_error = torch.zeros(self.num_envs,device=self.device)
        self._frontier_count = torch.zeros_like(self._frontier_error)
        self._air_time = torch.zeros(self.num_envs,4,device=self.device)
        self._air_contact = torch.zeros(self.num_envs,4,dtype=torch.bool,device=self.device)
        self.style_metrics = {}
        self.robustness_offset = 0
        self.robustness_curriculum = (RobustnessCurriculum(self.params["robustness_curriculum"])
                                      if self.params.get("robustness_curriculum") else None)
        self._robust_error = torch.zeros(self.num_envs, device=self.device)
        self._robust_samples = torch.zeros_like(self._robust_error)
        self._swing_peak = torch.zeros(self.num_envs, 4, device=self.device)
        self._ground_step = -1
        self.terrain_curriculum_enabled = True
        self.random_pushes_enabled = cfg.randomize
        self.push_strength_override = None
        self.pushes = None
        if self.params.get("disturbances"):
            masses = self.robot.root_physx_view.get_masses().sum(-1).to(self.device)
            self.pushes = PushPulses(self.params["disturbances"], masses, self.step_dt)
            self._push_body_ids = torch.tensor([self.imu_id], dtype=torch.int32, device=self.device)
        if self.style_parameters:
            self.toe_offset = torch.tensor(self.style_parameters["toe_center_lower_leg_m"],device=self.device).expand(self.num_envs,4,3)
            self.previous_style_contact = torch.zeros(self.num_envs,4,dtype=torch.bool,device=self.device)
            self.previous_toe_vz = torch.zeros(self.num_envs,4,device=self.device)
        self._episode_sums = {name: torch.zeros(self.num_envs, device=self.device) for name in (
            "track_velocity", "track_yaw", "upright", "height", "contact_schedule", "torque", "smoothness", "posture", "vertical_motion", "termination")}
        if self.style_parameters:
            for name in ("foot_slip","foot_clearance","touchdown","body_wobble"):
                self._episode_sums[name] = torch.zeros(self.num_envs,device=self.device)
        if self.params.get("reward_parameters",{}).get("locomotion_v2"):
            for name in ("tracking_error", "air_time", "stand_contacts", "stand_motion"):
                self._episode_sums[name] = torch.zeros(self.num_envs,device=self.device)
        if self.params.get("reward_parameters",{}).get("gait_quality"):
            for name in ("swing_peak", "diagonal_timing"):
                self._episode_sums[name] = torch.zeros(self.num_envs,device=self.device)
        print(f"[SpotMicro] Policy joint order: {names}")
        actor_size = cfg.state_space if self.params.get("policy_role") == "teacher" else cfg.observation_space
        print(f"[SpotMicro] Actor {actor_size} inputs; critic {cfg.state_space}; dt={self.step_dt}s")

    @property
    def env_origins(self):
        return self.terrain.env_origins

    def _update_ground(self):
        if self._ground_step == self.common_step_counter:
            return
        data = self.robot.data
        if hasattr(self.terrain, "ground_heights"):
            toe = data.body_link_pos_w[:,self.foot_ids] + quat_apply(data.body_link_quat_w[:,self.foot_ids],self.toe_offset)
            points = torch.cat((data.root_pos_w[:,None,:], toe), dim=1)
            heights = self.terrain.ground_heights(points)
            self._ground_missed = ~torch.isfinite(heights).all(-1)
            heights = torch.where(torch.isfinite(heights), heights, self.env_origins[:,2,None])
            self._base_ground = heights[:,0]
            self._foot_ground = heights[:,1:]
        else:
            self._base_ground = self.env_origins[:,2]
            self._foot_ground = self._base_ground[:,None].expand(-1,4)
            self._ground_missed = torch.zeros(self.num_envs,dtype=torch.bool,device=self.device)
        self._ground_step = self.common_step_counter

    def _relative_base_height(self):
        self._update_ground()
        return self.robot.data.root_pos_w[:,2]-self._base_ground

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self.robot
        self.contact_sensor = ContactSensor(self.cfg.contacts)
        self.scene.sensors["contacts"] = self.contact_sensor
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.8, 0.85, 1.0))
        light.func("/World/Light", light)

    def _sample_commands(self, ids):
        if self.command_override is not None:
            self.commands[ids] = torch.as_tensor(self.command_override, device=self.device, dtype=self.commands.dtype)
        elif self.cfg.task_mode == "stand":
            self.commands[ids] = 0
        else:
            # Increase difficulty by total transitions, independent of number of environments.
            progress = min((self.common_step_counter * self.num_envs + self.curriculum_offset) / self.cfg.curriculum_steps, 1.0)
            for axis, (low, high) in enumerate(command_ranges(self.params, progress)):
                if self.performance_curriculum and axis == 0:
                    high = self.performance_curriculum.maximum
                self.commands[ids, axis] = torch.empty(len(ids), device=self.device).uniform_(low, high)
            standing = torch.rand(len(ids), device=self.device) < 0.20
            self.commands[ids[standing]] = 0
        self.command_timer[ids] = self.cfg.command_resample_s

    def _pre_physics_step(self, actions):
        self.command_timer -= self.step_dt
        self.previous_target[:] = self.controller.target
        self.previous_delta[:] = self.delta
        # Inputs have already been clipped by the RSL wrapper. Count boundary
        # actions, not strictly out-of-range raw Gaussian samples.
        desired = (self.controller.rest + self.controller.scale * actions.clamp(-1, 1)).clamp(self.controller.lower, self.controller.upper)
        alpha = 1 - math.exp(-self.step_dt / self.params["action_filter_tau_s"])
        requested_delta = alpha * (desired - self.controller.target)
        self._control_diagnostics[:, 0] += (actions.abs() >= 1 - 1e-6).float().mean(dim=-1)
        self._control_diagnostics[:, 1] += (requested_delta.abs() > self.params["max_target_rate_rad_s"] * self.step_dt + 1e-7).float().mean(dim=-1)
        self._control_diagnostics[:, 2] += (actions.clamp(-1,1).abs() >= .98).float().mean(dim=-1)
        self._control_diagnostics[:, 3] += 1
        self.controller.apply(actions, self.commands)
        self.delta[:] = self.controller.target - self.previous_target
        if self.pushes is not None:
            strength = self.robustness_curriculum.strength if self.robustness_curriculum else 1.
            if self.push_strength_override is not None:
                strength = self.push_strength_override
            self.pushes.step(strength, self.random_pushes_enabled)
            self.robot.permanent_wrench_composer.set_forces_and_torques(
                forces=self.pushes.force[:,None,:], torques=self.pushes.torque[:,None,:],
                body_ids=self._push_body_ids, is_global=True)

    def _apply_action(self):
        torque = self.servo.compute(self.controller.target,
                                    self.robot.data.joint_pos[:, self.joint_ids],
                                    self.robot.data.joint_vel[:, self.joint_ids])
        self.robot.set_joint_effort_target(torque, joint_ids=self.joint_ids)

    def _get_observations(self):
        # RSL-RL requests observations more than once before the first rollout.
        # Never advance the history or IMU filter twice at the same control step.
        if self._observation_step == self.common_step_counter and self._cached_observations is not None:
            return self._cached_observations
        ids = (self.command_timer <= 0).nonzero(as_tuple=False).flatten()
        if len(ids):
            self._sample_commands(ids)
        data = self.robot.data
        gyro = data.root_ang_vel_b.clone()
        accel = quat_apply_inverse(data.root_quat_w,
                                   data.body_lin_acc_w[:, self.imu_id] - self.gravity_w)
        # The actor gets accelerometer/gyro measurements, never simulator orientation.
        if self.cfg.randomize:
            imu = self.params["imu"]
            gyro += self.gyro_bias + torch.randn_like(gyro) * imu["gyro_noise_std_rad_s"]
            accel += self.accel_bias + torch.randn_like(accel) * imu["accel_noise_std_m_s2"]
        policy = self.controller.observe(gyro, accel, self.commands)
        contact_forces = self.contact_sensor.data.net_forces_w[:, self.contact_foot_ids]
        self.feet_contact[:] = contact_forces.norm(dim=-1) > 0.5
        critic = torch.cat((policy,
                            data.root_lin_vel_b, data.root_ang_vel_b * 0.25,
                            data.projected_gravity_b,
                            data.joint_pos[:, self.joint_ids] - self.controller.rest,
                            data.joint_vel[:, self.joint_ids] * 0.1,
                            self._relative_base_height().unsqueeze(1),
                            self.feet_contact.float()), dim=-1).clamp(-20, 20)
        self._cached_observations = {"policy": policy, "critic": critic}
        self._observation_step = self.common_step_counter
        return self._cached_observations

    def _get_dones(self):
        data = self.robot.data
        height = self._relative_base_height()
        base_force = self.contact_sensor.data.net_forces_w_history[:, :, self.base_ids].norm(dim=-1).amax(dim=(1, 2))
        self.step_metrics = {
            "height": height.clone(),
            "velocity": torch.cat((data.root_lin_vel_b[:, :2], data.root_ang_vel_b[:, 2:3]), dim=-1).clone(),
            "torque": self.servo.last_torque.clone(),
            "joint_speed": data.joint_vel[:, self.joint_ids].clone(),
            "joint_position": data.joint_pos[:, self.joint_ids].clone(),
            "episode_step": self.episode_length_buf.clone(),
        }
        fallen = (height < 0.085) | (data.projected_gravity_b[:, 2] > -0.45) | (base_force > 3.0)
        # Give contact impulses after reset a short settling window.
        fallen &= self.episode_length_buf > 5
        timeout = self.episode_length_buf >= self.max_episode_length - 1
        terrain_exit = self._ground_missed.clone()
        if self.params.get("terrain", {}).get("type") == "rough":
            half = torch.tensor(self.params["terrain"].get("patch_size_m",[8.,8.]),device=self.device)/2-.3
            terrain_exit |= ((data.root_pos_w[:,:2]-self.env_origins[:,:2]).abs() > half).any(-1)
            timeout |= terrain_exit
            self.step_metrics["terrain_level"] = self.terrain.terrain_levels.clone()
        self.step_metrics["terrain_boundary_reset"] = terrain_exit.clone()
        if self.pushes is not None:
            self.step_metrics["external_force_w_n"] = self.pushes.force.clone()
            self.step_metrics["external_torque_w_nm"] = self.pushes.torque.clone()
        return fallen, timeout

    def _get_rewards(self):
        data = self.robot.data
        height = self._relative_base_height()
        moving = (self.commands[:, :2].norm(dim=-1) > 0.025) | (self.commands[:, 2].abs() > 0.05)
        lin_error = ((self.commands[:, :2] - data.root_lin_vel_b[:, :2]) ** 2).sum(dim=-1)
        yaw_error = (self.commands[:, 2] - data.root_ang_vel_b[:, 2]) ** 2
        if self.robustness_curriculum:
            self._robust_error += lin_error.sqrt()
            self._robust_samples += 1
        weights = self.params.get("reward_weights", {})
        locomotion = self.params.get("reward_parameters",{}).get("locomotion_v2",{})
        tracking_sigma = torch.where(moving,self.cfg.lin_tracking_sigma,
                                     float(locomotion.get("stand_tracking_sigma_m_s",self.cfg.lin_tracking_sigma)))
        contact = self.contact_sensor.data.net_forces_w[:, self.contact_foot_ids].norm(dim=-1) > 0.5
        self.step_metrics["foot_contact"] = contact.clone()
        desired_stance = ((self.controller.phase[:, None] + self.gait_offsets) % 1.0) < 0.60
        desired_stance[~moving] = True
        contact_score = (contact == desired_stance).float().mean(dim=-1)
        if self.performance_curriculum and self.command_override is None:
            frontier = self.commands[:,0] >= .7*self.performance_curriculum.maximum
            self._frontier_error += (self.commands[:,0]-data.root_lin_vel_b[:,0]).abs()*frontier
            self._frontier_count += frontier
        posture = ((data.joint_pos[:, self.joint_ids] - self.controller.rest) ** 2).sum(dim=-1)
        terms = {
            "track_velocity": self.cfg.lin_tracking_weight * torch.exp(-lin_error / tracking_sigma ** 2),
            "track_yaw": self.cfg.yaw_tracking_weight * torch.exp(-yaw_error / self.cfg.yaw_tracking_sigma ** 2),
            "upright": 0.6 - 2.0 * data.projected_gravity_b[:, :2].square().sum(dim=-1),
            "height": 0.5 * torch.exp(-((height - self.cfg.target_height) / 0.04) ** 2),
            "contact_schedule": float(weights.get("contact_schedule",.35)) * contact_score,
            "torque": -float(weights.get("torque",.06)) * self.servo.last_torque.square().sum(dim=-1),
            "smoothness": -float(weights.get("target_rate", 0.02)) * (self.delta / self.step_dt).square().sum(dim=-1)
                          - float(weights.get("target_rate_change", 0.01)) * ((self.delta - self.previous_delta) / self.step_dt).square().sum(dim=-1),
            "posture": -posture * torch.where(moving, float(weights.get("posture_moving",.08)),
                                               float(weights.get("posture_standing",.32))),
            "vertical_motion": -0.5 * data.root_lin_vel_b[:, 2].square()
                               - 0.04 * data.root_ang_vel_b[:, :2].square().sum(dim=-1),
            "termination": -2.0 * self.reset_terminated.float() / self.step_dt,
        }
        if self.params.get("reward_parameters",{}).get("locomotion_v2"):
            terms["stand_contacts"] = float(weights.get("stand_contacts",0.))*contact.float().mean(-1)*(~moving)
            terms["stand_motion"] = -float(weights.get("stand_joint_motion",0.))*data.joint_vel[:,self.joint_ids].square().sum(-1)*(~moving)
            terms["tracking_error"] = -float(weights.get("tracking_error",2.))*torch.sqrt(lin_error+.01**2)
            self._air_time += (~contact)*self.step_dt
            landed = contact & ~self._air_contact & (self.episode_length_buf[:,None] > 5)
            air_for_quality = self._air_time.clone()
            air_target = float(self.params["reward_parameters"]["locomotion_v2"].get("air_time_target_s",.14))
            terms["air_time"] = (float(weights.get("air_time",.4))*((self._air_time-air_target).clamp(-air_target,.14)*landed).sum(-1)*moving/self.step_dt)
            self._air_time[contact] = 0
            self._air_contact[:] = contact
        if self.style_parameters:
            if getattr(self,"record_collision_geometry",False):
                self.step_metrics["foot_link_position_w"] = data.body_link_pos_w[:,self.foot_ids].clone()
                self.step_metrics["foot_link_quaternion_w"] = data.body_link_quat_w[:,self.foot_ids].clone()
            offset_w = quat_apply(data.body_link_quat_w[:,self.foot_ids],self.toe_offset)
            toe_position = data.body_link_pos_w[:,self.foot_ids] + offset_w
            toe_velocity = data.body_link_lin_vel_w[:,self.foot_ids] + torch.linalg.cross(data.body_link_ang_vel_w[:,self.foot_ids],offset_w,dim=-1)
            radius = float(self.style_parameters["ground_radius_proxy_m"])
            ground_offset = torch.zeros_like(offset_w)
            ground_offset[...,2] = -radius
            point_velocity = toe_velocity + torch.linalg.cross(data.body_link_ang_vel_w[:,self.foot_ids],ground_offset,dim=-1)
            clearance = (toe_position[...,2]-self._foot_ground-radius).clamp(min=0)
            self.step_metrics["foot_ground_height_w"] = self._foot_ground.clone()
            self.step_metrics["foot_clearance"] = clearance.clone()
            self.step_metrics["foot_position_body"] = quat_apply_inverse(
                data.root_quat_w[:,None,:].expand(-1,4,-1), toe_position-data.root_pos_w[:,None,:])
            new_terms,self.style_metrics = style_terms(contact,clearance,point_velocity,
                (self.controller.phase[:,None]+self.gait_offsets)%1.,moving,
                data.root_lin_vel_b,data.root_ang_vel_b,self.previous_style_contact,
                self.previous_toe_vz,weights,self.style_parameters,self.step_dt)
            # Exclude initial settling from the event cost. Other costs ramp in.
            new_terms["touchdown"] *= self.episode_length_buf > 5
            for key in ("touchdown_count","touchdown_downward_speed_sum_m_s"):
                self.style_metrics[key] *= self.episode_length_buf > 5
            ramp_steps = float(self.style_parameters.get("ramp_transitions",1))
            progress = min((self.style_curriculum_offset+self.common_step_counter*self.num_envs)/max(ramp_steps,1),1.)
            terms.update({key:value*progress for key,value in new_terms.items()})
            terms["contact_schedule"] = (.35+progress*(float(weights.get("contact_schedule",.35))-.35))*contact_score
            self.previous_style_contact[:] = contact
            self.previous_toe_vz[:] = point_velocity[...,2]
            quality = self.params.get("reward_parameters",{}).get("gait_quality")
            if quality:
                self._swing_peak[:] = torch.maximum(self._swing_peak, clearance*(~contact))
                shaped = gait_quality_terms(air_for_quality, self._swing_peak, landed, moving, weights,
                    quality.get("swing_height_m",.03), quality.get("swing_height_width_m",.015), self.step_dt)
                ramp = min((self.robustness_offset+self.common_step_counter*self.num_envs)/max(quality.get("ramp_transitions",4_000_000),1),1.)
                terms.update({k:v*ramp for k,v in shaped.items()})
                self._swing_peak[contact] = 0
        self.reward_terms = terms
        for name, term in terms.items():
            self._episode_sums[name] += term * self.step_dt
        return sum(terms.values()) * self.step_dt

    def _reset_idx(self, env_ids):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        curriculum_log = {}
        if self.robustness_curriculum:
            valid = self.episode_length_buf[env_ids] > 5
            survived = valid & ~self.reset_terminated[env_ids]
            if self.terrain_curriculum_enabled and self.command_override is None:
                self.robustness_curriculum.observe(int(valid.sum()),int(survived.sum()),
                    float(self._robust_samples[env_ids].sum()),float(self._robust_error[env_ids].sum()),
                    self.robustness_offset+self.common_step_counter*self.num_envs)
            curriculum_log["Curriculum/push_strength_fraction"] = self.robustness_curriculum.strength
            if self.robustness_curriculum.last_error is not None:
                curriculum_log["Curriculum/robust_tracking_error_m_s"] = self.robustness_curriculum.last_error
                curriculum_log["Curriculum/robust_survival"] = self.robustness_curriculum.last_survival
        if self.terrain.terrain_origins is not None and self.terrain_curriculum_enabled and self.command_override is None:
            error = self._robust_error[env_ids]/self._robust_samples[env_ids].clamp(min=1)
            mature = self.episode_length_buf[env_ids]*self.step_dt >= 5.
            distance = (self.robot.data.root_pos_w[env_ids,:2]-self.env_origins[env_ids,:2]).norm(dim=-1)
            up = mature & ~self.reset_terminated[env_ids] & (error < .15) & (distance > .5)
            down = self.reset_terminated[env_ids] & (self.episode_length_buf[env_ids]>5)
            self.terrain.update_env_origins(env_ids, up, down)
            curriculum_log["Curriculum/terrain_level_mean"] = float(self.terrain.terrain_levels.float().mean())
        self._robust_error[env_ids] = 0
        self._robust_samples[env_ids] = 0
        self._swing_peak[env_ids] = 0
        if self.performance_curriculum and self.command_override is None:
            survived = ((~self.reset_terminated[env_ids]) & (self.episode_length_buf[env_ids] >= .8*self.max_episode_length)).sum()
            episodes = (self.episode_length_buf[env_ids] > 5).sum()
            self.performance_curriculum.observe(int(episodes), int(survived),
                float(self._frontier_count[env_ids].sum()),float(self._frontier_error[env_ids].sum()),
                self.curriculum_offset+self.common_step_counter*self.num_envs)
            curriculum_log["Curriculum/max_forward_m_s"] = self.performance_curriculum.maximum
            if self.performance_curriculum.last_error is not None:
                curriculum_log["Curriculum/frontier_error_m_s"] = self.performance_curriculum.last_error
                curriculum_log["Curriculum/survival_fraction"] = self.performance_curriculum.last_survival
        self._frontier_count[env_ids] = 0
        self._frontier_error[env_ids] = 0
        self._air_time[env_ids] = 0
        self._air_contact[env_ids] = False
        super()._reset_idx(env_ids)
        # Internal initialization in DirectRLEnv does not call this before our buffers exist.
        self.controller.reset(env_ids)
        self.previous_target[env_ids] = self.controller.rest
        self.previous_delta[env_ids] = 0
        self.delta[env_ids] = 0
        self._sample_commands(env_ids)
        imu = self.params["imu"]
        self.gyro_bias[env_ids] = torch.empty(len(env_ids), 3, device=self.device).uniform_(-imu["gyro_bias_max_rad_s"], imu["gyro_bias_max_rad_s"]) if self.cfg.randomize else 0
        self.accel_bias[env_ids] = torch.empty(len(env_ids), 3, device=self.device).uniform_(-imu["accel_bias_max_m_s2"], imu["accel_bias_max_m_s2"]) if self.cfg.randomize else 0
        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        if self.cfg.randomize:
            joint_pos[:, self.joint_ids] += torch.empty(len(env_ids), 12, device=self.device).uniform_(-0.025, 0.025)
        joint_vel = torch.zeros_like(joint_pos)
        root = self.robot.data.default_root_state[env_ids].clone()
        root[:, :3] += self.env_origins[env_ids]
        if self.cfg.randomize:
            root[:, 3:7] = quat_from_euler_xyz(
                torch.empty(len(env_ids), device=self.device).uniform_(-0.03, 0.03),
                torch.empty(len(env_ids), device=self.device).uniform_(-0.03, 0.03),
                torch.empty(len(env_ids), device=self.device).uniform_(-math.pi, math.pi))
        self.robot.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
        self.robot.write_root_velocity_to_sim(root[:, 7:], env_ids=env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        self.robot.set_joint_effort_target(torch.zeros_like(joint_pos), env_ids=env_ids)
        self.servo.reset(env_ids, joint_pos[:, self.joint_ids], randomize=self.cfg.randomize)
        if self.pushes is not None:
            self.pushes.reset(env_ids)
        self.feet_contact[env_ids] = False
        if self.style_parameters:
            self.previous_style_contact[env_ids] = False
            self.previous_toe_vz[env_ids] = 0
        log = curriculum_log
        for key, values in self._episode_sums.items():
            log[f"Episode_Reward/{key}"] = values[env_ids].mean() / self.max_episode_length_s
            values[env_ids] = 0
        log["Episode_Termination/fallen"] = self.reset_terminated[env_ids].float().mean()
        counts = self._control_diagnostics[env_ids, 3].clamp(min=1)
        log["Episode_Control/action_at_limit_fraction"] = (self._control_diagnostics[env_ids, 0] / counts).mean()
        log["Episode_Control/target_rate_limited_fraction"] = (self._control_diagnostics[env_ids, 1] / counts).mean()
        log["Episode_Control/near_boundary_fraction_abs_ge_098"] = (self._control_diagnostics[env_ids, 2] / counts).mean()
        self._control_diagnostics[env_ids] = 0
        self.extras["log"] = log
        self.total_resets += len(env_ids)
        self._observation_step = -1
        self._ground_step = -1


def register():
    import gymnasium as gym
    if "SpotMicro-Velocity-Flat-v0" not in gym.registry:
        gym.register(id="SpotMicro-Velocity-Flat-v0", entry_point=SpotMicroEnv,
                     disable_env_checker=True)
