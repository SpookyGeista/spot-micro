"""Local runner extension; installed RSL-RL remains unchanged."""
import copy
import torch
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.algorithms import PPO
from .bounded_policy import SquashedActorCritic, DirectSquashedActorCritic


def restore_robustness_state(raw, info):
    if getattr(raw, "robustness_curriculum", None) and "robustness_curriculum_state" in info:
        raw.robustness_curriculum.load_state_dict(info["robustness_curriculum_state"])
        raw.robustness_offset = info.get("robustness_transitions", 0)
    if raw.terrain.terrain_origins is not None and "terrain_levels" in info:
        levels = torch.tensor(info["terrain_levels"],device=raw.device,dtype=torch.long)
        if len(levels) != raw.num_envs:
            levels = levels[torch.randint(len(levels),(raw.num_envs,),device=raw.device)]
        raw.terrain.terrain_levels[:] = levels.clamp(0,raw.terrain.max_terrain_level-1)
        raw.terrain.env_origins[:] = raw.terrain.terrain_origins[raw.terrain.terrain_levels,raw.terrain.terrain_types]


class SpotOnPolicyRunner(OnPolicyRunner):
    def _construct_algorithm(self, obs):
        classes = {"SquashedActorCritic": SquashedActorCritic,
                   "DirectSquashedActorCritic": DirectSquashedActorCritic}
        if self.policy_cfg.get("class_name") not in classes:
            return super()._construct_algorithm(obs)
        if self.alg_cfg.get("class_name") != "PPO" or self.alg_cfg.get("rnd_cfg") or self.alg_cfg.get("symmetry_cfg"):
            raise ValueError("Bounded runner supports plain feed-forward PPO only")
        policy_cfg = copy.deepcopy(self.policy_cfg)
        policy_class = classes[policy_cfg.pop("class_name")]
        policy = policy_class(obs, self.cfg["obs_groups"], self.env.num_actions, **policy_cfg)
        alg_cfg = copy.deepcopy(self.alg_cfg)
        alg_cfg.pop("class_name")
        alg = PPO(policy, device=self.device, multi_gpu_cfg=self.multi_gpu_cfg, **alg_cfg)
        alg.init_storage("rl", self.env.num_envs, self.num_steps_per_env, obs, [self.env.num_actions])
        return alg

    def save(self, path, infos=None):
        raw = self.env.unwrapped
        info = dict(infos or {})
        info.setdefault("curriculum_transitions", raw.curriculum_offset + raw.common_step_counter * raw.num_envs)
        info.setdefault("style_curriculum_transitions", raw.style_curriculum_offset + raw.common_step_counter * raw.num_envs)
        if getattr(raw, "performance_curriculum", None):
            info["command_curriculum_state"] = raw.performance_curriculum.state_dict()
        if getattr(raw, "robustness_curriculum", None):
            info["robustness_curriculum_state"] = raw.robustness_curriculum.state_dict()
            info["robustness_transitions"] = raw.robustness_offset + raw.common_step_counter * raw.num_envs
        if raw.terrain.terrain_origins is not None:
            info["terrain_levels"] = raw.terrain.terrain_levels.cpu().tolist()
        return super().save(path, infos=info)

    def load(self, *args, **kwargs):
        info = super().load(*args, **kwargs)
        raw = self.env.unwrapped
        if info:
            raw.curriculum_offset = info.get("curriculum_transitions", raw.curriculum_offset)
            raw.style_curriculum_offset = info.get("style_curriculum_transitions", raw.style_curriculum_offset)
            if getattr(raw, "performance_curriculum", None) and "command_curriculum_state" in info:
                raw.performance_curriculum.load_state_dict(info["command_curriculum_state"])
            restore_robustness_state(raw, info)
        return info
