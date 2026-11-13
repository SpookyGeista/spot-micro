"""Small actor for CPU inference on the Raspberry Pi 4."""


def runner_config(seed=42, steps=32, save_interval=100):
    return {
        "seed": seed,
        "num_steps_per_env": steps,
        "save_interval": save_interval,
        "obs_groups": {"policy": ["policy"], "critic": ["critic"]},
        "logger": "tensorboard",
        "policy": {
            "class_name": "ActorCritic",
            "init_noise_std": 0.35,
            "noise_std_type": "log",
            "actor_obs_normalization": True,
            "critic_obs_normalization": True,
            "actor_hidden_dims": [128, 64],
            "critic_hidden_dims": [256, 128],
            "activation": "elu",
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.005,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 0.0005,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
        },
    }


def teacher_config(seed=42, steps=32, save_interval=100):
    cfg = runner_config(seed, steps, save_interval)
    cfg["obs_groups"] = {"policy": ["critic"], "critic": ["critic"]}
    cfg["policy"].update(class_name="DirectSquashedActorCritic", init_noise_std=.45,
                         min_noise_std=.15, max_noise_std=.8, latent_mean_bound=2.,
                         action_bound=.99999, actor_hidden_dims=[256,128,64],
                         critic_hidden_dims=[256,128,64])
    cfg["algorithm"].update(learning_rate=.0003, desired_kl=.02, entropy_coef=.003)
    return cfg
