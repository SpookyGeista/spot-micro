"""Isaac Lab 2.3.2 / Isaac Sim 5.1 configuration."""
from pathlib import Path
import hashlib

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from .contract import ROOT, FRAME_SIZE, frame_size, load_robot

ROBOT = load_robot()


@configclass
class RandomizationCfg:
    material = EventTermCfg(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*"),
                "static_friction_range": (0.6, 1.1), "dynamic_friction_range": (0.5, 0.9),
                "restitution_range": (0.0, 0.0), "num_buckets": 32, "make_consistent": True},
    )
    mass = EventTermCfg(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
                "mass_distribution_params": (0.85, 1.15), "operation": "scale", "recompute_inertia": True},
    )


@configclass
class SpotMicroEnvCfg(DirectRLEnvCfg):
    episode_length_s = 15.0
    decimation = 10
    action_space = 12
    observation_space = FRAME_SIZE * ROBOT["history_length"]
    state_space = observation_space + 38
    robot_parameters: dict = ROBOT
    randomize = True
    task_mode = "walk"
    curriculum_steps = 100_000_000
    command_resample_s = 4.0
    target_height = 0.18
    lin_tracking_sigma = 0.05
    yaw_tracking_sigma = 0.20
    lin_tracking_weight = 2.0
    yaw_tracking_weight = 0.7
    sim: SimulationCfg = SimulationCfg(
        dt=0.002,
        render_interval=10,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0, dynamic_friction=1.0, restitution=0.0,
            friction_combine_mode="multiply", restitution_combine_mode="multiply"),
        physx=sim_utils.PhysxCfg(solver_type=1, min_position_iteration_count=8, min_velocity_iteration_count=1),
    )
    scene = InteractiveSceneCfg(num_envs=2048, env_spacing=2.0, replicate_physics=True)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
        debug_vis=False,
    )
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(ROOT / "assets/spot_micro/spot_micro.urdf"),
            usd_dir=str(ROOT / "assets/spot_micro/generated"), usd_file_name="spot_micro.usd",
            fix_base=False, merge_fixed_joints=True, self_collision=True,
            joint_drive=sim_utils.UrdfFileCfg.JointDriveCfg(
                target_type="none", gains=sim_utils.UrdfFileCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)),
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False, max_depenetration_velocity=1.0,
                linear_damping=0.0, angular_damping=0.0),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=1),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.18),
            joint_pos=dict(zip(ROBOT["joint_names"], ROBOT["rest_positions_rad"])),
            joint_vel={".*": 0.0},
        ),
        actuators={"servos": IdealPDActuatorCfg(
            joint_names_expr=["motor_.*"], stiffness=0.0, damping=0.0,
            effort_limit=1.2, effort_limit_sim=1.2, velocity_limit_sim=12.0, armature=0.0005)},
    )
    contacts = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*", update_period=0.0, history_length=3, track_air_time=True)
    events = RandomizationCfg()


def make_env_cfg(robot_path=None, num_envs=2048, device="cuda:0", seed=42,
                 mode="walk", randomize=True, fixed_base=False):
    parameters = load_robot(robot_path)
    cfg = SpotMicroEnvCfg()
    cfg.robot_parameters = parameters
    weights = parameters.get("reward_weights", {})
    cfg.lin_tracking_weight = float(weights.get("track_velocity", 2.0))
    cfg.yaw_tracking_weight = float(weights.get("track_yaw", 0.7))
    tolerances = parameters.get("tracking_tolerances", {})
    cfg.lin_tracking_sigma = float(tolerances.get("linear_m_s", 0.05))
    cfg.yaw_tracking_sigma = float(tolerances.get("yaw_rad_s", 0.20))
    if cfg.lin_tracking_sigma <= 0 or cfg.yaw_tracking_sigma <= 0:
        raise ValueError("Tracking tolerances must be positive")
    if "urdf_asset_path" in parameters:
        path = ROOT / parameters["urdf_asset_path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != parameters["urdf_sha256"]:
            raise ValueError("Robot model changed since this configuration was saved")
        cfg.robot.spawn.asset_path = str(path)
        cfg.robot.spawn.usd_dir = str(path.parent / "generated")
    cfg.decimation = round(parameters["control_dt"] / parameters["physics_dt"])
    cfg.sim.dt = parameters["physics_dt"]
    cfg.sim.render_interval = cfg.decimation
    cfg.sim.device = device
    cfg.scene.num_envs = num_envs
    cfg.seed = seed
    cfg.task_mode = mode
    cfg.randomize = randomize
    cfg.observation_space = frame_size(parameters) * parameters["history_length"]
    cfg.state_space = cfg.observation_space + 38
    cfg.robot.init_state.joint_pos = dict(zip(parameters["joint_names"], parameters["rest_positions_rad"]))
    cfg.target_height = parameters.get("nominal_base_height_m", 0.18)
    cfg.robot.init_state.pos = (0.0, 0.0, cfg.target_height)
    cfg.robot.spawn.fix_base = fixed_base
    cfg.robot.actuators["servos"].armature = parameters["servo"]["armature_kg_m2"]
    if fixed_base:
        cfg.robot.spawn.usd_file_name = "spot_micro_fixed.usd"
    if not randomize:
        cfg.events = None
    else:
        domain = parameters.get("domain_randomization", {})
        cfg.events.material.params["static_friction_range"] = tuple(domain.get("static_friction_range", [.6, 1.1]))
        cfg.events.material.params["dynamic_friction_range"] = tuple(domain.get("dynamic_friction_range", [.5, .9]))
        cfg.events.mass.params["mass_distribution_params"] = tuple(domain.get("base_mass_scale_range", [.85, 1.15]))
    if parameters.get("terrain", {}).get("type") == "rough":
        from .rough_terrain import configure_rough_terrain
        configure_rough_terrain(cfg, parameters["terrain"], seed)
    if not Path(cfg.robot.spawn.asset_path).is_file():
        raise FileNotFoundError("Missing robot assets. Run: python scripts/fetch_assets.py")
    cfg.viewer.eye = (0.9, 0.9, 0.65)
    cfg.viewer.lookat = (0.0, 0.0, 0.12)
    cfg.viewer.origin_type = "env"
    cfg.viewer.env_index = 0
    cfg.viewer.resolution = (960, 720)
    return cfg
