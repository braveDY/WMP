from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
import copy
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_assets.robots.unitree import UNITREE_A1_CFG

from wmp.terrains.terrain_cfg import ROUGH_TERRAINS_CFG

from . import mdp

# Robot links & joints definitions
A1_BASE_LINK = "trunk"
A1_FOOT_LINKS = ".*_foot"
A1_FOOT_LINK_NAMES = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
A1_COLLISION_LINKS = [A1_BASE_LINK, ".*_calf", ".*_thigh"]
A1_JOINT_NAMES = [".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"]

A1_ROBOT_CFG = UNITREE_A1_CFG.replace(soft_joint_pos_limit_factor=0.9)

# Strictly aligned with master (17 x 11 = 187 points)
HEIGHT_MAP_X_SIZE = 1.6
HEIGHT_MAP_Y_SIZE = 1.0
HEIGHT_MAP_RESOLUTION = 0.1
HEIGHT_MAP_CHANNELS = 3
HEIGHT_MAP_GRID_ROWS = int(round(HEIGHT_MAP_X_SIZE / HEIGHT_MAP_RESOLUTION)) + 1  # 17
HEIGHT_MAP_GRID_COLS = int(round(HEIGHT_MAP_Y_SIZE / HEIGHT_MAP_RESOLUTION)) + 1  # 11


@configclass
class VelocitySceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path=f"{ISAACLAB_NUCLEUS_DIR}/Materials/TilesMarbleSpiderWhiteBrickBondHoned/TilesMarbleSpiderWhiteBrickBondHoned.mdl",
            project_uvw=True,
            texture_scale=(0.25, 0.25),
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = A1_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    height_scanner = RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/" + A1_BASE_LINK,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
        debug_vis=False,
        force_threshold=1.0,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    def __post_init__(self):
        self.robot.spawn.articulation_props.enabled_self_collisions = True
        self.robot.actuators["base_legs"] = DCMotorCfg(
            joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
            effort_limit=33.5,
            saturation_effort=33.5,
            velocity_limit=21.0,
            stiffness=40.0,
            damping=1.0,
            friction=0.0,
        )


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.8,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.8),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=A1_JOINT_NAMES,
        scale=0.25,
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=A1_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=A1_JOINT_NAMES, preserve_order=True)},
            scale=0.05,
            noise=Unoise(n_min=-1.5, n_max=1.5),
        )
        actions = ObsTerm(func=mdp.last_action)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.1, n_max=0.1))
        height_scan = ObsTerm(
            func=mdp.elevation_map,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "noise": False},
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=A1_JOINT_NAMES, preserve_order=True)},
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=A1_JOINT_NAMES, preserve_order=True)},
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        true_root_lin_vel_w = ObsTerm(func=mdp.true_root_lin_vel_w)
        true_root_ang_vel_w = ObsTerm(func=mdp.true_root_ang_vel_w)
        rigid_body_mass = ObsTerm(
            func=mdp.rigid_body_mass,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*")},
        )
        rigid_body_friction = ObsTerm(func=mdp.rigid_body_material_friction)
        contact_state = ObsTerm(
            func=mdp.contact_state,
            params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=A1_FOOT_LINKS)},
        )
        height_scan = ObsTerm(
            func=mdp.elevation_map,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "noise": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class AmpCfg(ObsGroup):
        amp = ObsTerm(func=mdp.amp_observations, params={"asset_cfg": SceneEntityCfg("robot")})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()
    amp: AmpCfg = AmpCfg()


@configclass
class EventCfg:
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range": (0.0, 0.1),
            "num_buckets": 64,
        },
    )
    add_base_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=A1_BASE_LINK),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=A1_BASE_LINK),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.01, 0.01)},
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.2, 0.2), "yaw": (0, 0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.5, 1.5), "velocity_range": (0.0, 0.0)},
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(6.0, 6.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    collision = RewTerm(
        func=mdp.parkour_reward_collision,
        weight=-1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=A1_COLLISION_LINKS)},
    )
    tracking_lin_vel = RewTerm(
        func=mdp.parkour_reward_tracking_lin_vel_clipped,
        weight=1.5,
        params={
            "command_name": "base_velocity",
            "tracking_sigma": 0.15,
            "lin_vel_clip": 0.1,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    tracking_ang_vel = RewTerm(
        func=mdp.parkour_reward_tracking_ang_vel,
        weight=0.5,
        params={"command_name": "base_velocity", "tracking_sigma": 0.15, "asset_cfg": SceneEntityCfg("robot")},
    )
    lin_vel_z = RewTerm(
        func=mdp.parkour_reward_lin_vel_z,
        weight=-1.0,
        params={"parkour_name": "base_parkour", "asset_cfg": SceneEntityCfg("robot")},
    )
    torques = RewTerm(
        func=mdp.parkour_reward_torques,
        weight=-1.0e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=A1_JOINT_NAMES)},
    )
    dof_acc = RewTerm(
        func=mdp.parkour_reward_dof_acc,
        weight=-2.5e-7,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=A1_JOINT_NAMES)},
    )
    action_rate = RewTerm(
        func=mdp.parkour_reward_action_rate,
        weight=-0.03,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    dof_error = RewTerm(
        func=mdp.parkour_reward_dof_error,
        weight=-0.04,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=A1_JOINT_NAMES)},
    )
    feet_air_time = RewTerm(
        func=mdp.parkour_reward_feet_air_time,
        weight=0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=A1_FOOT_LINK_NAMES, preserve_order=True
            ),
            "command_name": "base_velocity",
            "threshold": 0.5,
        },
    )
    feet_stumble = RewTerm(
        func=mdp.parkour_reward_feet_stumble,
        weight=-0.1,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=A1_FOOT_LINKS)},
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=A1_BASE_LINK), "threshold": 1.0},
    )
    bad_orientation = DoneTerm(
        func=mdp.parkour_bad_orientation_termination,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_projected_gravity_xy": 0.7,
        },
    )
    terrain_out_of_bounds = DoneTerm(
        func=mdp.terrain_out_of_bounds,
        params={"asset_cfg": SceneEntityCfg("robot"), "distance_buffer": 3.0},
        time_out=True,
    )


@configclass
class CurriculumCfg:
    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class UnitreeA1WMPEnvCfg(ManagerBasedRLEnvCfg):
    only_positive_rewards = True
    privileged_dim = 3
    height_dim = HEIGHT_MAP_GRID_ROWS * HEIGHT_MAP_GRID_COLS  # 17 * 11 = 187
    prop_dim = 48
    wm_prop_dim = 33
    commands_begin_dim = 6
    height_map_grid_rows = HEIGHT_MAP_GRID_ROWS
    height_map_grid_cols = HEIGHT_MAP_GRID_COLS
    height_map_height_range = 1.0

    scene: VelocitySceneCfg = VelocitySceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 16 * 2**15

        self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        self.scene.contact_forces.update_period = self.sim.dt
        self.scene.terrain.terrain_generator.curriculum = True
        self.scene.terrain.terrain_generator = copy.deepcopy(ROUGH_TERRAINS_CFG)


@configclass
class UnitreeA1WMPEnvCfg_PLAY(UnitreeA1WMPEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 16
        self.scene.env_spacing = 2.5
        self.scene.terrain.max_init_terrain_level = None
        self.scene.terrain.terrain_generator.num_rows = 6
        self.scene.terrain.terrain_generator.num_cols = 6
        self.scene.terrain.terrain_generator.curriculum = False

        self.observations.policy.enable_corruption = False
        self.events.push_robot = None

        self.commands.base_velocity.ranges.lin_vel_x = (0.6, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
