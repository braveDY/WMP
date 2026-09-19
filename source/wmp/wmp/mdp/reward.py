from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp as base_mdp
from isaaclab.envs.mdp import *  # noqa: F401, F403
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
import isaaclab.utils.math as math_utils
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset = env.scene[asset_cfg.name]
    body_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    return torch.sum(body_vel.norm(dim=-1) * contacts, dim=1)


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    return torch.any(forces_xy > 4.0 * forces_z, dim=1).float()


def track_lin_vel_xy_yaw_frame_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_ang_vel_z_world_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    return torch.exp(-ang_vel_error / std**2)


def progress_along_command(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)[:, :2]
    command_speed = torch.norm(command, dim=1)
    command_dir = command / torch.clamp(command_speed.unsqueeze(-1), min=1.0e-6)
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    progress = torch.sum(vel_yaw[:, :2] * command_dir, dim=1)
    return progress * (command_speed > command_threshold)


def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    return base_mdp.joint_deviation_l1(env, asset_cfg) * (torch.norm(command[:, :2], dim=1) < command_threshold)


def _get_parkour_event(env: ManagerBasedRLEnv, parkour_name: str):
    parkour_manager = getattr(env, "parkour_manager", None)
    if parkour_manager is None:
        return None
    return parkour_manager.get_term(parkour_name)


def parkour_reward_torques(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.applied_torque[:, asset_cfg.joint_ids]), dim=1)


def parkour_reward_dof_error(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(
        torch.square(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]),
        dim=1,
    )


class parkour_reward_action_rate(ManagerTermBase):
    def __init__(self, cfg: "RewardTermCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.previous_actions = torch.zeros(env.num_envs, 2, asset.num_joints, dtype=torch.float, device=self.device)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self.previous_actions[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        self.previous_actions[:, 0, :] = self.previous_actions[:, 1, :]
        self.previous_actions[:, 1, :] = env.action_manager.get_term("joint_pos").raw_actions
        return torch.sum(torch.square(self.previous_actions[:, 1, :] - self.previous_actions[:, 0, :]), dim=1)


class parkour_reward_dof_acc(ManagerTermBase):
    def __init__(self, cfg: "RewardTermCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        self.previous_joint_vel = torch.zeros(env.num_envs, 2, asset.num_joints, dtype=torch.float, device=self.device)
        self.dt = env.cfg.decimation * env.cfg.sim.dt

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        env_ids = slice(None) if env_ids is None else env_ids
        self.previous_joint_vel[env_ids] = 0.0

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
        asset: Articulation = env.scene[asset_cfg.name]
        self.previous_joint_vel[:, 0, :] = self.previous_joint_vel[:, 1, :]
        self.previous_joint_vel[:, 1, :] = asset.data.joint_vel
        return torch.sum(
            torch.square(
                (
                    self.previous_joint_vel[:, 1, asset_cfg.joint_ids]
                    - self.previous_joint_vel[:, 0, asset_cfg.joint_ids]
                )
                / self.dt
            ),
            dim=1,
        )


def parkour_reward_lin_vel_z(
    env: ManagerBasedRLEnv,
    parkour_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    parkour_event = _get_parkour_event(env, parkour_name)
    if parkour_event is not None:
        terrain_names = parkour_event.env_per_terrain_name
        reward[(terrain_names != "parkour_flat")[:, -1]] *= 0.5
    return reward


def parkour_reward_cheat(
    env: ManagerBasedRLEnv,
    heading_limit: float = 1.0,
    command_name: str = "base_velocity",
    use_command_heading: bool = False,
    min_terrain_level: int | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    target_heading = torch.zeros(env.num_envs, device=env.device)
    if use_command_heading:
        target_heading = _velocity_command_target_heading(env, command_name, asset)

    heading_error = torch.abs(math_utils.wrap_to_pi(asset.data.heading_w - target_heading))
    reward = (heading_error > heading_limit).float()
    terrain_levels = getattr(getattr(env.scene, "terrain", None), "terrain_levels", None)
    if min_terrain_level is not None and terrain_levels is not None:
        reward *= (terrain_levels > min_terrain_level).float()
    return reward


def parkour_reward_tracking_lin_vel_clipped(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    tracking_sigma: float = 0.25,
    lin_vel_clip: float = 0.2,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    lin_vel = asset.data.root_lin_vel_b[:, :2]
    lin_cmd = command[:, :2]
    upper_bound = torch.where(lin_cmd < 0.0, torch.full_like(lin_cmd, 1.0e5), lin_cmd + lin_vel_clip)
    lower_bound = torch.where(lin_cmd > 0.0, torch.full_like(lin_cmd, -1.0e5), lin_cmd - lin_vel_clip)
    clipped_lin_vel = torch.clip(lin_vel, lower_bound, upper_bound)
    lin_vel_error = torch.sum(torch.square(lin_cmd - clipped_lin_vel), dim=1)
    return torch.exp(-lin_vel_error / tracking_sigma)


def parkour_reward_tracking_ang_vel(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    tracking_sigma: float = 0.25,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    ang_vel_error = torch.square(command[:, 2] - asset.data.root_ang_vel_b[:, 2])
    return torch.exp(-ang_vel_error / tracking_sigma)


def parkour_reward_feet_air_time(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str = "base_velocity",
    threshold: float = 0.5,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :2], dim=1) > 0.1
    return reward


def parkour_reward_feet_edge(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    contact_sensor_cfg: SceneEntityCfg,
    edge_height_threshold: float = 0.08,
    nearest_k: int = 9,
    contact_threshold: float = 2.0,
    min_terrain_level: int | None = 3,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[contact_sensor_cfg.name]

    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids]
    relative_foot_pos_w = foot_pos_w - sensor.data.pos_w.unsqueeze(1)
    sensor_quat = sensor.data.quat_w
    if getattr(sensor.cfg, "ray_alignment", "base") == "yaw":
        sensor_quat = math_utils.yaw_quat(sensor_quat)
    foot_quat = sensor_quat.unsqueeze(1).expand(env.num_envs, len(asset_cfg.body_ids), 4)
    foot_pos_s = math_utils.quat_apply_inverse(
        foot_quat.reshape(-1, 4), relative_foot_pos_w.reshape(-1, 3)
    ).reshape_as(relative_foot_pos_w)

    ray_pos_s = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
    num_rays = ray_pos_s.shape[1]
    ray_quat = sensor_quat.unsqueeze(1).expand(env.num_envs, num_rays, 4)
    ray_pos_s = math_utils.quat_apply_inverse(ray_quat.reshape(-1, 4), ray_pos_s.reshape(-1, 3)).reshape(
        env.num_envs, num_rays, 3
    )
    ray_pos_s = torch.nan_to_num(ray_pos_s)

    dist_xy = torch.linalg.norm(ray_pos_s[:, None, :, :2] - foot_pos_s[:, :, None, :2], dim=-1)
    k = min(nearest_k, num_rays)
    nearest_ray_ids = torch.topk(dist_xy, k=k, dim=-1, largest=False).indices
    local_heights = torch.gather(
        ray_pos_s[:, None, :, 2].expand(-1, foot_pos_s.shape[1], -1),
        2,
        nearest_ray_ids,
    )
    feet_at_edge = (local_heights.max(dim=-1).values - local_heights.min(dim=-1).values) > edge_height_threshold

    contact_forces = contact_sensor.data.net_forces_w_history[:, 0, contact_sensor_cfg.body_ids]
    feet_contact = torch.norm(contact_forces, dim=-1) > contact_threshold
    reward = torch.sum((feet_at_edge & feet_contact).float(), dim=1)
    terrain_levels = getattr(getattr(env.scene, "terrain", None), "terrain_levels", None)
    if min_terrain_level is not None and terrain_levels is not None:
        reward *= (terrain_levels > min_terrain_level).float()
    return reward


def parkour_reward_feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]
    return torch.any(
        torch.norm(net_contact_forces[:, :, :2], dim=2) > 4 * torch.abs(net_contact_forces[:, :, 2]), dim=1
    ).float()


def parkour_reward_stuck(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    return ((torch.abs(asset.data.root_lin_vel_b[:, 0]) < 0.1) & (torch.abs(command[:, 0]) > 0.1)).float()


def parkour_reward_collision(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    net_contact_forces = contact_sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids]
    return torch.sum((torch.norm(net_contact_forces, dim=-1) > 0.1).float(), dim=1)


def _velocity_command_target_heading(
    env: ManagerBasedRLEnv, command_name: str, asset: RigidObject
) -> torch.Tensor:
    command_term = env.command_manager.get_term(command_name)
    if hasattr(command_term, "heading_target"):
        return command_term.heading_target
    command = env.command_manager.get_command(command_name)
    return asset.data.heading_w + command[:, 2] * env.step_dt
