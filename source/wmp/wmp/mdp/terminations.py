from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def terrain_out_of_bounds(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), distance_buffer: float = 3.0
) -> torch.Tensor:
    if env.scene.cfg.terrain.terrain_type == "plane":
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    if env.scene.cfg.terrain.terrain_type != "generator":
        raise ValueError("Received unsupported terrain type, must be either 'plane' or 'generator'.")

    terrain_gen_cfg = env.scene.terrain.cfg.terrain_generator
    grid_width, grid_length = terrain_gen_cfg.size
    map_width = terrain_gen_cfg.num_rows * grid_width + 2 * terrain_gen_cfg.border_width
    map_height = terrain_gen_cfg.num_cols * grid_length + 2 * terrain_gen_cfg.border_width

    asset: RigidObject = env.scene[asset_cfg.name]
    x_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 0]) > 0.5 * map_width - distance_buffer
    y_out_of_bounds = torch.abs(asset.data.root_pos_w[:, 1]) > 0.5 * map_height - distance_buffer
    return torch.logical_or(x_out_of_bounds, y_out_of_bounds)


def _ray_hits_sensor_frame(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    relative_pos_w = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
    sensor_quat = sensor.data.quat_w
    if getattr(sensor.cfg, "ray_alignment", "base") == "yaw":
        sensor_quat = math_utils.yaw_quat(sensor_quat)
    num_envs, num_rays, _ = relative_pos_w.shape
    sensor_quat = sensor_quat.unsqueeze(1).expand(num_envs, num_rays, 4).reshape(num_envs * num_rays, 4)
    hit_pos_s = math_utils.quat_apply_inverse(sensor_quat, relative_pos_w.reshape(num_envs * num_rays, 3))
    return torch.nan_to_num(hit_pos_s.reshape(num_envs, num_rays, 3))


def _root_pos_sensor_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    relative_pos_w = asset.data.root_pos_w - sensor.data.pos_w
    sensor_quat = sensor.data.quat_w
    if getattr(sensor.cfg, "ray_alignment", "base") == "yaw":
        sensor_quat = math_utils.yaw_quat(sensor_quat)
    return math_utils.quat_apply_inverse(sensor_quat, relative_pos_w)


def base_height_below_scan(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    min_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    ray_hits_s = _ray_hits_sensor_frame(env, sensor_cfg)
    root_pos_s = _root_pos_sensor_frame(env, asset_cfg, sensor_cfg)
    dist_xy = torch.linalg.norm(ray_hits_s[:, :, :2] - root_pos_s[:, None, :2], dim=-1)
    nearest_ray_ids = torch.argmin(dist_xy, dim=-1)
    ground_height = torch.gather(ray_hits_s[:, :, 2], 1, nearest_ray_ids.unsqueeze(-1)).squeeze(-1)
    base_height = root_pos_s[:, 2] - ground_height
    return torch.clamp(min_height - base_height, min=0.0)


def low_base_height_termination(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    min_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    return base_height_below_scan(env, sensor_cfg, min_height, asset_cfg) > 0.0


def sustained_contacts_termination(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_contact_time: float,
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return torch.any(contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > max_contact_time, dim=1)


def parkour_bad_orientation_termination(
    env: ManagerBasedRLEnv,
    max_projected_gravity_xy: float = 0.7,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    upside_down = asset.data.projected_gravity_b[:, 2] > 0.0
    severe_tilt = torch.any(torch.abs(asset.data.projected_gravity_b[:, :2]) > max_projected_gravity_xy, dim=1)
    return upside_down | severe_tilt


def parkour_downward_velocity_termination(
    env: ManagerBasedRLEnv,
    max_downward_velocity: float = 3.0,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w[:, 2] < -max_downward_velocity


def parkour_command_velocity_violation(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    max_error: float = 1.5,
    min_command: float = 0.1,
    min_terrain_level: int = 4,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    vel_error = asset.data.root_lin_vel_b[:, 0] - command[:, 0]
    violates_forward = (vel_error < -max_error) & (command[:, 0] > min_command)
    violates_backward = (vel_error > max_error) & (command[:, 0] < -min_command)
    violation = violates_forward | violates_backward

    terrain = getattr(env.scene, "terrain", None)
    terrain_levels = getattr(terrain, "terrain_levels", None)
    if terrain_levels is not None:
        violation &= terrain_levels > min_terrain_level
    return violation
