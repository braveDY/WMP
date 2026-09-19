from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def true_root_lin_vel_w(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w


def true_root_ang_vel_w(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_w


def rigid_body_mass(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    masses = asset.root_physx_view.get_masses().to(device=env.device)
    return masses[:, asset_cfg.body_ids]


def rigid_body_material_friction(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    material_properties = asset.root_physx_view.get_material_properties().to(device=env.device)
    return material_properties[:, :, :2].mean(dim=1)


def contact_state(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, threshold: float = 1.0
) -> torch.Tensor:
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    return (torch.linalg.norm(contact_forces, dim=-1) > threshold).float()


def amp_observations(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Extract standard 30-dim AMP observation matching motion capture data.
    
    Layout:
        joint_pos (12): Joint angles in radians
        root_lin_vel_b (3): Base linear velocity in body frame
        root_ang_vel_b (3): Base angular velocity in body frame
        joint_vel (12): Joint angular velocities
    Total: 30 dimensions
    """
    robot: Articulation = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos
    joint_vel = robot.data.joint_vel
    base_lin_vel = robot.data.root_lin_vel_b
    base_ang_vel = robot.data.root_ang_vel_b
    return torch.cat([joint_pos, base_lin_vel, base_ang_vel, joint_vel], dim=-1)


def height_map(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    offset: float = 0.5,
) -> torch.Tensor:
    """Extract 2D height map from RayCaster sensor."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    height_values = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset

    pattern_cfg = sensor.cfg.pattern_cfg
    if not hasattr(pattern_cfg, "resolution") or not hasattr(pattern_cfg, "size"):
        return height_values

    num_x = math.floor(pattern_cfg.size[0] / pattern_cfg.resolution + 1.0e-9) + 1
    num_y = math.floor(pattern_cfg.size[1] / pattern_cfg.resolution + 1.0e-9) + 1
    ordering = getattr(pattern_cfg, "ordering", "xy")
    if ordering == "xy":
        map_shape = (num_y, num_x)
    elif ordering == "yx":
        map_shape = (num_x, num_y)
    else:
        map_shape = (num_y, num_x)

    expected_num_rays = map_shape[0] * map_shape[1]
    if height_values.shape[-1] == expected_num_rays:
        return height_values.reshape(env.num_envs, 1, *map_shape)
    return height_values


def elevation_map(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    noise: bool = False,
) -> torch.Tensor:
    """Extract flattened [x, y, z] coordinate elevation map in yaw-aligned base frame."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    relative_pos_w = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
    sensor_quat = sensor.data.quat_w
    num_envs, num_rays, _ = relative_pos_w.shape

    if getattr(sensor.cfg, "ray_alignment", "base") == "yaw":
        from isaaclab.utils.math import yaw_quat
        sensor_quat = yaw_quat(sensor_quat)

    from isaaclab.utils.math import quat_apply_inverse
    sensor_quat = sensor_quat.unsqueeze(1).expand(num_envs, num_rays, 4).reshape(num_envs * num_rays, 4)
    sensor_coords = quat_apply_inverse(sensor_quat, relative_pos_w.reshape(num_envs * num_rays, 3))
    sensor_coords = torch.nan_to_num(sensor_coords.reshape(num_envs, num_rays, 3))

    if noise:
        if getattr(env, "_elevation_map_offset", None) is None or env._elevation_map_offset.shape != (num_envs, 1):
            env._elevation_map_offset = torch.zeros((num_envs, 1), device=env.device)
        if hasattr(env, "reset_buf"):
            reset_env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if reset_env_ids.numel() > 0:
                env._elevation_map_offset[reset_env_ids] = (
                    torch.rand((reset_env_ids.numel(), 1), device=env.device) * 0.1 - 0.05
                )
        sensor_coords[..., 2] += torch.randn_like(sensor_coords[..., 2]) * 0.03
        sensor_coords[..., 2] += env._elevation_map_offset

    sensor_coords[..., 2] = torch.clamp(sensor_coords[..., 2], min=-1.2, max=0.0)
    return sensor_coords.reshape(num_envs, num_rays * 3)

