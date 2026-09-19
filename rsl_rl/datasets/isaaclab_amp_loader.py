# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import numpy as np
import torch


class IsaacLabAMPLoader:
    """Loads IsaacLab-format AMP motions and samples expert transitions.

    Source frame layout:
    root_pos(3), root_rot_rpy(3), root_lin_vel_b(3), root_ang_vel_b(3),
    joint_pos(12), joint_vel(12), foot_contact(4).

    AMP state:
    joint_pos(12), root_lin_vel_b(3), root_ang_vel_b(3), joint_vel(12), foot_contact(4).
    """

    ROOT_LIN_VEL_B_SLICE = slice(6, 9)
    ROOT_ANG_VEL_B_SLICE = slice(9, 12)
    JOINT_POS_SLICE = slice(12, 24)
    JOINT_VEL_SLICE = slice(24, 36)
    FOOT_CONTACT_SLICE = slice(36, 40)

    def __init__(
        self,
        motion_files: str | list[str],
        device: str,
        time_between_frames: float = 0.02,
        preload_transitions: bool = True,
        num_preload_transitions: int = 100000,
        motion_fps: float | None = None,
    ):
        self.device = device
        self.time_between_frames = time_between_frames
        self.motion_files = self._resolve_motion_files(motion_files)
        self.motion_fps = float(motion_fps or self._infer_fps(self.motion_files) or 50.0)
        self.trajectories = self._load_trajectories(self.motion_files)
        if not self.trajectories:
            raise ValueError(f"No AMP motion trajectories found from: {motion_files}")

        self.frame_dt = 1.0 / self.motion_fps
        self.transition_frame_stride = max(1, int(round(self.time_between_frames / self.frame_dt)))
        self.trajectory_lengths = torch.tensor([traj.shape[0] for traj in self.trajectories], device=self.device)
        self.trajectory_weights = (self.trajectory_lengths.float() - self.transition_frame_stride).clamp(min=1.0)
        self.trajectory_weights /= self.trajectory_weights.sum()

        self.preload_transitions = preload_transitions
        if self.preload_transitions:
            self.preloaded_s, self.preloaded_s_next = self.sample(num_preload_transitions)

        print(
            "[INFO] Loaded IsaacLab AMP motion data: "
            f"{self.num_motions} motion(s), obs_dim={self.observation_dim}, fps={self.motion_fps:g}, "
            f"dt={self.frame_dt:.4f}s, transition_stride={self.transition_frame_stride}, "
            f"preloaded_transitions={num_preload_transitions if self.preload_transitions else 0}, "
            f"files={len(self.motion_files)}"
        )

    @staticmethod
    def _resolve_motion_files(motion_files: str | list[str]) -> list[str]:
        if isinstance(motion_files, (str, os.PathLike)):
            motion_files = str(motion_files)
            if any(char in motion_files for char in "*?[]"):
                return sorted(glob.glob(motion_files))
            return [motion_files]
        return sorted(str(path) for path in motion_files)

    def _load_trajectories(self, motion_files: list[str]) -> list[torch.Tensor]:
        trajectories = []
        for motion_file in motion_files:
            path = Path(motion_file)
            if path.suffix == ".npz":
                trajectories.extend(self._load_npz(path))
            elif path.suffix == ".npy":
                data = np.load(path).astype(np.float32)
                trajectories.append(self._select_amp_state(data))
            elif path.suffix == ".txt":
                trajectories.append(self._load_txt_json(path))
            else:
                raise ValueError(f"Unsupported IsaacLab AMP motion file type: {motion_file}")
        return [torch.as_tensor(traj, dtype=torch.float32, device=self.device) for traj in trajectories if len(traj) > 1]

    def _load_txt_json(self, path: Path) -> np.ndarray:
        with open(path, "r") as f:
            motion_json = json.load(f)
        frames = np.array(motion_json["Frames"], dtype=np.float32)
        # Frames format in WMP: 60-dim
        # 0:3 pos, 3:7 rot, 7:19 joint_pos, 19:31 tar_toe_pos_local, 31:34 lin_vel, 34:37 ang_vel, 37:49 joint_vel, 49:61 tar_toe_vel_local
        # Standard AMP state: joint_pos(12), root_lin_vel_b(3), root_ang_vel_b(3), joint_vel(12) [30-dim]
        joint_pos = frames[:, 7:19]
        lin_vel = frames[:, 31:34]
        ang_vel = frames[:, 34:37]
        joint_vel = frames[:, 37:49]
        return np.concatenate([joint_pos, lin_vel, ang_vel, joint_vel], axis=-1)

    def _infer_fps(self, motion_files: list[str]) -> float | None:
        for motion_file in motion_files:
            path = Path(motion_file)
            if path.suffix == ".npz":
                data = np.load(path)
                if "fps" in data:
                    return float(data["fps"])
            elif path.suffix == ".txt":
                try:
                    with open(path, "r") as f:
                        motion_json = json.load(f)
                    if "FrameDuration" in motion_json and float(motion_json["FrameDuration"]) > 0:
                        return 1.0 / float(motion_json["FrameDuration"])
                except Exception:
                    pass
        return None

    def _select_amp_state(self, frames: np.ndarray) -> np.ndarray:
        if frames.ndim != 2 or frames.shape[1] < 40:
            raise ValueError(f"Expected Go2 AMP motion frames with shape [T, >=40], got {frames.shape}")
        return np.concatenate(
            [
                frames[:, self.JOINT_POS_SLICE],
                frames[:, self.ROOT_LIN_VEL_B_SLICE],
                frames[:, self.ROOT_ANG_VEL_B_SLICE],
                frames[:, self.JOINT_VEL_SLICE],
                frames[:, self.FOOT_CONTACT_SLICE],
            ],
            axis=-1,
        ).astype(np.float32)

    @staticmethod
    def _select_amp_state_from_named(
        data: np.lib.npyio.NpzFile, start: int | None = None, end: int | None = None
    ) -> np.ndarray:
        if "foot_contact" not in data.files:
            raise ValueError(
                "Go2 AMP motion data must contain a 'foot_contact' array with four ordered foot contact channels"
            )
        sl = slice(start, end)
        return np.concatenate(
            [
                data["joint_pos"][sl].astype(np.float32),
                data["root_lin_vel_b"][sl].astype(np.float32),
                data["root_ang_vel_b"][sl].astype(np.float32),
                data["joint_vel"][sl].astype(np.float32),
                data["foot_contact"][sl].astype(np.float32),
            ],
            axis=-1,
        )

    def _sample_indices(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        traj_idxs = torch.multinomial(self.trajectory_weights, batch_size, replacement=True)
        frame_idxs = torch.empty(batch_size, dtype=torch.long, device=self.device)
        for traj_idx in torch.unique(traj_idxs).tolist():
            mask = traj_idxs == traj_idx
            count = int(mask.sum().item())
            max_start = int(self.trajectory_lengths[traj_idx].item()) - self.transition_frame_stride
            frame_idxs[mask] = torch.randint(max_start, (count,), device=self.device)
        return traj_idxs, frame_idxs

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        traj_idxs, frame_idxs = self._sample_indices(batch_size)
        states = torch.empty(batch_size, self.observation_dim, device=self.device)
        next_states = torch.empty_like(states)
        for traj_idx in torch.unique(traj_idxs).tolist():
            mask = traj_idxs == traj_idx
            idxs = frame_idxs[mask]
            trajectory = self.trajectories[traj_idx]
            states[mask] = trajectory[idxs]
            next_states[mask] = trajectory[idxs + self.transition_frame_stride]
        return states, next_states

    def feed_forward_generator(self, num_mini_batches: int, mini_batch_size: int):
        for _ in range(num_mini_batches):
            if self.preload_transitions:
                idxs = torch.randint(self.preloaded_s.shape[0], (mini_batch_size,), device=self.device)
                yield self.preloaded_s[idxs], self.preloaded_s_next[idxs]
            else:
                yield self.sample(mini_batch_size)

    @property
    def observation_dim(self) -> int:
        return int(self.trajectories[0].shape[-1]) if self.trajectories else 30

    @property
    def num_motions(self) -> int:
        return len(self.trajectories)
