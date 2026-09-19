from __future__ import annotations

from random import randint
from typing import TYPE_CHECKING

import numpy as np

from isaaclab.terrains.height_field.utils import height_field_to_mesh

if TYPE_CHECKING:
    from . import loco_hf_terrains_cfg


@height_field_to_mesh
def stones_bridge_terrain(difficulty: float, cfg: loco_hf_terrains_cfg.HfStonesBridgeTerrainCfg) -> np.ndarray:
    stone_width = cfg.stone_width_range[1] - difficulty * (cfg.stone_width_range[1] - cfg.stone_width_range[0])
    stone_length = cfg.stone_length_range[1] - difficulty * (cfg.stone_length_range[1] - cfg.stone_length_range[0])
    stone_distance = cfg.stone_distance_range[0] + difficulty * (
        cfg.stone_distance_range[1] - cfg.stone_distance_range[0]
    )
    stone_lateral_distance = cfg.stone_lateral_distance_range[0] + difficulty * (
        cfg.stone_lateral_distance_range[1] - cfg.stone_lateral_distance_range[0]
    )

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    stone_distance = int(stone_distance / cfg.horizontal_scale)
    stone_lateral_distance = int(stone_lateral_distance / cfg.horizontal_scale)
    stone_width = int(stone_width / cfg.horizontal_scale)
    stone_length = int(stone_length / cfg.horizontal_scale)
    stone_height_max = int(cfg.stone_height_max / cfg.vertical_scale)
    holes_depth = int(cfg.holes_depth / cfg.vertical_scale)
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)

    stone_height_range = np.arange(-stone_height_max - 1, stone_height_max, step=1)
    hf_raw = np.full((width_pixels, length_pixels), holes_depth)

    start_x = stone_distance
    while start_x < width_pixels:
        stop_x = min(width_pixels, start_x + stone_width)
        start_y = (length_pixels - stone_length) // 2 + np.random.choice(
            [-stone_lateral_distance, stone_lateral_distance]
        )
        stop_y = start_y + stone_length
        hf_raw[start_x:stop_x, start_y:stop_y] = np.random.choice(stone_height_range)
        start_x = stop_x + stone_distance

    start_y = stone_distance
    while start_y < length_pixels:
        stop_y = min(length_pixels, start_y + stone_width)
        start_x = (width_pixels - stone_length) // 2 + np.random.choice(
            [-stone_lateral_distance, stone_lateral_distance]
        )
        stop_x = start_x + stone_length
        hf_raw[start_x:stop_x, start_y:stop_y] = np.random.choice(stone_height_range)
        start_y = stop_y + stone_distance

    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0

    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def double_column_stakes_terrain(
    difficulty: float, cfg: loco_hf_terrains_cfg.HfDoubleColumnStakesTerrainCfg
) -> np.ndarray:
    stake_side = cfg.stake_side_range[1] - difficulty * (cfg.stake_side_range[1] - cfg.stake_side_range[0])
    stake_gap = cfg.stake_gap_range[0] + difficulty * (cfg.stake_gap_range[1] - cfg.stake_gap_range[0])
    column_gap = cfg.column_gap_range[0] + difficulty * (cfg.column_gap_range[1] - cfg.column_gap_range[0])

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    stake_side_px = max(1, int(stake_side / cfg.horizontal_scale))
    stake_gap_px = max(0, int(stake_gap / cfg.horizontal_scale))
    column_gap_px = max(0, int(column_gap / cfg.horizontal_scale))
    column_jitter_px = max(0, int(cfg.column_jitter / cfg.horizontal_scale))
    stake_height_max_px = max(0, int(cfg.stake_height_max / cfg.vertical_scale))
    holes_depth_px = int(cfg.holes_depth / cfg.vertical_scale)
    platform_width_px = max(1, int(cfg.platform_width / cfg.horizontal_scale))

    hf_raw = np.full((width_pixels, length_pixels), holes_depth_px, dtype=float)
    half_lower = stake_side_px // 2
    half_upper = stake_side_px - half_lower
    center_offset_px = stake_side_px + column_gap_px
    center_x = width_pixels // 2
    center_y = length_pixels // 2

    if getattr(cfg, "seed", None) is not None:
        difficulty_key = int(round(float(difficulty) * 1_000_000.0))
        local_seed = (int(cfg.seed) * 1_000_003 + difficulty_key) % (2**32)
        rng = np.random.default_rng(local_seed)
    else:
        rng = np.random.default_rng()

    stake_height_values = (
        np.arange(-stake_height_max_px, stake_height_max_px + 1) if stake_height_max_px > 0 else np.array([0], dtype=int)
    )

    def paint_square(cx: int, cy: int, value: int) -> None:
        if cx < 0 or cx >= width_pixels or cy < 0 or cy >= length_pixels:
            return
        x1 = max(0, cx - half_lower)
        x2 = min(width_pixels, cx + half_upper)
        y1 = max(0, cy - half_lower)
        y2 = min(length_pixels, cy + half_upper)
        hf_raw[x1:x2, y1:y2] = value

    def place_column_pair(primary_pos: int, along_x: bool) -> None:
        if along_x:
            axis_limit_low = half_lower
            axis_limit_high = length_pixels - half_upper
            base_offset = max(center_offset_px // 2, half_lower)
            for sign in (-1, 1):
                jitter = rng.integers(-column_jitter_px, column_jitter_px + 1) if column_jitter_px > 0 else 0
                cy = int(np.clip(center_y + sign * base_offset + jitter, axis_limit_low, axis_limit_high))
                paint_square(primary_pos, cy, int(rng.choice(stake_height_values)))
        else:
            axis_limit_low = half_lower
            axis_limit_high = width_pixels - half_upper
            base_offset = max(center_offset_px // 2, half_lower)
            for sign in (-1, 1):
                jitter = rng.integers(-column_jitter_px, column_jitter_px + 1) if column_jitter_px > 0 else 0
                cx = int(np.clip(center_x + sign * base_offset + jitter, axis_limit_low, axis_limit_high))
                paint_square(cx, primary_pos, int(rng.choice(stake_height_values)))

    def extend_from_edge(along_x: bool) -> None:
        start = 0
        step = stake_gap_px + stake_side_px
        while 0 <= start < width_pixels:
            place_column_pair(int(start), along_x)
            start += step

    extend_from_edge(along_x=True)
    extend_from_edge(along_x=False)

    x1 = (width_pixels - platform_width_px) // 2
    x2 = (width_pixels + platform_width_px) // 2
    y1 = (length_pixels - platform_width_px) // 2
    y2 = (length_pixels + platform_width_px) // 2
    hf_raw[x1:x2, y1:y2] = 0

    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def concentric_gap_terrain(difficulty: float, cfg: loco_hf_terrains_cfg.HfConcentricGapTerrainCfg) -> np.ndarray:
    gap_depth = int(2.0 / cfg.vertical_scale)
    gap_width = cfg.gap_width_range[0] + difficulty * (cfg.gap_width_range[1] - cfg.gap_width_range[0])
    gap_width = int(gap_width / cfg.horizontal_scale)
    ground_width = cfg.ground_width_range[0] + (1.0 - difficulty) * (
        cfg.ground_width_range[1] - cfg.ground_width_range[0]
    )
    ground_width = int(ground_width / cfg.horizontal_scale)
    ground_height_max = int(cfg.ground_height_max / cfg.vertical_scale)
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)

    hf_raw = np.zeros((width_pixels, length_pixels))
    start_x, start_y = 0, 0
    stop_x, stop_y = width_pixels, length_pixels
    is_gap = True
    while (stop_x - start_x) > platform_width and (stop_y - start_y) > platform_width:
        if is_gap:
            hf_raw[start_x:stop_x, start_y:stop_y] = -gap_depth
            start_x += gap_width
            stop_x -= gap_width
            start_y += gap_width
            stop_y -= gap_width
        else:
            hf_raw[start_x:stop_x, start_y:stop_y] = randint(-ground_height_max, ground_height_max)
            start_x += ground_width
            stop_x -= ground_width
            start_y += ground_width
            stop_y -= ground_width
        is_gap = not is_gap

    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0
    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def alternate_column_stakes_terrain(
    difficulty: float, cfg: loco_hf_terrains_cfg.HfAlternateColumnStakesTerrainCfg
) -> np.ndarray:
    stake_side = cfg.stake_side_range[1] - difficulty * (cfg.stake_side_range[1] - cfg.stake_side_range[0])
    stake_gap = cfg.stake_gap_range[0] + difficulty * (cfg.stake_gap_range[1] - cfg.stake_gap_range[0])
    column_gap = cfg.column_gap_range[1] - difficulty * (cfg.column_gap_range[1] - cfg.column_gap_range[0])

    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    stake_side_px = max(1, int(stake_side / cfg.horizontal_scale))
    stake_gap_px = max(0, int(stake_gap / cfg.horizontal_scale))
    column_gap_px = max(0, int(column_gap / cfg.horizontal_scale))
    column_jitter_px = max(0, int(cfg.column_jitter / cfg.horizontal_scale))
    stake_height_max_px = max(0, int(cfg.stake_height_max / cfg.vertical_scale))
    holes_depth_px = int(cfg.holes_depth / cfg.vertical_scale)
    platform_width_px = max(1, int(cfg.platform_width / cfg.horizontal_scale))

    hf_raw = np.full((width_pixels, length_pixels), holes_depth_px, dtype=float)
    half_lower = stake_side_px // 2
    half_upper = stake_side_px - half_lower

    if getattr(cfg, "seed", None) is not None:
        difficulty_key = int(round(float(difficulty) * 1_000_000.0))
        local_seed = (int(cfg.seed) * 1_000_003 + difficulty_key) % (2**32)
        rng = np.random.default_rng(local_seed)
    else:
        rng = np.random.default_rng()

    stake_height_values = (
        np.arange(-stake_height_max_px, stake_height_max_px + 1) if stake_height_max_px > 0 else np.array([0], dtype=int)
    )

    def paint_square(cx: int, cy: int, value: int) -> None:
        if cx < 0 or cx >= width_pixels or cy < 0 or cy >= length_pixels:
            return
        x1 = max(0, cx - half_lower)
        x2 = min(width_pixels, cx + half_upper)
        y1 = max(0, cy - half_lower)
        y2 = min(length_pixels, cy + half_upper)
        hf_raw[x1:x2, y1:y2] = value

    def place_alternate_columns(start_pos: int, along_x: bool) -> None:
        offset = column_gap_px // 2
        step = stake_gap_px + stake_side_px
        while start_pos < (width_pixels if along_x else length_pixels):
            jitter = rng.integers(-column_jitter_px, column_jitter_px + 1) if column_jitter_px > 0 else 0
            height_value = int(rng.choice(stake_height_values))
            if along_x:
                paint_square(start_pos, (length_pixels // 2) + offset + jitter, height_value)
            else:
                paint_square((width_pixels // 2) + offset + jitter, start_pos, height_value)
            offset = -offset
            start_pos += step

    place_alternate_columns(0, along_x=True)
    place_alternate_columns(0, along_x=False)

    x1 = (width_pixels - platform_width_px) // 2
    x2 = (width_pixels + platform_width_px) // 2
    y1 = (length_pixels - platform_width_px) // 2
    y2 = (length_pixels + platform_width_px) // 2
    hf_raw[x1:x2, y1:y2] = 0

    return np.rint(hf_raw).astype(np.int16)
