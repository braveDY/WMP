from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Export a deterministic WMP terrain mesh.")
parser.add_argument("--terrain", choices=("rough", "finetune"), default="rough")
parser.add_argument("--terrain-name", default=None)
parser.add_argument("--rows", type=int, default=1)
parser.add_argument("--cols", type=int, default=1)
parser.add_argument("--difficulty", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=2026)
parser.add_argument("--border-width", type=float, default=1.0)
parser.add_argument("--curriculum", action="store_true")
parser.add_argument(
    "--output",
    type=Path,
    default=Path(__file__).resolve().parents[1] / "outputs" / "terrain_export",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

project_root = Path(__file__).resolve().parents[1]
local_wmp_path = project_root / "source" / "wmp"
if str(local_wmp_path) not in sys.path:
    sys.path.insert(0, str(local_wmp_path))

import numpy as np

from isaaclab.terrains import TerrainGenerator
from wmp.terrains.terrain_cfg import ROUGH_TERRAINS_CFG


def build_config():
    source_cfg = ROUGH_TERRAINS_CFG
    cfg = copy.deepcopy(source_cfg)

    if args_cli.terrain_name is not None:
        if args_cli.terrain_name not in cfg.sub_terrains:
            available = ", ".join(cfg.sub_terrains)
            raise ValueError(f"Unknown terrain '{args_cli.terrain_name}'. Available: {available}")
        cfg.sub_terrains = {args_cli.terrain_name: cfg.sub_terrains[args_cli.terrain_name]}

    if args_cli.rows < 1 or args_cli.cols < 1:
        raise ValueError("--rows and --cols must be positive integers")
    if not 0.0 <= args_cli.difficulty <= 1.0:
        raise ValueError("--difficulty must be between 0 and 1")

    cfg.seed = args_cli.seed
    cfg.num_rows = args_cli.rows
    cfg.num_cols = args_cli.cols
    cfg.border_width = args_cli.border_width
    cfg.curriculum = args_cli.curriculum
    cfg.difficulty_range = (args_cli.difficulty, args_cli.difficulty)
    cfg.use_cache = False
    cfg.color_scheme = "none"
    return cfg


def export_terrain() -> None:
    random.seed(args_cli.seed)
    np.random.seed(args_cli.seed)

    cfg = build_config()
    generator = TerrainGenerator(cfg)
    output_dir = args_cli.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    mesh_path = output_dir / "terrain.obj"
    stl_path = output_dir / "terrain.stl"
    origins_path = output_dir / "terrain_origins.csv"
    generator.terrain_mesh.export(mesh_path)
    generator.terrain_mesh.export(stl_path)

    origins = generator.terrain_origins.reshape(-1, 3)
    indices = np.indices((cfg.num_rows, cfg.num_cols)).reshape(2, -1).T
    origin_table = np.column_stack((indices, origins))
    np.savetxt(
        origins_path,
        origin_table,
        delimiter=",",
        header="row,col,x,y,z",
        comments="",
    )

    bounds = generator.terrain_mesh.bounds
    size = bounds[1] - bounds[0]
    print(f"[INFO] Terrain config: {args_cli.terrain}")
    print(f"[INFO] Terrain names: {', '.join(cfg.sub_terrains)}")
    print(f"[INFO] Seed: {cfg.seed}")
    print(f"[INFO] Grid: {cfg.num_rows} rows x {cfg.num_cols} cols")
    print(f"[INFO] Mesh: {len(generator.terrain_mesh.vertices)} vertices, {len(generator.terrain_mesh.faces)} faces")
    print(f"[INFO] Bounds: min={bounds[0].tolist()}, max={bounds[1].tolist()}")
    print(f"[INFO] Size: {size.tolist()}")
    print(f"[SUCCESS] OBJ: {mesh_path}")
    print(f"[SUCCESS] STL: {stl_path}")
    print(f"[SUCCESS] Origins: {origins_path}")
    sys.stdout.flush()


if __name__ == "__main__":
    try:
        export_terrain()
    finally:
        simulation_app.close()
