# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to print all the available environments in Isaac Lab.
"""

import argparse
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="List Isaac Lab environments.")
parser.add_argument("--keyword", type=str, default=None, help="Keyword to filter environments.")
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

"""Rest everything follows."""

_local_wmp_path = Path(__file__).resolve().parents[1] / "source" / "wmp"
if _local_wmp_path.is_dir() and str(_local_wmp_path) not in sys.path:
    sys.path.insert(0, str(_local_wmp_path))

import gymnasium as gym
from prettytable import PrettyTable

import wmp  # noqa: F401


def main():
    """Print all environments registered in `wmp` extension."""
    table = PrettyTable(["S. No.", "Task Name", "Entry Point", "Config"])
    table.title = "Available Environments in Isaac Lab"
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"

    index = 0
    for task_spec in gym.registry.values():
        if args_cli.keyword is not None and args_cli.keyword.lower() not in task_spec.id.lower():
            continue
        config_entry_point = task_spec.kwargs.get("env_cfg_entry_point", "None")
        table.add_row([index, task_spec.id, task_spec.entry_point, config_entry_point])
        index += 1

    print(table)


if __name__ == "__main__":
    main()
    simulation_app.close()
