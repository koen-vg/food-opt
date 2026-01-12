# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Utilities for Snakemake workflow execution."""

from pathlib import Path

import yaml


def _recursive_update(target: dict, source: dict) -> dict:
    """Recursively update the target dictionary with the source dictionary."""
    for key, value in source.items():
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            _recursive_update(target[key], value)
        else:
            target[key] = value
    return target


def load_scenarios(config: dict) -> dict:
    """Load scenario definitions from the file specified in the config."""
    scenario_path = config.get("scenario_defs")
    if not scenario_path:
        raise ValueError("Config key 'scenario_defs' must be set to use scenarios.")

    # Resolve path relative to project root (assuming script runs in working dir or subfolder)
    # We try to find the file.
    path = Path(scenario_path)
    if not path.exists():
        # Fallback if running from a subdirectory
        path = Path("../") / scenario_path

    if not path.exists():
        raise FileNotFoundError(
            f"Scenario definitions file not found at {scenario_path}"
        )

    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def apply_scenario_config(config: dict, scenario_name: str) -> None:
    """Apply scenario config overrides in-place.

    Parameters
    ----------
    config : dict
        The Snakemake config dictionary (will be modified in-place)
    scenario_name : str
        The scenario name (e.g., "HG", "HighGHG")
    """
    if not scenario_name:
        return

    scenarios = load_scenarios(config)

    if scenario_name not in scenarios:
        raise ValueError(
            f"Scenario '{scenario_name}' not found in scenario definitions."
        )

    overrides = scenarios[scenario_name]
    _recursive_update(config, overrides)
