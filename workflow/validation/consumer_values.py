# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Validation checks for consumer values configuration."""

from pathlib import Path

import yaml


def _consumer_values_enabled(config: dict, scenario_defs: dict) -> bool:
    if bool(config["consumer_values"]["enabled"]):
        return True

    for overrides in scenario_defs.values():
        if not isinstance(overrides, dict):
            continue
        cv_cfg = overrides.get("consumer_values", {})
        if isinstance(cv_cfg, dict) and bool(cv_cfg.get("enabled", False)):
            return True

    return False


def validate_consumer_values(config: dict, project_root: Path) -> None:
    """Ensure consumer values runs have a baseline scenario defined."""
    scenario_defs_path = config.get("scenario_defs")
    if not scenario_defs_path:
        if bool(config["consumer_values"]["enabled"]):
            raise ValueError(
                "consumer_values enabled but scenario_defs is not configured; "
                "a baseline scenario is required"
            )
        return

    scenario_defs_file = project_root / scenario_defs_path
    if not scenario_defs_file.exists():
        raise FileNotFoundError(
            f"scenario_defs not found at '{scenario_defs_file.as_posix()}'"
        )

    with open(scenario_defs_file, encoding="utf-8") as f:
        scenario_defs = yaml.safe_load(f) or {}

    if not _consumer_values_enabled(config, scenario_defs):
        return

    if "baseline" not in scenario_defs:
        raise ValueError(
            "consumer_values enabled but scenario_defs does not define a 'baseline' scenario"
        )
