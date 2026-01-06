# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Validation checks for optimal taxes configuration."""

from pathlib import Path

import yaml


def _optimal_taxes_enabled(config: dict, scenario_defs: dict) -> bool:
    if bool(config["optimal_taxes"]["enabled"]):
        return True

    for overrides in scenario_defs.values():
        if not isinstance(overrides, dict):
            continue
        ot_cfg = overrides.get("optimal_taxes", {})
        if isinstance(ot_cfg, dict) and bool(ot_cfg.get("enabled", False)):
            return True

    return False


def validate_optimal_taxes(config: dict, project_root: Path) -> None:
    """Ensure optimal taxes runs have required scenarios defined."""
    scenario_defs_path = config.get("scenario_defs")
    if not scenario_defs_path:
        if bool(config["optimal_taxes"]["enabled"]):
            raise ValueError(
                "optimal_taxes enabled but scenario_defs is not configured; "
                "required scenarios: 'baseline', 'optimize', 'extract_taxes', 'apply_taxes'"
            )
        return

    scenario_defs_file = project_root / scenario_defs_path
    if not scenario_defs_file.exists():
        raise FileNotFoundError(
            f"scenario_defs not found at '{scenario_defs_file.as_posix()}'"
        )

    with open(scenario_defs_file, encoding="utf-8") as f:
        scenario_defs = yaml.safe_load(f) or {}

    if not _optimal_taxes_enabled(config, scenario_defs):
        return

    required_scenarios = {"baseline", "optimize", "extract_taxes", "apply_taxes"}
    missing = required_scenarios - set(scenario_defs.keys())
    if missing:
        raise ValueError(
            "optimal_taxes enabled but scenario_defs is missing required scenarios: "
            f"{missing}"
        )
