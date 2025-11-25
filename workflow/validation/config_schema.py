# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Validation of configuration schema and values."""

from pathlib import Path


def validate_config_schema(config: dict, _project_root: Path) -> None:
    """Validate configuration schema and values.

    Parameters
    ----------
    config:
        The merged Snakemake configuration dictionary.
    _project_root:
        Root directory of the repository (unused).

    Raises
    ------
    KeyError
        If required configuration keys are missing.
    ValueError
        If any configuration value is invalid.
    """
    # Validate harvest_area_source exists and has valid value
    if "validation" not in config:
        raise KeyError("Missing 'validation' section in config")

    validation_cfg = config["validation"]

    if "harvest_area_source" not in validation_cfg:
        raise KeyError("Missing 'validation.harvest_area_source' in config")

    harvest_source = validation_cfg["harvest_area_source"]
    valid_sources = ("gaez", "cropgrids")
    if harvest_source not in valid_sources:
        raise ValueError(
            f"validation.harvest_area_source must be one of {valid_sources}, "
            f"got '{harvest_source}'"
        )
