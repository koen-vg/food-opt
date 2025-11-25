# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Utilities for Snakemake workflow execution."""


def parse_objective_wildcard(wildcard_value: str) -> dict:
    """Parse objective wildcard and return config overrides.

    The objective wildcard controls which costs are included in the
    objective function. The wildcard value contains flags:

    - H: Health impacts (years of life lost)
    - G: GHG pricing (greenhouse gas emissions)

    Parameters
    ----------
    wildcard_value : str
        The objective wildcard value containing flags (e.g., "HG", "H", "G", "")
        Note: The wildcard captures just the flags, not the "obj-" prefix

    Returns
    -------
    dict
        A dictionary with config overrides for health and emissions settings

    Examples
    --------
    >>> parse_objective_wildcard("HG")
    {'health': {'enabled': True}, 'emissions': {'ghg_pricing_enabled': True}}

    >>> parse_objective_wildcard("H")
    {'health': {'enabled': True}, 'emissions': {'ghg_pricing_enabled': False}}

    >>> parse_objective_wildcard("")
    {'health': {'enabled': False}, 'emissions': {'ghg_pricing_enabled': False}}
    """
    # The wildcard value is just the flags (e.g., "HG", "H", "G", "")
    flags = wildcard_value

    return {
        "health": {"enabled": "H" in flags},
        "emissions": {"ghg_pricing_enabled": "G" in flags},
    }


def apply_objective_config(config: dict, objective_wildcard: str) -> None:
    """Apply objective wildcard config overrides in-place.

    This function is designed to be called from Snakemake scripts to override
    the config based on the objective wildcard.

    Parameters
    ----------
    config : dict
        The Snakemake config dictionary (will be modified in-place)
    objective_wildcard : str
        The objective wildcard value (e.g., "obj-HG", "obj-H", "obj-")
    """
    overrides = parse_objective_wildcard(objective_wildcard)

    for section, values in overrides.items():
        if section not in config:
            config[section] = {}
        config[section].update(values)
