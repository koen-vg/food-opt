# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Extract consumer values (dual variables) from food group equality constraints.

Consumer values represent the marginal value of consuming one additional unit
of each food group, as revealed by the dual variables of fixed consumption
constraints. These values can be used to construct an objective function that
replicates consumer preferences.

Expects a solved network with:
- validation.enforce_gdd_baseline=True (fixed food group consumption)
- Global constraints with food_group and country columns set
"""

import logging

import pandas as pd
import pypsa

from workflow.scripts.logging_config import setup_script_logging

# Enable new PyPSA components API
pypsa.options.api.new_components_api = True

logger = logging.getLogger(__name__)


def extract_consumer_values(n: pypsa.Network) -> pd.DataFrame:
    """Extract consumer values from food group equality constraint duals.

    Parameters
    ----------
    n : pypsa.Network
        Solved network with food group equality constraints.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: group, country, value_bnusd_per_mt,
        adjustment_bnusd_per_mt. The adjustment column is the value with
        sign flipped for direct use as a marginal cost incentive.
    """
    gc_df = n.global_constraints.static

    # Filter to food group equality constraints using the food_group column
    food_group_constraints = gc_df[
        gc_df["food_group"].notna() & gc_df.index.str.startswith("food_group_equal_")
    ]

    if food_group_constraints.empty:
        raise ValueError(
            "No food group equality constraints found in the network. "
            "Ensure the model was solved with validation.enforce_gdd_baseline=true"
        )

    # Use columns to get group and country - no name parsing needed
    groups = food_group_constraints["food_group"].astype(str)
    countries = food_group_constraints["country"].astype(str).str.upper()
    duals = food_group_constraints["mu"].fillna(0.0).astype(float)

    df = pd.DataFrame(
        {
            "group": groups.values,
            "country": countries.values,
            "value_bnusd_per_mt": duals.values,
            "adjustment_bnusd_per_mt": -duals.values,
        }
    )

    logger.info("Extracted consumer values for %d (group, country) pairs", len(df))
    return df


if __name__ == "__main__":
    logger = setup_script_logging(log_file=snakemake.log[0] if snakemake.log else None)  # type: ignore[name-defined]

    logger.info("Loading solved network from %s", snakemake.input.network)
    n = pypsa.Network(snakemake.input.network)

    df = extract_consumer_values(n)

    output_path = snakemake.output.consumer_values
    df.to_csv(output_path, index=False)
    logger.info("Saved consumer values to %s", output_path)
