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
- Global constraints named: food_group_equal_{group}_store_{group}_{country}
"""

import logging

import pandas as pd
import pypsa

from workflow.scripts.logging_config import setup_script_logging

logger = logging.getLogger(__name__)


def _parse_constraint_name(name: str) -> tuple[str, str] | None:
    """Parse constraint name to extract group and country.

    Constraint names follow: food_group_equal_{group}_store_{group}_{country}
    where {group} may contain underscores (e.g., nuts_seeds).
    """
    prefix = "food_group_equal_"
    if not name.startswith(prefix):
        return None

    remainder = name[len(prefix) :]

    # Find "_store_" which separates the two occurrences of {group}
    store_idx = remainder.find("_store_")
    if store_idx == -1:
        return None

    group = remainder[:store_idx]

    # After "_store_{group}_" comes the country code (3 uppercase letters)
    after_store = remainder[store_idx + len("_store_") :]
    expected_prefix = f"{group}_"
    if not after_store.startswith(expected_prefix):
        return None

    country = after_store[len(expected_prefix) :]
    if len(country) != 3 or not country.isupper():
        return None

    return group, country


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
    gc_df = n.global_constraints

    # Filter for food group equality constraints
    food_group_constraints = gc_df[gc_df.index.str.startswith("food_group_equal_")]

    if food_group_constraints.empty:
        raise ValueError(
            "No food group equality constraints found in the network. "
            "Ensure the model was solved with validation.enforce_gdd_baseline=true"
        )

    records = []
    for name, row in food_group_constraints.iterrows():
        # Parse constraint name: food_group_equal_{group}_store_{group}_{country}
        parsed = _parse_constraint_name(str(name))
        if parsed is None:
            logger.warning("Could not parse constraint name: %s", name)
            continue

        group, country = parsed

        # Extract dual value (shadow price)
        # Units: bnUSD/Mt (model internal units)
        dual = float(row["mu"]) if "mu" in row.index else 0.0

        records.append(
            {
                "group": group,
                "country": country,
                "value_bnusd_per_mt": dual,
            }
        )

    if not records:
        raise ValueError(
            "No valid food group constraints could be parsed. "
            f"Found constraints: {list(food_group_constraints.index[:5])}"
        )

    df = pd.DataFrame.from_records(records)
    df["adjustment_bnusd_per_mt"] = -df["value_bnusd_per_mt"]
    logger.info(
        "Extracted consumer values for %d (group, country) pairs",
        len(df),
    )
    return df


if __name__ == "__main__":
    logger = setup_script_logging(log_file=snakemake.log[0] if snakemake.log else None)  # type: ignore[name-defined]

    logger.info("Loading solved network from %s", snakemake.input.network)
    n = pypsa.Network(snakemake.input.network)

    df = extract_consumer_values(n)

    output_path = snakemake.output.consumer_values
    df.to_csv(output_path, index=False)
    logger.info("Saved consumer values to %s", output_path)
