# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Extract optimal taxes from food group equality constraint duals.

Taxes represent the Pigouvian tax (positive) or subsidy (negative) needed
to incentivize the optimal consumption pattern derived in Stage 1.

The dual variable (shadow price) of each consumption constraint indicates
the marginal cost of forcing that consumption level when only production
costs are considered. This is the tax/subsidy that would make the
production-cost-only equilibrium match the health/GHG-optimal outcome.

Interpretation:
- Positive dual: Increasing consumption at this constraint increases production
  cost. The health/GHG-optimal consumption is higher than what pure production
  costs would yield, so a SUBSIDY is needed to encourage more consumption.
- Negative dual: Increasing consumption would decrease production cost (e.g.,
  due to economies of scale or efficient utilization). The optimal consumption
  is lower, so a TAX is needed to discourage consumption.

Therefore: tax = -dual (positive tax discourages, negative tax = subsidy encourages)
"""

import pandas as pd
import pypsa

from workflow.scripts.logging_config import setup_script_logging

# Enable new PyPSA components API
pypsa.options.api.new_components_api = True


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


def extract_optimal_taxes(n: pypsa.Network) -> pd.DataFrame:
    """Extract optimal taxes from food group equality constraint duals.

    Parameters
    ----------
    n : pypsa.Network
        Solved network with food group equality constraints.

    Returns
    -------
    pd.DataFrame
        Columns: group, country, tax_bnusd_per_mt, tax_usd_per_kg,
        adjustment_bnusd_per_mt
        Positive values indicate taxes (discourage consumption),
        negative values indicate subsidies (encourage consumption).
    """
    gc_df = n.global_constraints.static
    food_group_constraints = gc_df[gc_df.index.str.startswith("food_group_equal_")]

    if food_group_constraints.empty:
        raise ValueError(
            "No food group equality constraints found in the network. "
            "Ensure the model was solved with fixed consumption constraints."
        )

    records = []
    for name, row in food_group_constraints.iterrows():
        parsed = _parse_constraint_name(str(name))
        if parsed is None:
            logger.warning("Could not parse constraint name: %s", name)
            continue

        group, country = parsed

        # Extract dual value (shadow price)
        # Units: bnUSD/Mt (model internal units)
        dual = float(row["mu"]) if "mu" in row.index else 0.0

        # Tax = -dual
        # - If dual > 0: consumption is costly to production → need subsidy (tax < 0)
        # - If dual < 0: consumption saves production cost → need tax (tax > 0)
        tax_bnusd_per_mt = -dual

        # Unit conversion: bnUSD/Mt = 1e9 USD / 1e9 kg = USD/kg
        tax_usd_per_kg = tax_bnusd_per_mt

        records.append(
            {
                "group": group,
                "country": country,
                "tax_bnusd_per_mt": tax_bnusd_per_mt,
                "tax_usd_per_kg": tax_usd_per_kg,
                "adjustment_bnusd_per_mt": tax_bnusd_per_mt,
            }
        )

    if not records:
        raise ValueError(
            "No valid food group constraints could be parsed. "
            f"Found constraints: {list(food_group_constraints.index[:5])}"
        )

    df = pd.DataFrame.from_records(records)
    logger.info("Extracted taxes for %d (group, country) pairs", len(df))
    return df


if __name__ == "__main__":
    logger = setup_script_logging(
        log_file=snakemake.log[0] if snakemake.log else None  # type: ignore[name-defined]
    )

    logger.info("Loading solved network from %s", snakemake.input.network)
    n = pypsa.Network(snakemake.input.network)

    df = extract_optimal_taxes(n)

    output_path = snakemake.output.taxes
    df.to_csv(output_path, index=False)
    logger.info("Saved optimal taxes to %s", output_path)
