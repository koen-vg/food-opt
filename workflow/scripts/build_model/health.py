# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Health outcome representation in the PyPSA network.

The health objective is encoded via PyPSA Stores so that the solved network
contains explicit assets for years of life lost (YLL). Each health cluster gets
its own bus, and each (cluster, cause) pair gets a store whose energy level
represents *million YLL*. The store's capital cost is the monetary value per
million YLL, ensuring the contribution shows up in standard PyPSA statistics
instead of a manual objective adjustment.
"""

from collections.abc import Mapping
import logging
import math
from pathlib import Path

import pandas as pd
import pypsa

from .. import constants

logger = logging.getLogger(__name__)


def _load_health_tables(
    cluster_summary_path: str | Path, cluster_cause_path: str | Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cluster_summary = pd.read_csv(cluster_summary_path)
    cluster_cause = pd.read_csv(cluster_cause_path)

    if "health_cluster" not in cluster_summary.columns:
        raise ValueError("cluster_summary must contain 'health_cluster'")
    if not {"health_cluster", "cause", "yll_base", "log_rr_total_ref"}.issubset(
        cluster_cause.columns
    ):
        raise ValueError(
            "cluster_cause must contain health_cluster, cause, yll_base, log_rr_total_ref"
        )

    cluster_summary["health_cluster"] = cluster_summary["health_cluster"].astype(int)
    cluster_cause["health_cluster"] = cluster_cause["health_cluster"].astype(int)
    cluster_cause["cause"] = cluster_cause["cause"].astype(str)

    return cluster_summary, cluster_cause


def add_health_stores(
    n: pypsa.Network,
    cluster_summary_path: str | Path,
    cluster_cause_path: str | Path,
    health_cfg: Mapping[str, object],
) -> None:
    """Add per-cluster, per-cause YLL stores to the network.

    Parameters
    ----------
    n
        Network to mutate.
    cluster_summary_path
        CSV with at least ``health_cluster`` column.
    cluster_cause_path
        CSV with columns ``health_cluster``, ``cause``, ``yll_base``,
        ``log_rr_total_ref``.
    health_cfg
        The ``health`` section from the configuration; must contain
        ``causes`` and ``value_per_yll``.
    """
    cluster_summary, cluster_cause = _load_health_tables(
        cluster_summary_path, cluster_cause_path
    )

    causes: list[str] = list(health_cfg["causes"])
    value_per_yll_usd = float(health_cfg["value_per_yll"])

    # bnUSD per million YLL = USD/YLL * USD_TO_BNUSD / YLL_TO_MILLION_YLL
    cost_per_myll = (
        value_per_yll_usd * constants.USD_TO_BNUSD / constants.YLL_TO_MILLION_YLL
    )

    n.carriers.add("health", unit="million YLL")
    n.carriers.add([f"yll_{cause}" for cause in causes], unit="million YLL")

    # Create one bus per cluster
    unique_clusters = sorted(cluster_summary["health_cluster"].unique())
    cluster_buses = [f"health:cluster:{cluster:03d}" for cluster in unique_clusters]
    n.buses.add(cluster_buses, carrier="health", health_cluster=unique_clusters)

    # Add one generator per cluster to supply YLL (satisfies bus balance)
    gen_names = [f"supply:health:cluster{cluster:03d}" for cluster in unique_clusters]
    n.generators.add(
        gen_names,
        bus=cluster_buses,
        carrier="health",
        p_nom_extendable=True,
        marginal_cost=0,
        health_cluster=unique_clusters,
    )

    # Build stores from the cluster_cause table so we retain per-cause metadata
    # Filter to configured causes and extract data in one pass
    mask = cluster_cause["cause"].isin(causes)
    filtered = cluster_cause[mask]

    if filtered.empty:
        logger.info("No health stores to add (empty cluster_cause table)")
        return

    # Build lists for bulk add
    clusters = filtered["health_cluster"].astype(int)
    cause_names = filtered["cause"].astype(str)

    store_names = [
        f"store:yll:{cause}:cluster{cluster:03d}"
        for cause, cluster in zip(cause_names, clusters)
    ]
    store_buses = [f"health:cluster:{cluster:03d}" for cluster in clusters]
    carriers = [f"yll_{cause}" for cause in cause_names]

    n.stores.add(
        store_names,
        bus=store_buses,
        carrier=carriers,
        e_nom_extendable=True,
        marginal_cost_storage=cost_per_myll,
        health_cluster=clusters.tolist(),
        cause=cause_names.tolist(),
        yll_base=filtered["yll_base"].astype(float).tolist(),
        rr_ref=[
            math.exp(log_rr) for log_rr in filtered["log_rr_total_ref"].astype(float)
        ],
    )

    logger.info(
        "Added %d health stores and %d generators across %d clusters and %d causes",
        len(store_names),
        len(gen_names),
        len(unique_clusters),
        len(causes),
    )
