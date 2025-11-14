# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Food systems optimization model builder.

Modular package for constructing PyPSA networks representing global food production,
conversion, trade, and nutrition constraints.
"""

# Re-export submodules for convenience
from . import (
    animals,
    biomass,
    constants,
    crops,
    food,
    infrastructure,
    nutrition,
    primary_resources,
    trade,
    utils,
)

__all__ = [
    "animals",
    "biomass",
    "constants",
    "crops",
    "food",
    "infrastructure",
    "nutrition",
    "primary_resources",
    "trade",
    "utils",
]
