# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Constants and unit conversion factors for food systems model.

This module contains all conversion factors, constants, and supported
unit definitions used throughout the model building process.
"""

# Unit conversion factors
KM3_PER_M3 = 1e-9  # convert cubic metres to cubic kilometres
TONNE_TO_MEGATONNE = 1e-6  # convert tonnes to megatonnes
KG_TO_MEGATONNE = 1e-9  # convert kilograms to megatonnes
KCAL_TO_MCAL = 1e-6  # convert kilocalories to megacalories
KCAL_PER_100G_TO_MCAL_PER_TONNE = 1e-2  # kcal/100g to Mcal per tonne of food
DAYS_PER_YEAR = 365
N2O_N_TO_N2O = 44.0 / 28.0  # molecular weight ratio to convert N2O-N to N2O

# Supported nutrition unit definitions
SUPPORTED_NUTRITION_UNITS = {
    "g/100g": {"kind": "mass", "efficiency_factor": TONNE_TO_MEGATONNE},
    "kcal/100g": {
        "kind": "energy",
        "efficiency_factor": KCAL_PER_100G_TO_MCAL_PER_TONNE,
    },
}
