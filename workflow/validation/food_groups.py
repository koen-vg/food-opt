# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Pydantic + Pandera powered validation for food group inputs."""

from collections.abc import Iterable
from pathlib import Path

import pandas as pd
from pandera import Column, DataFrameSchema
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

FOOD_GROUP_SCHEMA = DataFrameSchema(
    {
        "food": Column(str, nullable=False, coerce=True),
        "group": Column(str, nullable=False, coerce=True),
    },
    strict=True,
    coerce=True,
)


class FoodGroupsConfig(BaseModel):
    """Typed representation of the ``food_groups`` config section."""

    model_config = ConfigDict(extra="allow")

    included: list[str]

    @field_validator("included")
    @classmethod
    def _normalize_groups(cls, value: list[str]) -> list[str]:
        """Ensure entries are unique, stripped strings."""

        if not value:
            raise ValueError("must list at least one food group")

        normalized: list[str] = []
        seen: set[str] = set()
        for entry in value:
            if not isinstance(entry, str):
                raise TypeError("entries must be strings")
            group = entry.strip()
            if not group:
                raise ValueError("entries may not be empty strings")
            if group not in seen:
                normalized.append(group)
                seen.add(group)

        return normalized

    def require_groups(self, expected: Iterable[str]) -> None:
        """Raise if ``expected`` contains groups missing from ``included``."""

        missing = sorted(set(expected) - set(self.included))
        if missing:
            missing_text = ", ".join(missing)
            raise ValueError(f"missing food groups in config: {missing_text}")


def validate_food_groups(config: dict, project_root: Path) -> None:
    """Validate that config food groups cover all categories present in the CSV."""

    section = config.get("food_groups")
    if section is None:
        raise ValueError("config missing 'food_groups' section")

    try:
        food_groups_cfg = FoodGroupsConfig(**section)
    except ValidationError as exc:
        raise ValueError("invalid food_groups configuration") from exc

    csv_path = project_root / "data" / "food_groups.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"expected data file at {csv_path}")

    df = FOOD_GROUP_SCHEMA.validate(pd.read_csv(csv_path))
    categories = df["group"].dropna().astype(str).str.strip()
    food_groups_cfg.require_groups(group for group in categories.unique())
