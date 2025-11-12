# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Validation entry points for configuration and data consistency checks."""

from collections.abc import Iterable
from pathlib import Path
from typing import Callable

from .food_groups import validate_food_groups

Validator = Callable[[dict, Path], None]

_CHECKS: dict[str, Validator] = {
    "food_groups": validate_food_groups,
}


def validate(
    config: dict,
    project_root: Path | None = None,
    *,
    enabled_checks: Iterable[str] | None = None,
) -> None:
    """Run configured validation checks against the active config and data.

    Parameters
    ----------
    config:
        The merged Snakemake configuration dictionary.
    project_root:
        Root directory of the repository. Defaults to the current working directory.
    enabled_checks:
        Optional iterable of check names to run. When omitted, all registered checks
        are executed.
    """

    root = Path(project_root) if project_root else Path.cwd()
    check_names = tuple(enabled_checks) if enabled_checks else tuple(_CHECKS)

    errors: list[str] = []
    for name in check_names:
        try:
            check = _CHECKS[name]
        except KeyError as exc:
            raise KeyError(f"Unknown validation check '{name}'") from exc

        try:
            check(config, root)
        except Exception as exc:
            errors.append(f"{name}: {exc}")

    if errors:
        bullet_list = "\n".join(f" - {msg}" for msg in errors)
        raise RuntimeError(f"Validation failed:\n{bullet_list}")


__all__ = ["validate"]
