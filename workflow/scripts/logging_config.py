# SPDX-FileCopyrightText: 2025 Koen van Greevenbroek
#
# SPDX-License-Identifier: GPL-3.0-or-later

"""Centralized logging configuration for all workflow scripts."""

import logging
from pathlib import Path
import sys


def setup_script_logging(log_file=None, level=logging.INFO):
    """
    Configure logging for a Snakemake script.

    Args:
        log_file: Path to log file (from snakemake.log[0]), or None for console-only
        level: Logging level (default: INFO)

    Returns:
        Configured logger for the calling module
    """
    handlers = []

    # Always add console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setFormatter(
        logging.Formatter("[%(name)s] %(levelname)s: %(message)s")
    )
    handlers.append(console_handler)

    # Add file handler if log file specified
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
        )
        handlers.append(file_handler)

    # Configure root logger
    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True,  # Override any existing configuration
    )

    # Return logger for the calling module
    # We get the caller's module name from the stack
    import inspect

    frame = inspect.currentframe()
    if frame and frame.f_back:
        caller_module = frame.f_back.f_globals.get("__name__", __name__)
    else:
        caller_module = __name__

    return logging.getLogger(caller_module)
