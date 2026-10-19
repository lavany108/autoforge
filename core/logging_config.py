"""
AutoForge Structured Logging Configuration
Provides unified logging formatting and handlers across all services and core modules.
"""

import logging
import os
import sys
from typing import Optional


def setup_logging(
    name: Optional[str] = "autoforge",
    level: Optional[str] = None,
) -> logging.Logger:
    """
    Configures and returns a logger instance with consistent formatting.
    Reads LOG_LEVEL from environment if not explicitly provided.
    """
    log_level_str = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    log_level = getattr(logging, log_level_str, logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid adding duplicate handlers if already configured
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s:%(lineno)d] - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Do not propagate to root logger to prevent duplicate lines in certain runners
    logger.propagate = False
    return logger


# Default logger instance
logger = setup_logging()
