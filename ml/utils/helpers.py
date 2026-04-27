"""
ml/utils/helpers.py
-------------------
Shared utilities: device detection, seeding, logger setup.
"""

import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch


# ─────────────────────────────────────────────
# Device
# ─────────────────────────────────────────────

def get_device(device_str: str = "auto") -> torch.device:
    """
    Return the appropriate torch.device.

    Args:
        device_str: "auto" | "cuda" | "cpu"

    Returns:
        torch.device
    """
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ─────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────

def set_seed(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Make cuDNN deterministic (slightly slower, but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ─────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────

def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """
    Create (or retrieve) a named logger with a timestamped console handler.

    Args:
        name:  Logger name (usually __name__ of the calling module).
        level: Logging level (default: INFO).

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        # Avoid adding duplicate handlers on repeated calls
        return logger

    logger.setLevel(level)

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


# ─────────────────────────────────────────────
# Path helpers
# ─────────────────────────────────────────────

def ensure_dir(path: str | Path) -> Path:
    """Create directory (and parents) if it does not exist. Returns Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_yaml(path: str | Path) -> dict:
    """Load a YAML file and return as a dict."""
    import yaml  # lazy import so module is importable without PyYAML in non-ML contexts
    with open(path, "r") as f:
        return yaml.safe_load(f)
