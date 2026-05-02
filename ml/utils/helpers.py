from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch


def load_yaml(path):
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_class_mapping(class_names, save_path="ml/models/saved/class_mapping.json"):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=4)


def load_class_mapping(load_path="ml/models/saved/class_mapping.json"):
    with open(load_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)
    return class_names


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target


def setup_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(preferred: str | None = None) -> torch.device:
    if preferred is None:
        preferred = "cpu"
    pref = preferred.lower()
    if pref.startswith("cuda") and torch.cuda.is_available():
        return torch.device(preferred)
    return torch.device("cpu")


def resolve_cfg_paths(cfg: dict, base_dir: str | Path) -> dict:
    """
    Resolve common relative config paths to absolute paths.
    """
    root = Path(base_dir).resolve()
    out = dict(cfg)

    for section in ("data", "model", "detection"):
        if section in out and isinstance(out[section], dict):
            out[section] = dict(out[section])

    path_keys = {
        "data": ("raw_dir", "train_dir", "val_dir", "test_dir"),
        "model": ("save_dir",),
        "detection": ("weights",),
    }

    for section, keys in path_keys.items():
        section_data = out.get(section)
        if not isinstance(section_data, dict):
            continue
        for key in keys:
            value = section_data.get(key)
            if not value or not isinstance(value, str):
                continue
            p = Path(value)
            if not p.is_absolute():
                section_data[key] = str((root / p).resolve())

    return out
