"""
ml/models/model_builder.py
--------------------------
Builds and returns the EfficientNet-B0 classification model with a
custom head matching the number of fruit-grade classes.
"""

from pathlib import Path

import torch
import torch.nn as nn
import torchvision.models as tv_models

from ml.utils.helpers import get_device, setup_logger

logger = setup_logger(__name__)


def build_model(cfg: dict, num_classes: int | None = None) -> nn.Module:
    """
    Load EfficientNet-B0 (optionally pretrained) and replace the
    classifier head with a Linear layer matching ``num_classes``.

    Args:
        cfg:         Config dict.
        num_classes: Override config value when provided (useful for
                     loading a checkpoint whose class count differs
                     from the default config).

    Returns:
        nn.Module moved to the configured device.
    """
    n_classes = num_classes if num_classes is not None else cfg["model"]["num_classes"]
    pretrained = cfg["model"]["pretrained"]

    logger.info(
        f"Building EfficientNet-B0 | pretrained={pretrained} | num_classes={n_classes}"
    )

    if pretrained:
        weights = tv_models.EfficientNet_B0_Weights.IMAGENET1K_V1
        model = tv_models.efficientnet_b0(weights=weights)
    else:
        model = tv_models.efficientnet_b0(weights=None)

    # Replace the classifier head
    # EfficientNet-B0 classifier: Sequential(Dropout(0.2), Linear(1280, 1000))
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features, n_classes),
    )

    device = get_device(cfg["inference"]["device"])
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"  Total params:     {total_params:,}")
    logger.info(f"  Trainable params: {trainable:,}")
    logger.info(f"  Device:           {device}")

    return model


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    cfg: dict,
    filename: str | None = None,
) -> Path:
    """
    Save model + optimizer state dict to ml/models/saved/.

    Args:
        model:     The model to save.
        optimizer: Current optimizer (state saved for resuming).
        epoch:     Current epoch number.
        metrics:   Dict of metric values to store alongside checkpoint.
        cfg:       Config dict.
        filename:  Override file name (default: cfg checkpoint_name).

    Returns:
        Path where checkpoint was saved.
    """
    save_dir = Path(cfg["model"]["save_dir"])
    save_dir.mkdir(parents=True, exist_ok=True)

    fname = filename or cfg["model"]["checkpoint_name"]
    save_path = save_dir / fname

    torch.save(
        {
            "epoch":      epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics":    metrics,
            "num_classes": cfg["model"]["num_classes"],
        },
        save_path,
    )
    logger.info(f"Checkpoint saved → {save_path}")
    return save_path


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    optimizer: torch.optim.Optimizer | None = None,
    device: torch.device | None = None,
) -> tuple[nn.Module, dict]:
    """
    Load a saved checkpoint into a model (and optionally optimizer).

    Args:
        model:           Model with matching architecture.
        checkpoint_path: Path to .pth file.
        optimizer:       If provided, optimizer state is also restored.
        device:          Target device; defaults to CPU if None.

    Returns:
        (model, checkpoint_dict)
    """
    map_loc = device or torch.device("cpu")
    ckpt = torch.load(checkpoint_path, map_location=map_loc, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    epoch = ckpt.get("epoch", "?")
    metrics = ckpt.get("metrics", {})
    logger.info(f"Checkpoint loaded from {checkpoint_path} (epoch {epoch}) | {metrics}")

    return model, ckpt
