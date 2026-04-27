"""
ml/classification/predictor.py
--------------------------------
Loads a trained EfficientNet-B0 checkpoint and classifies
cropped fruit images into fruit+grade labels (e.g. "apple_A").

Key improvements:
  - Unknown-class rejection via configurable softmax confidence threshold
  - Structured, consistent output schema across predict() and predict_batch()
  - Top-K probabilities returned for explainability / demo use
  - Clear separation between low-confidence "unknown" and normal predictions

Usage:
    predictor = FruitGradePredictor(cfg)

    # Single image
    result = predictor.predict(image_array)
    # → {
    #     "label":      "apple_A",
    #     "fruit":      "apple",
    #     "grade":      "A",
    #     "confidence": 0.93,
    #     "unknown":    False,
    #     "top_k":      [{"label": "apple_A", "prob": 0.93}, ...]
    #   }

    # If confidence < threshold:
    # → {
    #     "label":      "unknown",
    #     "fruit":      "unknown",
    #     "grade":      null,
    #     "confidence": 0.42,
    #     "unknown":    True,
    #     "top_k":      [...]
    #   }
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from ml.models.model_builder import build_model
from ml.training.data_loader import get_class_names
from ml.utils.helpers import get_device, setup_logger

logger = setup_logger(__name__)

# ── Sentinel values used in output when class is unknown ────────────────────
_UNKNOWN_LABEL = "unknown"
_UNKNOWN_GRADE = None          # explicit null in JSON / None in Python


class FruitGradePredictor:
    """
    EfficientNet-B0 classifier for fruit + grade prediction.

    Unknown-class handling
    ----------------------
    After computing softmax probabilities, the maximum probability is compared
    against ``inference.confidence_threshold`` (from config.yaml, default 0.70).
    If below threshold → returns "unknown" with grade=null instead of a
    potentially wrong class label. This prevents confident-looking wrong answers.

    Args:
        cfg:             Config dict from config.yaml.
        checkpoint_path: Path to .pth checkpoint. If None, uses
                         cfg['model']['save_dir'] / cfg['model']['checkpoint_name'].
        class_names:     Ordered class list. If None, derives from train dir.
        top_k:           How many top predictions to include for explainability.
    """

    def __init__(
        self,
        cfg: dict,
        checkpoint_path: str | Path | None = None,
        class_names: list[str] | None = None,
        top_k: int = 3,
    ) -> None:
        self.cfg = cfg
        self.device           = get_device(cfg["inference"]["device"])
        self.conf_threshold   = cfg["inference"]["confidence_threshold"]
        self.top_k            = top_k

        # ── Class names ────────────────────────────────────────────
        if class_names is not None:
            self.class_names = class_names
        else:
            self.class_names = get_class_names(cfg)

        num_classes = len(self.class_names)
        logger.info(f"Predictor: {num_classes} classes | threshold={self.conf_threshold}")
        logger.debug(f"  Classes: {self.class_names}")

        # ── Model ─────────────────────────────────────────────────
        cfg_copy = dict(cfg)
        cfg_copy["model"] = dict(cfg["model"])
        cfg_copy["model"]["num_classes"] = num_classes

        self.model: nn.Module = build_model(cfg_copy, num_classes=num_classes)

        # ── Load checkpoint ───────────────────────────────────────
        ckpt_path = Path(
            checkpoint_path
            or (Path(cfg["model"]["save_dir"]) / cfg["model"]["checkpoint_name"])
        )
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(ckpt["model_state_dict"])
            epoch   = ckpt.get("epoch", "?")
            metrics = ckpt.get("metrics", {})
            logger.info(f"Checkpoint loaded: {ckpt_path} (epoch {epoch}, {metrics})")
        else:
            logger.warning(
                f"No checkpoint found at {ckpt_path}. "
                "Using randomly initialised weights — predictions will be meaningless. "
                "Train the model first: python scripts/train.py"
            )

        self.model.eval()

        # ── Preprocessing transform (identical to val/test in training) ───
        pp = cfg["preprocessing"]
        self._transform = transforms.Compose([
            transforms.Resize((pp["image_size"], pp["image_size"])),
            transforms.CenterCrop(pp["center_crop"]),
            transforms.ToTensor(),
            transforms.Normalize(mean=pp["mean"], std=pp["std"]),
        ])

    # ─────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────

    def _preprocess(self, image: np.ndarray | Image.Image) -> torch.Tensor:
        """
        Convert image (numpy array or PIL) to a model-ready (1, C, H, W) tensor.

        Accepts:
          - np.ndarray (H, W, 3) in RGB or BGR (OpenCV) order
          - PIL Image in any mode
        """
        if isinstance(image, np.ndarray):
            # OpenCV arrays are BGR — convert channel order to RGB before PIL
            arr = image[..., ::-1].astype(np.uint8) if image.ndim == 3 else image
            pil_img = Image.fromarray(arr).convert("RGB")
        else:
            pil_img = image.convert("RGB")

        return self._transform(pil_img).unsqueeze(0).to(self.device)  # (1, C, H, W)

    def _parse_label(self, label: str) -> tuple[str, str]:
        """
        Split composite label "apple_A" → ("apple", "A").
        The grade is always the last underscore-separated segment.
        Handles multi-word fruits: "Honeydew_B" → ("Honeydew", "B").
        """
        parts = label.rsplit("_", 1)
        if len(parts) == 2:
            return parts[0], parts[1]
        return label, "?"

    def _build_top_k(self, probs: torch.Tensor) -> list[dict]:
        """
        Return the top-K class probabilities as a list of dicts,
        ordered from highest to lowest.

        Args:
            probs: 1-D tensor of class probabilities (already softmax-ed).

        Returns:
            [{"label": "apple_A", "prob": 0.93}, ...]
        """
        k = min(self.top_k, len(self.class_names))
        top_probs, top_idxs = probs.topk(k)
        return [
            {
                "label": self.class_names[int(i)],
                "prob":  round(float(p), 4),
            }
            for p, i in zip(top_probs.tolist(), top_idxs.tolist())
        ]

    def _make_result(
        self,
        confidence: float,
        label: str,
        top_k_list: list[dict],
    ) -> dict:
        """
        Build the canonical prediction result dict.

        If confidence < threshold → "unknown" with grade=null.
        Otherwise → fruit + grade from label.

        Returns:
            {
              "label":      str,
              "fruit":      str,
              "grade":      str | None,
              "confidence": float,
              "unknown":    bool,
              "top_k":      list[dict],
            }
        """
        if confidence < self.conf_threshold:
            # ── Below threshold: reject as unknown ─────────────────
            logger.debug(
                f"  Low confidence ({confidence:.4f} < {self.conf_threshold}) "
                f"→ returning unknown"
            )
            return {
                "label":      _UNKNOWN_LABEL,
                "fruit":      _UNKNOWN_LABEL,
                "grade":      _UNKNOWN_GRADE,   # explicit None / null
                "confidence": round(confidence, 4),
                "unknown":    True,
                "top_k":      top_k_list,
            }

        # ── Above threshold: return predicted class ────────────────
        fruit, grade = self._parse_label(label)
        return {
            "label":      label,
            "fruit":      fruit,
            "grade":      grade,
            "confidence": round(confidence, 4),
            "unknown":    False,
            "top_k":      top_k_list,
        }

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    @torch.no_grad()
    def predict(self, image: np.ndarray | Image.Image) -> dict:
        """
        Classify a single cropped (or full) fruit image.

        Args:
            image: np.ndarray (H, W, 3) or PIL Image.

        Returns:
            {
              "label":      "apple_A" | "unknown",
              "fruit":      "apple"   | "unknown",
              "grade":      "A"       | None,
              "confidence": float,
              "unknown":    bool,
              "top_k":      [{"label": ..., "prob": ...}, ...],
            }
        """
        tensor  = self._preprocess(image)
        outputs = self.model(tensor)
        probs   = torch.softmax(outputs, dim=1).squeeze(0)   # (num_classes,)

        confidence = float(probs.max().item())
        label      = self.class_names[int(probs.argmax().item())]
        top_k_list = self._build_top_k(probs)

        return self._make_result(confidence, label, top_k_list)

    @torch.no_grad()
    def predict_batch(self, images: list[np.ndarray | Image.Image]) -> list[dict]:
        """
        Classify a batch of images in a single forward pass.

        Args:
            images: List of np.ndarray or PIL Image objects.

        Returns:
            List of result dicts (same schema as predict()).
        """
        if not images:
            return []

        # Stack tensors: list of (1, C, H, W) → (N, C, H, W)
        tensors = torch.cat([self._preprocess(img) for img in images], dim=0)
        outputs = self.model(tensors)
        probs   = torch.softmax(outputs, dim=1)   # (N, num_classes)

        results: list[dict] = []
        for row_probs in probs:
            confidence = float(row_probs.max().item())
            label      = self.class_names[int(row_probs.argmax().item())]
            top_k_list = self._build_top_k(row_probs)
            results.append(self._make_result(confidence, label, top_k_list))

        return results
