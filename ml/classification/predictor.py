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
  - CLAHE-based lighting normalization for real-world robustness
  - Image validation to reject corrupt or empty inputs
  - THIS IS THE ONLY threshold check — no duplicate checks downstream

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

import cv2
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
    against ``inference.confidence_threshold`` (from config.yaml, default 0.55).
    If below threshold → returns "unknown" with grade=null instead of a
    potentially wrong class label. This prevents confident-looking wrong answers.

    IMPORTANT: This predictor is the SINGLE source of truth for the confidence
    threshold. No downstream code (controller, pipeline, inference_service)
    should re-check or override this decision.

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
        self.device           = get_device(cfg.get("inference", {}).get("device", "cpu"))
        self.conf_threshold   = cfg.get("inference", {}).get("confidence_threshold", 0.55)
        self.top_k            = top_k

        # ── Class names ────────────────────────────────────────────
        if class_names is not None:
            self.class_names = class_names
        else:
            try:
                from ml.utils.helpers import load_class_mapping
                self.class_names = load_class_mapping()
            except Exception:
                self.class_names = get_class_names(cfg)

        num_classes = len(self.class_names)
        logger.info(f"Predictor: {num_classes} classes | threshold={self.conf_threshold}")
        logger.info(f"  Classes: {self.class_names}")

        # ── Model ─────────────────────────────────────────────────
        cfg_copy = dict(cfg)
        cfg_copy["model"] = dict(cfg.get("model", {}))
        cfg_copy["model"]["num_classes"] = num_classes

        self.model: nn.Module = build_model(cfg_copy, num_classes=num_classes)

        # ── Load checkpoint ───────────────────────────────────────
        model_cfg = cfg.get("model", {})
        ckpt_path = Path(
            checkpoint_path
            or model_cfg.get("save_path", str(Path(model_cfg.get("save_dir", "ml/models/saved")) / model_cfg.get("checkpoint_name", "best_model.pth")))
        )
        if ckpt_path.exists():
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
            if "model_state_dict" in ckpt:
                self.model.load_state_dict(ckpt["model_state_dict"])
                epoch   = ckpt.get("epoch", "?")
                metrics = ckpt.get("metrics", {})
                logger.info(f"[BOOT] Active model loaded: {ckpt_path} (epoch {epoch})")
            else:
                self.model.load_state_dict(ckpt)
                logger.info(f"[BOOT] Active model loaded (raw state_dict): {ckpt_path}")
        else:
            logger.warning(
                f"No checkpoint found at {ckpt_path}. "
                "Using randomly initialised weights — predictions will be meaningless. "
                "Train the model first: python scripts/train.py"
            )

        self.model.eval()

        # ── Preprocessing transform (identical to val/test in training) ───
        pp = cfg.get("preprocessing", {})
        self.image_size = pp.get("image_size", 224)
        self.mean       = pp.get("mean", [0.485, 0.456, 0.406])
        self.std        = pp.get("std", [0.229, 0.224, 0.225])
        # Match exactly the training pipeline: Resize → ToTensor → Normalize.
        # No CenterCrop — training (dataset.py) used Resize(224,224) directly.
        self._transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        # ── CLAHE for lighting robustness ─────────────────────────
        # Applied in LAB color space to normalize brightness without
        # distorting hue/saturation.  Handles low-light, shadows,
        # conveyor reflections, and high-brightness conditions.
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

        # ── Temporal smoothing for realtime stability ─────────────
        # When processing live camera frames, confidence and label can
        # flicker frame-to-frame due to minor lighting/angle changes.
        # An exponential moving average (EMA) on softmax probabilities
        # smooths these fluctuations.  Alpha=0.6 means 60% current
        # frame + 40% history → stable within ~3 frames.
        self._ema_alpha = 0.6   # weight for current frame (higher = more responsive)
        self._ema_probs: torch.Tensor | None = None  # running avg probabilities
        self._smoothing_enabled = False  # enabled only for video/realtime

    # ─────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────

    @staticmethod
    def _validate_image(image: np.ndarray | Image.Image) -> np.ndarray:
        """
        Validate and convert the input image to a well-formed numpy RGB array.

        Checks:
          - Image is not None
          - Image is not empty (zero-size)
          - Image has valid dimensions (H, W, 3)
          - Image is in uint8 format

        Returns:
            np.ndarray (H, W, 3) in RGB uint8 format.

        Raises:
            ValueError: If the image is invalid or corrupt.
        """
        if image is None:
            raise ValueError("Image is None — cannot classify empty input.")

        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))

        if not isinstance(image, np.ndarray):
            raise ValueError(f"Unexpected image type: {type(image)}")

        if image.size == 0:
            raise ValueError("Image is empty (zero pixels).")

        if image.ndim == 2:
            # Greyscale — expand to 3 channels
            image = np.stack([image] * 3, axis=-1)

        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(
                f"Invalid image shape: {image.shape}. "
                "Expected (H, W, 3) RGB array."
            )

        if image.shape[0] < 10 or image.shape[1] < 10:
            raise ValueError(
                f"Image too small: {image.shape[:2]}. "
                "Minimum 10×10 pixels required."
            )

        return image.astype(np.uint8)

    def _normalize_lighting(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE-based lighting normalization in LAB color space.

        This handles:
          - Low light / dark conveyor environments
          - High brightness / overexposed areas
          - Shadows and uneven illumination
          - Conveyor belt reflections

        The L (lightness) channel is equalized; A and B (color) channels
        are preserved to maintain accurate fruit colors for classification.

        Args:
            image: np.ndarray (H, W, 3) in RGB uint8 format.

        Returns:
            Lighting-normalized RGB uint8 array, same shape.
        """
        # Convert RGB → LAB (L = lightness, A/B = color)
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # CLAHE on L channel only — normalizes brightness without
        # distorting colors (critical for fruit classification)
        l_normalized = self._clahe.apply(l_channel)

        # Merge back and convert LAB → RGB
        lab_normalized = cv2.merge([l_normalized, a_channel, b_channel])
        result = cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2RGB)
        return result

    def _preprocess(self, image: np.ndarray | Image.Image) -> torch.Tensor:
        """
        Convert image (numpy array or PIL) to a model-ready (1, C, H, W) tensor.

        Pipeline (all in RGB — NO BGR conversion anywhere):
          1. Validate image dimensions and type
          2. Normalize lighting via CLAHE (LAB color space)
          3. Convert to PIL Image (RGB)
          4. Apply standard torchvision transforms (Resize → CenterCrop → ToTensor → Normalize)

        Accepts:
          - np.ndarray (H, W, 3) in RGB order (from pipeline._load_image)
          - PIL Image in any mode

        IMPORTANT: Images arriving here are ALREADY RGB (converted by
        inference_pipeline._load_image). Do NOT flip channels.
        """
        # Step 1: Validate and ensure numpy RGB uint8
        arr = self._validate_image(image)

        # Step 2: Lighting normalization (CLAHE in LAB space — preserves colors)
        arr = self._normalize_lighting(arr)

        # Step 3: Convert to PIL (already RGB — no channel flip!)
        pil_img = Image.fromarray(arr, mode="RGB")

        # Step 4: Standard torchvision transforms
        tensor = self._transform(pil_img).unsqueeze(0).to(self.device)  # (1, C, H, W)

        return tensor

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

        This is the SINGLE place where the confidence threshold is applied.
        No downstream code should override this decision.

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

    def enable_smoothing(self) -> None:
        """Enable temporal EMA smoothing for realtime/video mode."""
        self._smoothing_enabled = True

    def disable_smoothing(self) -> None:
        """Disable temporal smoothing (for single-image API calls)."""
        self._smoothing_enabled = False
        self._ema_probs = None

    def reset_smoothing(self) -> None:
        """Reset EMA history (call when a new fruit enters the frame)."""
        self._ema_probs = None

    @torch.no_grad()
    def predict(self, image: np.ndarray | Image.Image) -> dict:
        """
        Classify a single cropped (or full) fruit image.

        When temporal smoothing is enabled (realtime mode), softmax
        probabilities are averaged with a running EMA to prevent
        frame-to-frame flicker.

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

        # ── Temporal smoothing (only for realtime/video) ─────────
        if self._smoothing_enabled:
            if self._ema_probs is None:
                self._ema_probs = probs.clone()
            else:
                # EMA: smoothed = alpha * current + (1-alpha) * history
                self._ema_probs = (
                    self._ema_alpha * probs +
                    (1.0 - self._ema_alpha) * self._ema_probs
                )
            probs = self._ema_probs

        confidence = float(probs.max().item())
        pred_idx   = int(probs.argmax().item())
        label      = self.class_names[pred_idx]
        top_k_list = self._build_top_k(probs)

        result = self._make_result(confidence, label, top_k_list)

        # ── Diagnostics (visible only at DEBUG log level) ────────
        logger.debug(
            f"predict | label={result['label']} conf={confidence:.4f} "
            f"threshold={self.conf_threshold} unknown={result['unknown']} "
            f"smooth={self._smoothing_enabled}"
        )

        return result

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

        # ── Temporal smoothing for YOLO crops (N=1 only) ─────────
        # On the conveyor belt, YOLO returns exactly 1 cropped fruit.
        if self._smoothing_enabled and probs.size(0) == 1:
            if self._ema_probs is None:
                self._ema_probs = probs[0].clone()
            else:
                self._ema_probs = (
                    self._ema_alpha * probs[0] +
                    (1.0 - self._ema_alpha) * self._ema_probs
                )
            probs[0] = self._ema_probs
        elif self._smoothing_enabled and probs.size(0) > 1:
            # If multiple fruits detected, skip smoothing to avoid mixing classes
            logger.debug("[SMOOTHING ACTIVE] Skipped — multiple objects detected")

        results: list[dict] = []
        for i, row_probs in enumerate(probs):
            confidence = float(row_probs.max().item())
            pred_idx   = int(row_probs.argmax().item())
            label      = self.class_names[pred_idx]
            top_k_list = self._build_top_k(row_probs)

            result = self._make_result(confidence, label, top_k_list)

            # ── Diagnostics (visible only at DEBUG log level) ────
            logger.debug(
                f"batch[{i}] | label={result['label']} conf={confidence:.4f} "
                f"threshold={self.conf_threshold} unknown={result['unknown']} "
                f"top_k={top_k_list}"
            )

            results.append(result)

        return results
