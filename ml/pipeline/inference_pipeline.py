"""
ml/pipeline/inference_pipeline.py
-----------------------------------
End-to-end fruit grading pipeline:

    image / frame
        ↓
    YOLOv8 → detect fruit regions (bounding boxes)
        ↓  ← if nothing detected → fallback: use full image
    Crop detected regions
        ↓
    EfficientNet-B0 → classify fruit + grade
        ↓
    Confidence threshold filter (configurable via config.yaml)
        ↓
    Structured JSON-ready output  +  optional visualization

Output schema (every result always has these keys):

    {
      "fruit":      "apple"  | "unknown",
      "grade":      "A"      | null,
      "label":      "apple_A"| "unknown",
      "confidence": float,            # EfficientNet softmax max-prob
      "unknown":    bool,             # True when confidence < threshold
      "bbox":       [x1,y1,x2,y2]   | null,   # null on fallback
      "detection":  "yolo"           | "fallback",
      "yolo_conf":  float            | null,   # null on fallback
      "top_k":      [...],                     # top-3 class probs
    }

Usage:
    pipeline = InferencePipeline(cfg)

    # Single image (path / PIL / numpy)
    results = pipeline.run("path/to/fruit.jpg")

    # Video frame (numpy BGR from OpenCV)
    results = pipeline.run_on_video_frame(frame)

    # With visualization saved to disk
    results = pipeline.run("img.jpg", visualize=True, vis_save_path="out.jpg")
"""

from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ml.classification.predictor import FruitGradePredictor
from ml.detection.yolo_detector import Detection, YOLODetector
from ml.utils.helpers import ensure_dir, setup_logger

logger = setup_logger(__name__)


# ── Visualization constants ──────────────────────────────────────────────────
_VIS_BOX_COLOR_YOLO     = (50, 205, 50)    # lime-green  for YOLO boxes
_VIS_BOX_COLOR_FALLBACK = (255, 165, 0)    # orange      for full-image fallback
_VIS_TEXT_COLOR         = (255, 255, 255)  # white text
_VIS_BOX_WIDTH          = 3               # px


class InferencePipeline:
    """
    Orchestrates YOLO detection → crop → EfficientNet classification.

    YOLO-failure safety
    -------------------
    If YOLO finds no objects (and the config's ``fallback_full_image`` is True),
    the entire image is passed to the classifier as a single "crop".
    The result for that detection is marked  ``detection="fallback"``
    and ``bbox=null``  so the caller always gets a usable response.

    Args:
        cfg:             Config dict from config.yaml.
        checkpoint_path: Optional EfficientNet checkpoint path override.
        class_names:     Optional class list; derived from train dir if None.
    """

    def __init__(
        self,
        cfg: dict,
        checkpoint_path: str | Path | None = None,
        class_names: list[str] | None = None,
    ) -> None:
        self.cfg         = cfg
        self._detector   = YOLODetector(cfg)
        self._classifier = FruitGradePredictor(cfg, checkpoint_path, class_names)

    # ─────────────────────────────────────────────
    # Image loading
    # ─────────────────────────────────────────────

    @staticmethod
    def _load_image(source: str | Path | np.ndarray | Image.Image) -> np.ndarray:
        """
        Normalise any image source to a numpy RGB array (H, W, 3).

        Supports:
          - File path (str / Path)
          - PIL Image
          - numpy array (greyscale or RGB/BGR)
        """
        if isinstance(source, (str, Path)):
            img = Image.open(source).convert("RGB")
            return np.array(img)
        if isinstance(source, Image.Image):
            return np.array(source.convert("RGB"))
        if isinstance(source, np.ndarray):
            if source.ndim == 2:
                source = np.stack([source] * 3, axis=-1)  # greyscale → RGB
            return source
        raise TypeError(f"Unsupported image source type: {type(source)}")

    # ─────────────────────────────────────────────
    # Result assembly
    # ─────────────────────────────────────────────

    # ─────────────────────────────────────────────
    # Output enrichment helpers
    # ─────────────────────────────────────────────

    @staticmethod
    def _confidence_level(confidence: float) -> str:
        """
        Bucket a softmax confidence score into a human-readable tier.

        Thresholds:
            >= 0.85  → "high"
            >= 0.60  → "medium"
            < 0.60   → "low"
        """
        if confidence >= 0.85:
            return "high"
        if confidence >= 0.60:
            return "medium"
        return "low"

    @staticmethod
    def _explanation(unknown: bool, detection: str, confidence_level: str) -> str:
        """
        Generate a one-sentence human-readable explanation of the result.
        Wording is consistent with the ``confidence_level`` field so that
        demo / viva output is self-coherent.

        Priority order:
            1. unknown class  → confidence was too low to make a prediction
            2. fallback mode  → YOLO found nothing, full-image used
            3. normal YOLO    → detection + classification succeeded

        Args:
            unknown:          True when classification was rejected.
            detection:        "yolo" or "fallback".
            confidence_level: "high" | "medium" | "low" (from _confidence_level).
        """
        # Map tier to the phrase used in the explanation sentence
        level_phrase = {
            "high":   "high confidence",
            "medium": "moderate confidence",
            "low":    "low confidence",
        }.get(confidence_level, "low confidence")

        if unknown:
            return (
                f"Prediction rejected — {level_phrase} score "
                "is below the acceptance threshold."
            )
        if detection == "fallback":
            return (
                f"YOLO detected no fruit; classification performed on the full image "
                f"with {level_phrase}."
            )
        return f"Fruit detected by YOLO and classified with {level_phrase}."

    @staticmethod
    def _assemble_result(
        cls_result: dict,
        det: Detection,
        is_fallback: bool,
    ) -> dict:
        """
        Merge classifier output + detection metadata into the canonical schema.

        Args:
            cls_result:  Dict from FruitGradePredictor.predict / predict_batch.
            det:         Detection object (bbox, confidence, class_name).
            is_fallback: True when det covers the full image (no YOLO box).

        Returns:
            Canonical result dict with all required keys.
        """
        if is_fallback:
            bbox      = None   # explicit null — no real bounding box
            yolo_conf = None
            detection = "fallback"
        else:
            x1, y1, x2, y2 = det.bbox
            bbox      = [int(x1), int(y1), int(x2), int(y2)]
            yolo_conf = round(det.confidence, 4)
            detection = "yolo"

        conf     = cls_result["confidence"]
        unknown  = cls_result["unknown"]

        return {
            # ── Core prediction ───────────────────────────────────
            "fruit":            cls_result["fruit"],
            "grade":            cls_result["grade"],   # None when unknown
            "label":            cls_result["label"],
            # ── Confidence ────────────────────────────────────────
            "confidence":       conf,
            "confidence_level": InferencePipeline._confidence_level(conf),
            "unknown":          unknown,
            # ── Detection metadata ────────────────────────────────
            "bbox":             bbox,                  # null on fallback
            "detection":        detection,             # "yolo" | "fallback"
            "yolo_conf":        yolo_conf,             # null on fallback
            # ── System / explainability ───────────────────────────
            "mode":             "hybrid_detection_classification",
            "explanation":      InferencePipeline._explanation(
                unknown, detection,
                InferencePipeline._confidence_level(conf),
            ),
            "top_k":            cls_result.get("top_k", []),
        }

    # ─────────────────────────────────────────────
    # Visualization
    # ─────────────────────────────────────────────

    @staticmethod
    def draw_results(
        source: str | Path | np.ndarray | Image.Image,
        results: list[dict],
        save_path: str | Path | None = None,
    ) -> Image.Image:
        """
        Draw bounding boxes and labels on the image for demo / viva use.

        Behavior:
          - YOLO box   → lime-green rectangle + "fruit_grade (conf)"
          - Fallback   → orange border around the full image
          - Unknown    → label text shows "unknown (conf)"
          - If save_path provided → image is saved to that path

        Args:
            source:    Original image (any format accepted by _load_image).
            results:   List of result dicts from run().
            save_path: Optional path to save the annotated image.

        Returns:
            PIL Image with annotations drawn.
        """
        # Load as PIL
        if isinstance(source, np.ndarray):
            pil_img = Image.fromarray(source).convert("RGB")
        elif isinstance(source, (str, Path)):
            pil_img = Image.open(source).convert("RGB")
        else:
            pil_img = source.convert("RGB")

        draw = ImageDraw.Draw(pil_img)
        W, H = pil_img.size

        # Try to use a truetype font; fall back to default PIL bitmap font
        try:
            font = ImageFont.truetype("arial.ttf", size=18)
        except (IOError, OSError):
            font = ImageFont.load_default()

        for r in results:
            is_fallback = r["detection"] == "fallback"
            color       = _VIS_BOX_COLOR_FALLBACK if is_fallback else _VIS_BOX_COLOR_YOLO
            label_text  = f"{r['label']} ({r['confidence']:.2f})"

            if is_fallback or r["bbox"] is None:
                # ── Full-image orange border ───────────────────────
                margin = _VIS_BOX_WIDTH
                draw.rectangle(
                    [margin, margin, W - margin, H - margin],
                    outline=color,
                    width=_VIS_BOX_WIDTH,
                )
                # Label in top-left corner
                draw.rectangle(
                    [margin, margin, margin + len(label_text) * 11, margin + 24],
                    fill=(0, 0, 0, 180),
                )
                draw.text((margin + 4, margin + 4), label_text, fill=color, font=font)

            else:
                x1, y1, x2, y2 = r["bbox"]

                # ── Bounding box ───────────────────────────────────
                draw.rectangle([x1, y1, x2, y2], outline=color, width=_VIS_BOX_WIDTH)

                # ── Label background pill ──────────────────────────
                text_w = len(label_text) * 10
                text_h = 22
                lx1, ly1 = x1, max(0, y1 - text_h - 4)
                lx2, ly2 = lx1 + text_w + 8, ly1 + text_h + 4
                draw.rectangle([lx1, ly1, lx2, ly2], fill=color)
                draw.text((lx1 + 4, ly1 + 2), label_text, fill=_VIS_TEXT_COLOR, font=font)

        if save_path is not None:
            ensure_dir(Path(save_path).parent)
            pil_img.save(save_path)
            logger.info(f"Visualization saved → {save_path}")

        return pil_img

    # ─────────────────────────────────────────────
    # Core pipeline
    # ─────────────────────────────────────────────

    def run(
        self,
        source: str | Path | np.ndarray | Image.Image,
        bbox_padding: int = 10,
        visualize: bool = False,
        vis_save_path: str | Path | None = None,
    ) -> list[dict]:
        """
        Run the full detection → classification pipeline on one image.

        YOLO-failure safety
        -------------------
        If YOLO returns no detections and ``fallback_full_image`` is True in
        config, the whole image is classified. The result carries
        ``detection="fallback"`` and ``bbox=null``.

        Args:
            source:        File path, PIL Image, or numpy array (H, W, 3).
            bbox_padding:  Extra pixels around each YOLO bounding box.
            visualize:     If True, annotate and optionally save the image.
            vis_save_path: Path to save visualization (only when visualize=True).

        Returns:
            List of result dicts (one per detected / fallback region):
            [
              {
                "fruit":      "apple",
                "grade":      "A",
                "label":      "apple_A",
                "confidence": 0.91,
                "unknown":    False,
                "bbox":       [x1, y1, x2, y2],   # or null on fallback
                "detection":  "yolo",              # or "fallback"
                "yolo_conf":  0.85,                # or null on fallback
                "top_k":      [...],
              },
              ...
            ]
        """
        # ── Timing starts here (before any heavy computation) ─────
        _t_start = time.perf_counter()

        image = self._load_image(source)
        logger.debug(f"Pipeline input: shape={image.shape}")

        # ── Step 1: Detection ──────────────────────────────────────
        try:
            detections: list[Detection] = self._detector.detect(image)
        except Exception as exc:
            # YOLO failed entirely (e.g. weights not found) — treat as fallback
            logger.warning(f"YOLO detection raised an exception: {exc}")
            detections = []

        logger.debug(f"  Raw detections: {len(detections)}")

        # ── Determine fallback mode ────────────────────────────────
        # A detection with class_id == -1 is the synthetic full-image
        # fallback inserted by YOLODetector when nothing was found.
        real_detections  = [d for d in detections if d.class_id != -1]
        fallback_dets    = [d for d in detections if d.class_id == -1]

        if real_detections:
            active_dets  = real_detections
            is_fallback  = False
            logger.debug(f"  Using {len(active_dets)} YOLO detection(s)")
        elif fallback_dets:
            # YOLO found nothing → full-image fallback injected by detector
            active_dets  = fallback_dets
            is_fallback  = True
            logger.info("YOLO found no fruit — using full-image fallback")
        else:
            # No detections AND fallback disabled in config
            logger.info("No detections and fallback disabled — returning []")
            return []

        # ── Step 2: Crop ───────────────────────────────────────────
        # For fallback detections (full image), padding is irrelevant
        pad    = 0 if is_fallback else bbox_padding
        crops  = self._detector.crop_detections(image, active_dets, padding=pad)

        # ── Step 3: Batch classify ─────────────────────────────────
        crop_arrays     = [crop for crop, _ in crops]
        classifications = self._classifier.predict_batch(crop_arrays)

        # ── Step 4: Assemble canonical output ─────────────────────
        # Snapshot timestamp once for the whole batch so every result in
        # the same call shares the same wall-clock moment.
        _iso_timestamp      = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        _processing_time_ms = round((time.perf_counter() - _t_start) * 1000, 1)

        results: list[dict] = []
        for (crop_arr, det), cls_result in zip(crops, classifications):
            result = self._assemble_result(cls_result, det, is_fallback)
            # ── Inject output metadata ────────────────────────────
            result["timestamp"]          = _iso_timestamp
            result["processing_time_ms"] = _processing_time_ms
            results.append(result)

        logger.info(
            f"Pipeline done — {len(results)} result(s) "
            f"[detection={'fallback' if is_fallback else 'yolo'}]: "
            + ", ".join(
                f"{r['label']}({r['confidence']:.2f})" for r in results
            )
        )

        # ── Step 5: Optional visualization ────────────────────────
        if visualize:
            self.draw_results(source, results, save_path=vis_save_path)

        return results

    # ─────────────────────────────────────────────
    # Convenience wrappers
    # ─────────────────────────────────────────────

    def run_on_video_frame(
        self,
        frame: np.ndarray,
        bbox_padding: int = 10,
        visualize: bool = False,
        vis_save_path: str | Path | None = None,
    ) -> list[dict]:
        """
        Wrapper for real-time video / conveyor-belt frames.

        Identical to run(), but named explicitly for streaming contexts.
        Expects numpy array in BGR (OpenCV) or RGB order.

        Args:
            frame:         Video frame, shape (H, W, 3).
            bbox_padding:  Pixels to expand bounding boxes.
            visualize:     Annotate and save frame if True.
            vis_save_path: Save path for annotated frame.

        Returns:
            Same list of result dicts as run().
        """
        return self.run(
            source=frame,
            bbox_padding=bbox_padding,
            visualize=visualize,
            vis_save_path=vis_save_path,
        )

    def run_on_image_path(
        self,
        image_path: str | Path,
        visualize: bool = False,
        vis_save_path: str | Path | None = None,
    ) -> list[dict]:
        """
        Wrapper for single-image file paths.

        Args:
            image_path:    Path to image file.
            visualize:     Annotate and save image if True.
            vis_save_path: Save path for annotated image.

        Returns:
            Same list of result dicts as run().
        """
        return self.run(
            source=Path(image_path),
            visualize=visualize,
            vis_save_path=vis_save_path,
        )
