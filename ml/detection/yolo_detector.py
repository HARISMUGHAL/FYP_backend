"""
ml/detection/yolo_detector.py
------------------------------
Wraps YOLOv8 (Ultralytics) for fruit detection.

Usage:
    detector = YOLODetector(cfg)
    boxes = detector.detect(image)   # image: np.ndarray (H, W, 3 BGR or RGB)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from ml.utils.helpers import setup_logger

logger = setup_logger(__name__)

# COCO class IDs for common fruits (YOLOv8 trained on COCO-80)
# 46=banana, 47=apple, 49=orange
# Note: grapes, guava, apricot, honeydew, tangerine, watermelon are NOT
# in COCO-80. configure filter_coco_ids=[] in config to accept all detections.
COCO_FRUIT_IDS: set[int] = {46, 47, 49}


@dataclass
class Detection:
    """A single YOLO detection result."""
    bbox:       tuple[int, int, int, int]   # (x1, y1, x2, y2) — pixel coords
    confidence: float                        # YOLO detection confidence
    class_id:   int                          # COCO class id
    class_name: str                          # COCO class name (e.g. "apple")


class YOLODetector:
    """
    Lightweight wrapper around YOLOv8 for fruit region detection.

    Args:
        cfg: Config dict from config.yaml.

    Config keys used (under ``detection``):
        weights          : YOLOv8 weights file or ultralytics model name
        conf_threshold   : Minimum detection confidence (YOLO)
        iou_threshold    : NMS IoU threshold
        filter_coco_ids  : List of COCO class IDs to keep; empty = keep all
        fallback_full_image: Return full-image box when nothing detected
    """

    def __init__(self, cfg: dict) -> None:
        det_cfg = cfg["detection"]

        self.weights          = det_cfg["weights"]
        self.conf_threshold   = det_cfg["conf_threshold"]
        self.iou_threshold    = det_cfg["iou_threshold"]
        self.filter_ids: set[int] = set(det_cfg.get("filter_coco_ids") or [])
        self.fallback         = det_cfg.get("fallback_full_image", True)

        self._model = None   # lazy-load on first call

    def _load_model(self) -> None:
        """Lazy-load the YOLO model (downloads weights if needed)."""
        if self._model is not None:
            return
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError(
                "ultralytics is not installed. "
                "Run: pip install ultralytics"
            ) from exc

        logger.info(f"Loading YOLO model: {self.weights}")
        self._model = YOLO(self.weights)
        logger.info("YOLO model loaded ✓")

    # ─────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────

    def detect(self, image: np.ndarray) -> list[Detection]:
        """
        Run YOLOv8 detection on a single image.

        Args:
            image: np.ndarray with shape (H, W, 3).
                   Accepts both RGB and BGR (YOLO handles internally).

        Returns:
            List of Detection objects, filtered by confidence and class.
            If no detections pass the filter AND fallback_full_image=True,
            returns a single Detection covering the entire image.
        """
        self._load_model()

        results = self._model.predict(
            source=image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False,
        )

        detections: list[Detection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            for box in boxes:
                cls_id   = int(box.cls.item())
                cls_name = result.names.get(cls_id, str(cls_id))
                conf     = float(box.conf.item())

                # Class filter (skip if filter list set and ID not in it)
                if self.filter_ids and cls_id not in self.filter_ids:
                    continue

                xyxy = box.xyxy[0].cpu().numpy().astype(int)
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                detections.append(Detection(
                    bbox=(x1, y1, x2, y2),
                    confidence=conf,
                    class_id=cls_id,
                    class_name=cls_name,
                ))

        if not detections and self.fallback:
            h, w = image.shape[:2]
            logger.debug("No YOLO detections — falling back to full image")
            detections.append(Detection(
                bbox=(0, 0, w, h),
                confidence=1.0,
                class_id=-1,
                class_name="full_image",
            ))

        logger.debug(f"  YOLO → {len(detections)} detection(s)")
        return detections

    def crop_detections(
        self,
        image: np.ndarray,
        detections: list[Detection],
        padding: int = 10,
    ) -> list[tuple[np.ndarray, Detection]]:
        """
        Crop image regions for each detection.

        Args:
            image:      Source image (H, W, 3).
            detections: List of Detection objects.
            padding:    Extra pixels to include around the bounding box.

        Returns:
            List of (cropped_image, detection) tuples.
        """
        h, w   = image.shape[:2]
        crops: list[tuple[np.ndarray, Detection]] = []

        for det in detections:
            x1, y1, x2, y2 = det.bbox
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)

            crop = image[y1:y2, x1:x2]
            crops.append((crop, det))

        return crops
