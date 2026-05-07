from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ml.pipeline.inference_pipeline import InferencePipeline
from ml.utils.helpers import load_yaml


@dataclass
class Prediction:
    fruit: str
    grade: Optional[str]
    confidence: float
    label: str
    status: str


class InferenceService:
    """
    Thin wrapper around existing ML pipeline.
    Does not modify or reimplement pipeline behavior.

    IMPORTANT: The confidence threshold is applied ONCE inside the
    predictor (FruitGradePredictor).  This service does NOT re-check
    or override the threshold decision.
    """

    def __init__(
        self,
        config_path: str | Path = "configs/config.yaml",
    ) -> None:
        self.config_path = Path(config_path)

        cfg: dict[str, Any] = load_yaml(str(self.config_path))
        self.pipe = InferencePipeline(cfg)

    def predict_frame(self, frame: np.ndarray) -> Prediction:
        """
        Predict one frame using the existing pipeline.

        The pipeline's predictor already applies the confidence threshold
        from config.yaml.  This method trusts that decision and does NOT
        apply a second threshold check.
        """
        result: Any

        run_on_image = getattr(self.pipe, "run_on_image", None)
        if callable(run_on_image):
            result = run_on_image(frame)
        else:
            run_on_video_frame = getattr(self.pipe, "run_on_video_frame", None)
            if callable(run_on_video_frame):
                result = run_on_video_frame(frame)
            else:
                result = self.pipe.run(frame)

        # The pipeline usually returns a list; consume the first prediction.
        if isinstance(result, list):
            if not result:
                raise RuntimeError("Pipeline returned empty prediction list.")
            result = result[0]

        # ── Trust the predictor's threshold decision ──────────────
        # Do NOT re-check confidence against a local threshold here.
        # The predictor is the SINGLE source of truth.
        fruit      = str(result.get("fruit", "unknown"))
        grade      = result.get("grade")
        label      = str(result.get("label", "unknown"))
        confidence = float(result.get("confidence", 0.0))
        status     = str(result.get("status", "UNKNOWN" if result.get("unknown", True) else "KNOWN"))

        return Prediction(
            fruit=fruit,
            grade=str(grade) if grade is not None else None,
            confidence=confidence,
            label=label,
            status=status,
        )
