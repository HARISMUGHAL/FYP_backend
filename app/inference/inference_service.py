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
    """

    def __init__(
        self,
        config_path: str | Path = "configs/config.yaml",
        confidence_threshold: float = 0.7,
    ) -> None:
        self.config_path = Path(config_path)
        self.confidence_threshold = confidence_threshold

        cfg: dict[str, Any] = load_yaml(str(self.config_path))
        self.pipe = InferencePipeline(cfg)

    def predict_frame(self, frame: np.ndarray) -> Prediction:
        """
        Predict one frame using the existing pipeline.

        Preferred call from user requirement:
            pipe.run_on_image(frame)
        Fallbacks are used for compatibility if that method is unavailable.
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

        fruit = str(result.get("fruit", "unknown"))
        grade = result.get("grade")
        label = str(result.get("label", "unknown"))
        confidence = float(result.get("confidence", 0.0))

        if confidence < self.confidence_threshold:
            return Prediction(
                fruit="unknown",
                grade=None,
                confidence=confidence,
                label="unknown",
                status="UNKNOWN",
            )

        return Prediction(
            fruit=fruit,
            grade=str(grade) if grade is not None else None,
            confidence=confidence,
            label=label,
            status="KNOWN",
        )
