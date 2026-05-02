from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from flask import jsonify, request

from ml.utils.helpers import load_yaml


class FruitAnalyzerController:
    """
    Backend-only API controller for realtime fruit grading.
    """

    _pipe = None
    _default_threshold: float = 0.70
    _upload_dir = Path("uploads/predict")

    @classmethod
    def _get_pipeline(cls):
        if cls._pipe is None:
            from ml.pipeline.inference_pipeline import InferencePipeline
            cfg = load_yaml("configs/config.yaml")
            cls._pipe = InferencePipeline(cfg)
        return cls._pipe

    @classmethod
    def _predict_with_pipeline(cls, image_path: Path) -> dict:
        pipe = cls._get_pipeline()

        # Required call intent: pipe.run_on_image(image)
        run_on_image = getattr(pipe, "run_on_image", None)
        if callable(run_on_image):
            out = run_on_image(str(image_path))
        else:
            run_on_image_path = getattr(pipe, "run_on_image_path", None)
            if callable(run_on_image_path):
                out = run_on_image_path(image_path)
            else:
                out = pipe.run(image_path)

        if isinstance(out, list):
            if not out:
                raise RuntimeError("Pipeline returned empty predictions.")
            return out[0]
        if isinstance(out, dict):
            return out
        raise RuntimeError("Unexpected pipeline output format.")

    @staticmethod
    def health():
        return jsonify({"status": "ok"}), 200

    @classmethod
    def predict(cls):
        if "image" not in request.files:
            return jsonify({"error": "image file is required"}), 400

        image_file = request.files["image"]
        if image_file.filename is None or image_file.filename.strip() == "":
            return jsonify({"error": "image file is required"}), 400

        cls._upload_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(image_file.filename).suffix or ".jpg"
        saved_path = cls._upload_dir / f"{uuid4().hex}{suffix}"

        try:
            image_file.save(saved_path)
            pred = cls._predict_with_pipeline(saved_path)

            fruit = str(pred.get("fruit", "unknown"))
            grade = pred.get("grade")
            confidence = float(pred.get("confidence", 0.0))
            bbox = pred.get("bbox")

            threshold_raw = request.form.get("threshold")
            threshold = (
                float(threshold_raw)
                if threshold_raw not in (None, "")
                else cls._default_threshold
            )

            if confidence < threshold:
                fruit = "unknown"
                grade = None
                status = "UNKNOWN"
            else:
                status = "KNOWN"

            return jsonify(
                {
                    "fruit": fruit,
                    "grade": grade,
                    "confidence": round(confidence, 4),
                    "bbox": bbox if bbox is not None else None,
                    "status": status,
                }
            ), 200
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500
