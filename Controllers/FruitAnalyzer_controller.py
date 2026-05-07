from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from flask import jsonify, request
from PIL import Image, UnidentifiedImageError

from ml.utils.helpers import load_yaml, setup_logger

logger = setup_logger(__name__)


class FruitAnalyzerController:
    """
    Backend-only API controller for realtime fruit grading.
    """

    _pipe = None
    # NOTE: _default_threshold is NOT USED — the predictor's config
    # threshold (inference.confidence_threshold) is the single source of truth.
    _default_threshold: float = 0.70  # legacy, kept for reference only
    _upload_dir = Path("uploads/predict")
    _allowed_exts = {".jpg", ".jpeg", ".png", ".webp"}
    _file_keys = ("image", "file", "photo")

    @classmethod
    def _get_pipeline(cls):
        if cls._pipe is None:
            try:
                from ml.pipeline.inference_pipeline import InferencePipeline
                cfg = load_yaml("configs/config.yaml")
                logger.info("[BOOT] FruitAnalyzerController initializing InferencePipeline")
                
                cls._pipe = InferencePipeline(cfg)
                
                # ── FORCE SMOOTHING ON FOR REALTIME API ──
                # The Flutter app sends live camera frames to /predict.
                # We MUST enable temporal smoothing to stop UI flicker.
                cls._pipe._classifier.enable_smoothing()
                logger.info("[BOOT] Temporal smoothing FORCED ON globally for API")
                
                # Print active config values to verify they loaded
                conf_th = cfg.get("inference", {}).get("confidence_threshold")
                model_p = cfg.get("model", {}).get("save_path")
                yolo_p = cfg.get("detection", {}).get("weights")
                logger.info(f"[BOOT] Loaded Config | Threshold: {conf_th} | Model: {model_p} | YOLO: {yolo_p}")

                # ── FORCE CLEAN START ──
                # Clear out any leftover temp images from previous crashed runs
                try:
                    if cls._upload_dir.exists():
                        count = 0
                        for f in cls._upload_dir.iterdir():
                            if f.is_file():
                                f.unlink(missing_ok=True)
                                count += 1
                        logger.info(f"[BOOT] Cleaned {count} stale temp files from {cls._upload_dir}")
                except Exception as e:
                    logger.warning(f"[BOOT] Failed to clean temp dir: {e}")
                
            except Exception as e:
                import traceback
                logger.error(f"Pipeline init failed: {e}")
                traceback.print_exc()
                raise
        return cls._pipe

    @classmethod
    def _predict_with_pipeline(cls, image_path: Path) -> dict:
        pipe = cls._get_pipeline()
        
        # Log that smoothing is active
        if getattr(pipe._classifier, "_smoothing_enabled", False):
            logger.debug("[SMOOTHING ACTIVE] Processing frame with temporal history")
        
        # pipe.run() is the canonical entry point — always returns list[dict]
        out = pipe.run(image_path)

        if isinstance(out, list):
            if not out:
                raise RuntimeError("Pipeline returned empty predictions.")
            return out[0]
        if isinstance(out, dict):
            return out
        raise RuntimeError("Unexpected pipeline output format.")

    @staticmethod
    def _suffix_from_upload(filename: str) -> str | None:
        suffix = Path(filename).suffix.lower()
        return suffix if suffix in FruitAnalyzerController._allowed_exts else None

    @staticmethod
    def _suffix_from_saved_file(path: Path) -> str:
        try:
            with Image.open(path) as img:
                fmt = (img.format or "").upper()
        except (UnidentifiedImageError, OSError, ValueError):
            return ".jpg"

        mapping = {"JPEG": ".jpg", "PNG": ".png", "WEBP": ".webp"}
        return mapping.get(fmt, ".jpg")

    @staticmethod
    def health():
        return jsonify({"status": "ok"}), 200

    @classmethod
    def predict(cls):
        # Clean structured logging (one line per request)
        logger.debug(f"[PREDICT] content_type={request.content_type} files={list(request.files.keys())}")

        try:
            # Mandatory safe access path
            image_file = request.files.get("image")
            used_key = "image"

            # Backward-compatible fallback keys (only from request.files)
            if image_file is None:
                for key in ("file", "photo"):
                    cand = request.files.get(key)
                    if cand is not None:
                        image_file = cand
                        used_key = key
                        break

            if image_file is None:
                logger.debug("[PREDICT] chosen file field: None")
                return jsonify({"error": "image file is required"}), 400

            filename = (image_file.filename or "").strip()
            logger.debug(f"[PREDICT] chosen file field: {used_key}")
            logger.debug(f"[PREDICT] received filename: {filename!r}")
            if not filename:
                return jsonify({"error": "image file is required"}), 400

            cls._upload_dir.mkdir(parents=True, exist_ok=True)
            suffix = cls._suffix_from_upload(filename) or ".jpg"
            temp_path = cls._upload_dir / f"{uuid4().hex}{suffix}"
            image_file.save(temp_path)

            real_suffix = cls._suffix_from_saved_file(temp_path)
            allowed_formats = {"JPEG", "PNG", "WEBP"}
            try:
                with Image.open(temp_path) as verify_img:
                    if (verify_img.format or "").upper() not in allowed_formats:
                        temp_path.unlink(missing_ok=True)
                        return jsonify({"error": "invalid image file"}), 400
            except (UnidentifiedImageError, OSError, ValueError):
                temp_path.unlink(missing_ok=True)
                return jsonify({"error": "invalid image file"}), 400

            if real_suffix != suffix:
                final_path = temp_path.with_suffix(real_suffix)
                temp_path.rename(final_path)
                saved_path = final_path
            else:
                saved_path = temp_path

            pred = cls._predict_with_pipeline(saved_path)

            # ── Cleanup: remove temp upload file after inference ───
            try:
                saved_path.unlink(missing_ok=True)
            except OSError:
                pass  # non-critical, don't crash on cleanup failure

            logger.debug(f"Pipeline output: fruit={pred.get('fruit')} conf={pred.get('confidence')}")

            # ── Trust the predictor's threshold decision ──────────
            # The predictor (FruitGradePredictor) is the SINGLE source
            # of truth for the confidence threshold.  Do NOT re-check
            # or override the unknown / status flags here.
            fruit      = str(pred.get("fruit", "unknown"))
            grade      = pred.get("grade")
            confidence = float(pred.get("confidence", 0.0))
            bbox       = pred.get("bbox", None)
            status     = str(pred.get("status", "UNKNOWN" if pred.get("unknown", True) else "KNOWN"))
            label      = str(pred.get("label", "unknown"))

            return jsonify(
                {
                    "fruit": fruit,
                    "grade": grade,
                    "label": label,
                    "confidence": round(confidence, 4),
                    "bbox": bbox if bbox is not None else None,
                    "status": status,
                    "top_k": pred.get("top_k", []),
                    "explanation": pred.get("explanation", ""),
                }
            ), 200
        except ValueError:
            return jsonify({"error": "invalid threshold value"}), 400
        except Exception as exc:
            return jsonify({"error": str(exc)}), 500
