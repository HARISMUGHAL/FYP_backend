from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.batch.batch_tracker import BatchTracker
from app.camera.camera_service import CameraConfig, CameraService
from app.inference.inference_service import InferenceService
from app.ui.console_view import ConsoleView


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FruitAnalyzer realtime app")
    parser.add_argument(
        "--config",
        default=str(BASE_DIR / "configs" / "config.yaml"),
        help="Path to YAML config.",
    )
    parser.add_argument(
        "--camera-source",
        default="0",
        help="Camera index (0/1/2) or camera URL.",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=3.0,
        help="Capture interval in seconds (default: 3).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.55,
        help="Confidence threshold (display only — actual threshold is in config.yaml).",
    )
    parser.add_argument(
        "--summary-every",
        type=int,
        default=5,
        help="Print batch summary every N predictions.",
    )
    return parser.parse_args()


def _parse_camera_source(raw_source: str) -> str | int:
    # Numeric values are interpreted as webcam indexes.
    return int(raw_source) if raw_source.isdigit() else raw_source


def main() -> None:
    args = parse_args()

    camera_source = _parse_camera_source(args.camera_source)
    camera = CameraService(CameraConfig(source=camera_source))
    infer = InferenceService(
        config_path=args.config,
    )
    ui = ConsoleView(history_size=5)
    batch = BatchTracker()

    print("=" * 60)
    print("FruitAnalyzer Realtime System Started")
    print(f"Camera Source : {camera_source}")
    print(f"Capture Every : {args.interval:.1f}s")
    print(f"Threshold     : {args.threshold:.2f}")
    print("Press Ctrl+C to stop.")
    print("=" * 60)

    prediction_count = 0
    try:
        while True:
            loop_start = time.time()

            try:
                frame = camera.capture_frame()
            except Exception as exc:
                print(f"[WARN] Camera capture failed: {exc}")
                camera.close()
                time.sleep(1.0)
                continue

            try:
                pred = infer.predict_frame(frame)
            except Exception as exc:
                print(f"[WARN] Prediction failed, skipping frame: {exc}")
                elapsed = time.time() - loop_start
                time.sleep(max(0.0, args.interval - elapsed))
                continue

            prediction_count += 1
            batch.update(pred)
            ui.push_prediction(pred)

            ui.render_current(pred)
            ui.render_last_predictions()

            if prediction_count % max(1, args.summary_every) == 0:
                batch.render_summary()
                print("-" * 60)

            elapsed = time.time() - loop_start
            time.sleep(max(0.0, args.interval - elapsed))

    except KeyboardInterrupt:
        print("\n[INFO] Stopping FruitAnalyzer...")
    finally:
        camera.close()
        print("[INFO] Final Batch Summary:")
        batch.render_summary()


if __name__ == "__main__":
    main()
