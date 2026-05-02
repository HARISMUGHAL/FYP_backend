from __future__ import annotations

from collections import deque
from typing import Deque

from app.inference.inference_service import Prediction


class ConsoleView:
    """Simple readable console renderer for realtime predictions."""

    def __init__(self, history_size: int = 5) -> None:
        self._history: Deque[Prediction] = deque(maxlen=history_size)

    def push_prediction(self, pred: Prediction) -> None:
        self._history.appendleft(pred)

    def render_current(self, pred: Prediction) -> None:
        print("-" * 34)
        print(f"Fruit: {pred.fruit}")
        print(f"Grade: {pred.grade}")
        print(f"Confidence: {pred.confidence:.2f}")
        print(f"Status: {pred.status}")
        print("-" * 34)

    def render_last_predictions(self) -> None:
        print("Last 5 Predictions:")
        if not self._history:
            print("  (no predictions yet)")
            return

        for idx, pred in enumerate(self._history, start=1):
            print(
                f"  {idx}. {pred.fruit} | grade={pred.grade} | "
                f"conf={pred.confidence:.2f} | {pred.status}"
            )
