from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict

from app.inference.inference_service import Prediction


class BatchTracker:
    """Maintains running counts of graded fruit labels."""

    def __init__(self) -> None:
        self._batch: DefaultDict[str, int] = defaultdict(int)

    def update(self, pred: Prediction) -> None:
        self._batch[pred.label] += 1

    def summary(self) -> dict[str, int]:
        return dict(sorted(self._batch.items()))

    def render_summary(self) -> None:
        print("Batch Summary:")
        data = self.summary()
        if not data:
            print("  (no items yet)")
            return
        for label, count in data.items():
            print(f"  {label}: {count}")
