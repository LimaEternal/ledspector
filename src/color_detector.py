"""Детектор цвета и яркости светодиода (анализ одного кадра)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Пороги и диапазоны HSV (OpenCV: H ∈ [0, 180])
# ---------------------------------------------------------------------------

OFF_THRESHOLD: float = 40.0

HUE_RANGES: Dict[str, list[Tuple[int, int]]] = {
    "RED": [(0, 10), (160, 180)],
    "AMBER": [(11, 34)],
    "GREEN": [(35, 85)],
    "BLUE": [(100, 130)],
}


def _classify_hue(mean_hue: float) -> str:
    """Определяет имя цвета по среднему значению Hue.

    Args:
        mean_hue: Среднее значение канала H (0–180).

    Returns:
        Строка-идентификатор цвета: ``"RED"``, ``"AMBER"``, ``"GREEN"``,
        ``"BLUE"`` или ``"UNKNOWN"``.
    """
    for color_name, ranges in HUE_RANGES.items():
        for lo, hi in ranges:
            if lo <= mean_hue <= hi:
                return color_name
    return "UNKNOWN"


class ColorDetector:
    """Детектор состояния и цвета светодиода по BGR-кропу.

    Анализирует один кадр: вычисляет среднюю яркость (V-канал HSV)
    и определяет цвет по среднему Hue.

    Example::

        detector = ColorDetector()
        result = detector.detect(bgr_crop)
        print(result)
        # {"is_on": True, "color": "GREEN", "brightness": 215.4}
    """

    def __init__(self, off_threshold: float = OFF_THRESHOLD) -> None:
        """Инициализация детектора.

        Args:
            off_threshold: Порог яркости, ниже которого диод считается
                выключенным.
        """
        self.off_threshold: float = off_threshold

    def detect(self, crop: np.ndarray) -> Dict[str, Any]:
        """Анализ BGR-кропа светодиода.

        Args:
            crop: Вырезанный BGR-кадр (``numpy.ndarray``). Может быть
                пустым массивом.

        Returns:
            Словарь::

                {
                    "is_on": bool,
                    "color": str,       # "RED"|"GREEN"|"AMBER"|"BLUE"|"OFF"|"UNKNOWN"
                    "brightness": float  # 0.0–255.0
                }
        """
        if crop is None or crop.size == 0 or crop.ndim < 2:
            return {"is_on": False, "color": "OFF", "brightness": 0.0}

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        mean_h: float = float(np.mean(hsv[:, :, 0]))
        mean_v: float = float(np.mean(hsv[:, :, 2]))

        if mean_v < self.off_threshold:
            return {"is_on": False, "color": "OFF", "brightness": mean_v}

        color = _classify_hue(mean_h)
        return {"is_on": True, "color": color, "brightness": mean_v}
