"""Детектор цвета и яркости светодиода (анализ одного кадра)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Пороги и диапазоны HSV (OpenCV: H ∈ [0, 180])
# ---------------------------------------------------------------------------

OFF_THRESHOLD: float = 40.0

# Пороги для нейтральных цветов (dark/desaturated → BLACK, bright → WHITE)
# BLACK_V_THRESHOLD должен быть выше OFF_THRESHOLD, иначе BLACK недостижим.
BLACK_V_THRESHOLD: float = 70.0
BLACK_S_THRESHOLD: float = 50.0
WHITE_V_THRESHOLD: float = 180.0
WHITE_S_THRESHOLD: float = 40.0

# Диапазоны Hue для цветных светодиодов.
# Для добавления нового цвета — допишите запись в этот словарь.
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
    и определяет цвет по средним значениям Hue/Saturation.

    Логика классификации:
        1. Очень низкая яркость (V < ``OFF_THRESHOLD``) → ``OFF``.
        2. Тёмный, ненасыщенный кадр → ``BLACK`` (диод есть, но не горит).
        3. Яркий, ненасыщенный кадр → ``WHITE``.
        4. Цвет по оттенку ``HUE_RANGES`` → ``RED``/``AMBER``/``GREEN``/``BLUE``.
        5. Ничего не совпало → ``UNKNOWN``.

    Для добавления новых цветов достаточно расширить ``HUE_RANGES``
    или добавить дополнительную проверку на базе S/V в ``detect()``.

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
                    "color": str,  # "RED"|"GREEN"|"AMBER"|"BLUE"|"WHITE"|
                                   # "BLACK"|"OFF"|"UNKNOWN"
                    "brightness": float  # 0.0–255.0
                }
        """
        if crop is None or crop.size == 0 or crop.ndim < 2:
            return {"is_on": False, "color": "OFF", "brightness": 0.0}

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        mean_h: float = float(np.mean(hsv[:, :, 0]))
        mean_s: float = float(np.mean(hsv[:, :, 1]))
        mean_v: float = float(np.mean(hsv[:, :, 2]))

        # --- Выключенный диод ---
        if mean_v < self.off_threshold:
            return {"is_on": False, "color": "OFF", "brightness": mean_v}

        # --- Тёмный ненасыщенный кадр: диод есть, но не светится ---
        if mean_v < BLACK_V_THRESHOLD and mean_s < BLACK_S_THRESHOLD:
            return {"is_on": False, "color": "BLACK", "brightness": mean_v}

        # --- Яркий ненасыщенный кадр: белый свет ---
        if mean_v > WHITE_V_THRESHOLD and mean_s < WHITE_S_THRESHOLD:
            return {"is_on": True, "color": "WHITE", "brightness": mean_v}

        color = _classify_hue(mean_h)
        return {"is_on": True, "color": color, "brightness": mean_v}