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

# --- Пересвет -------------------------------------------------------------
# Канал, ушедший в 255, больше не несёт информации: чем больше доля таких
# пикселей, тем сильнее искажены и оттенок, и насыщенность. При клиппинге
# сразу трёх каналов оттенок восстановить нечем в принципе — цвет диода
# теряется безвозвратно, и понижение экспозиции не помогает, потому что
# суммарный поток от соседних диодов остаётся выше порога.
CLIPPED_V: float = 250.0
BLOWN_OUT_FRACTION: float = 0.30

# Диапазоны Hue для цветных светодиодов.
# Для добавления нового цвета — допишите запись в этот словарь.
HUE_RANGES: Dict[str, list[Tuple[int, int]]] = {
    "RED": [(0, 10), (160, 180)],
    "AMBER": [(11, 34)],
    "GREEN": [(35, 85)],
    "BLUE": [(100, 130)],
}

# Статус пересвеченного диода. Вводится отдельно от WHITE, потому что
# пересвет и настоящий белый светодиод нельзя смешивать: первый требует
# правки точки замера или оптики, второй читается корректно.
BLOWN_OUT: str = "BLOWN_OUT"


def _measure(crop: np.ndarray) -> Dict[str, float]:
    """Считает средние H/S/V и долю срезанных каналов."""
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]

    # Кандидат в пересвет — не обязательно тот, у которого среднее V=255:
    # клиппинг может задеть часть ROI, и это тоже портит оттенок.
    clipped_fraction = float(np.mean(value >= CLIPPED_V))

    return {
        "hue": float(np.mean(hsv[:, :, 0])),
        "saturation": float(np.mean(hsv[:, :, 1])),
        "value": float(np.mean(value)),
        "clipped_fraction": clipped_fraction,
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

    def __init__(
        self,
        off_threshold: float = OFF_THRESHOLD,
        blown_out_fraction: float = BLOWN_OUT_FRACTION,
    ) -> None:
        """Инициализация детектора.

        Args:
            off_threshold: Порог яркости, ниже которого диод считается
                выключенным.
            blown_out_fraction: Доля срезанных каналов, выше которой диод
                считается пересвеченным.
        """
        self.off_threshold: float = off_threshold
        self.blown_out_fraction: float = blown_out_fraction

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
                                   # "BLACK"|"OFF"|"UNKNOWN"|"BLOWN_OUT"
                    "brightness": float,   # среднее V, 0.0–255.0
                    "saturation": float,   # среднее S, 0.0–255.0
                    "hue": float,          # среднее H, 0–180
                    "clipped_fraction": float,  # доля пикселей V >= 250
                    "blown_out": bool,
                }
        """
        if crop is None or crop.size == 0 or crop.ndim < 2:
            return {
                "is_on": False,
                "color": "OFF",
                "brightness": 0.0,
                "saturation": 0.0,
                "hue": 0.0,
                "clipped_fraction": 0.0,
                "blown_out": False,
            }

        stats = _measure(crop)
        mean_h = stats["hue"]
        mean_s = stats["saturation"]
        mean_v = stats["value"]
        clipped = stats["clipped_fraction"]
        blown = clipped >= self.blown_out_fraction

        base: Dict[str, Any] = {
            "brightness": mean_v,
            "saturation": mean_s,
            "hue": mean_h,
            "clipped_fraction": clipped,
            "blown_out": blown,
        }

        # --- Выключенный диод ---
        if mean_v < self.off_threshold:
            return {**base, "is_on": False, "color": "OFF"}

        # --- Пересвет: цвет потерян, оттенок не восстановить ---
        # Проверяется до BLACK/WHITE: срезанные каналы дают высокое V и
        # низкий S, из-за чего пересвеченный диод выглядел бы белым.
        if blown:
            return {**base, "is_on": True, "color": BLOWN_OUT}

        # --- Тёмный ненасыщенный кадр: диод есть, но не светится ---
        if mean_v < BLACK_V_THRESHOLD and mean_s < BLACK_S_THRESHOLD:
            return {**base, "is_on": False, "color": "BLACK"}

        # --- Яркий ненасыщенный кадр: белый свет ---
        if mean_v > WHITE_V_THRESHOLD and mean_s < WHITE_S_THRESHOLD:
            return {**base, "is_on": True, "color": "WHITE"}

        color = _classify_hue(mean_h)
        return {**base, "is_on": True, "color": color}