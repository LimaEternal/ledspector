"""Анализатор частоты мигания и состояний светодиодов.

Накапливает историю яркости для каждого ROI и классифицирует
режим горения: OFF, SOLID_ON, BLINK_1HZ, BLINK_4HZ, UNKNOWN.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Пороги классификации
# ---------------------------------------------------------------------------

OFF_THRESHOLD: float = 40.0
SOLID_ON_THRESHOLD: float = 60.0
BINARY_THRESHOLD: float = 50.0

# Диапазоны частот для классификации мигания
FREQ_1HZ_MIN: float = 0.6
FREQ_1HZ_MAX: float = 1.8
FREQ_4HZ_MIN: float = 3.0
FREQ_4HZ_MAX: float = 5.5

# Минимальное количество отсчётов для анализа
MIN_SAMPLES: int = 10


class FrequencyAnalyser:
    """Анализатор временных рядов яркости для определения режима мигания.

    Для каждого ROI поддерживается скользящее окно ``(timestamp, brightness)``.
    По содержимому окна определяется: выключен ли диод, горит постоянно
    или мигает с определённой частотой.

    Example::

        analyser = FrequencyAnalyser(window_seconds=2.0)
        analyser.update("PWR_LED", 210.0, time.time())
        analyser.update("PWR_LED", 5.0,  time.time() + 0.5)
        result = analyser.analyze_state("PWR_LED")
    """

    def __init__(
        self,
        window_seconds: float = 2.0,
        max_buffer_size: int = 120,
    ) -> None:
        """Инициализация анализатора.

        Args:
            window_seconds: Длина временного окна в секундах.
            max_buffer_size: Максимальный размер буфера (защита от
                переполнения при большом FPS).
        """
        self.window_seconds: float = window_seconds
        self.max_buffer_size: int = max_buffer_size
        self._history: Dict[str, deque[Tuple[float, float]]] = {}

    # ------------------------------------------------------------------
    # Публичный API
    # ------------------------------------------------------------------

    def update(
        self, roi_id: str, brightness: float, timestamp: float,
    ) -> None:
        """Добавляет измерение яркости для указанного ROI.

        Записи старше ``window_seconds`` автоматически удаляются.

        Args:
            roi_id: Идентификатор зоны интереса.
            brightness: Текущая яркость (0–255).
            timestamp: Метка времени (``time.time()``).
        """
        if roi_id not in self._history:
            self._history[roi_id] = deque(maxlen=self.max_buffer_size)

        buf = self._history[roi_id]
        buf.append((timestamp, brightness))

        # Удаляем записи, вышедшие за пределы окна
        cutoff = timestamp - self.window_seconds
        while buf and buf[0][0] < cutoff:
            buf.popleft()

    def analyze_state(self, roi_id: str) -> Dict[str, Any]:
        """Определяет текущий режим работы светодиода.

        Args:
            roi_id: Идентификатор зоны интереса.

        Returns:
            Словарь::

                {
                    "state": str,         # "OFF"|"SOLID_ON"|"BLINK_1HZ"|
                                         # "BLINK_4HZ"|"UNKNOWN"|"NO_DATA"|
                                         # "CALCULATING"
                    "frequency_hz": float  # 0.0 если не определена
                }
        """
        buf = self._history.get(roi_id)
        if buf is None or len(buf) == 0:
            return {"state": "NO_DATA", "frequency_hz": 0.0}

        samples: List[Tuple[float, float]] = list(buf)

        if len(samples) < MIN_SAMPLES:
            return {"state": "CALCULATING", "frequency_hz": 0.0}

        brightnesses = [b for _, b in samples]
        max_b = max(brightnesses)
        min_b = min(brightnesses)

        # --- OFF ---
        if max_b < OFF_THRESHOLD:
            return {"state": "OFF", "frequency_hz": 0.0}

        # --- SOLID ON ---
        if min_b > SOLID_ON_THRESHOLD:
            return {"state": "SOLID_ON", "frequency_hz": 0.0}

        # --- Blinking analysis ---
        return self._analyze_blink(samples)

    def reset(self, roi_id: Optional[str] = None) -> None:
        """Сбрасывает историю для одного или всех ROI.

        Args:
            roi_id: Если указан — сбрасывает только этот ROI.
                Если ``None`` — очищает всё.
        """
        if roi_id is not None:
            self._history.pop(roi_id, None)
        else:
            self._history.clear()

    # ------------------------------------------------------------------
    # Внутренняя логика
    # ------------------------------------------------------------------

    def _analyze_blink(
        self, samples: List[Tuple[float, float]],
    ) -> Dict[str, Any]:
        """Вычисляет частоту мигания по восходящим фронтам.

        Args:
            samples: Список ``(timestamp, brightness)``.

        Returns:
            Словарь с полями ``state`` и ``frequency_hz``.
        """
        brightnesses = [b for _, b in samples]
        timestamps = [t for t, _ in samples]

        # Бинарный массив: 1 — горит, 0 — не горит
        binary = [1 if b > BINARY_THRESHOLD else 0 for b in brightnesses]

        # Подсчёт восходящих фронтов (0 → 1)
        rising_edges = 0
        for i in range(1, len(binary)):
            if binary[i - 1] == 0 and binary[i] == 1:
                rising_edges += 1

        dt = timestamps[-1] - timestamps[0]
        if dt <= 0.0:
            return {"state": "UNKNOWN", "frequency_hz": 0.0}

        frequency = rising_edges / dt

        if FREQ_1HZ_MIN <= frequency <= FREQ_1HZ_MAX:
            return {"state": "BLINK_1HZ", "frequency_hz": round(frequency, 2)}

        if FREQ_4HZ_MIN <= frequency <= FREQ_4HZ_MAX:
            return {"state": "BLINK_4HZ", "frequency_hz": round(frequency, 2)}

        return {"state": "UNKNOWN", "frequency_hz": round(frequency, 2)}
