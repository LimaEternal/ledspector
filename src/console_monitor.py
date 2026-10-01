"""Headless-наблюдение за светодиодами: кадр -> цвет -> частота -> консоль.

Цикл обработки переиспользует `ColorDetector` и `FrequencyAnalyser` из
основного кода — они зависят только от numpy/cv2 и платформенно
независимы. Здесь добавлено то, чего нет в GUI-варианте: вывод в
терминал без окна и устойчивость к обрыву SSH.

Два режима вывода:

* `monitor` — таблица перерисовывается на месте (курсор вверх на N строк,
  без очистки экрана, иначе скроллбек засоряется), а смены состояний
  выводятся в колонке событий под таблицей;
* `log` — только строки на смену состояния, append-only.

Оба режима дополнительно пишут смены в файл, чтобы история пережила
обрыв сессии.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, TextIO, Tuple

import numpy as np
import cv2

from .color_detector import ColorDetector
from .frequency_analyser import FrequencyAnalyser
from .led_config import LED, LedConfig
from .linux_camera import LinuxCamera

__all__ = [
    "LedReading",
    "Event",
    "Monitor",
    "STATE_COLORS",
    "crop_led",
]

# Цвета состояний в терминале. Унаследованы из GUI-варианта
# (test_analyzer.py::_COLORS_BGR), чтобы одна и та же панель читалась
# одинаково в окне и в консоли.
STATE_COLORS: Dict[str, str] = {
    "OFF": "\033[90m",
    "SOLID_ON": "\033[32m",
    "BLINK_1HZ": "\033[93m",
    "BLINK_4HZ": "\033[33m",
    "UNKNOWN": "\033[91m",
    "CALCULATING": "\033[94m",
    "NO_DATA": "\033[90m",
}

COLOR_RESET = "\033[0m"

CALCULATING_STATES = {"CALCULATING", "NO_DATA"}


@dataclass
class LedReading:
    """Текущее состояние одного светодиода."""

    id: str
    state: str
    frequency_hz: float
    color: str
    brightness: float
    is_on: bool

    @property
    def settled(self) -> bool:
        """Готово ли состояние, то есть его уже можно считать достоверным."""
        return self.state not in CALCULATING_STATES


@dataclass
class Event:
    """Смена состояния светодиода."""

    timestamp: float
    led_id: str
    old_state: str
    new_state: str
    color: str

    def format(self, with_time: bool = True) -> str:
        stamp = time.strftime("%H:%M:%S", time.localtime(self.timestamp))
        if not with_time:
            return f"{self.led_id}: {self.old_state} -> {self.new_state} ({self.color})"
        return (
            f"[{stamp}] {self.led_id:<8} {self.old_state:<12} -> "
            f"{self.new_state:<12} {self.color}"
        )


def crop_led(frame: np.ndarray, led: LED) -> Optional[np.ndarray]:
    """Вырезает область LED с обрезкой по границам кадра.

    Пустой массив означает, что прямоугольник целиком вне кадра —
    вызывающий код должен трактовать это как «нет данных», а не как
    «диод погашен».
    """
    height, width = frame.shape[:2]
    x1 = max(0, led.x)
    y1 = max(0, led.y)
    x2 = min(width, led.x + led.w)
    y2 = min(height, led.y + led.h)
    if x2 <= x1 or y2 <= y1:
        return np.array([], dtype=frame.dtype)
    return frame[y1:y2, x1:x2].copy()


class Monitor:
    """Наблюдение за панелью: захват кадров и анализ состояний LED."""

    def __init__(
        self,
        config: LedConfig,
        camera: LinuxCamera,
        window_seconds: float = 2.0,
        use_color: bool = True,
    ) -> None:
        self.config = config
        self.camera = camera
        self.detector = ColorDetector()
        self.analyser = FrequencyAnalyser(window_seconds=window_seconds)
        self.use_color = use_color

        self.frames = 0
        self.dropped = 0
        self.started_at = time.time()
        self.fps = 0.0
        self._fps_window_started = time.monotonic()
        self._fps_window_frames = 0

        self._previous: Dict[str, Tuple[str, str]] = {}
        self._log_handle: Optional[TextIO] = None
        self._events: List[Event] = []
        self._max_events_shown = 8
        self._last_frame: Optional[np.ndarray] = None
        self._snapshot_requested: Optional[str] = None
        self._snapshot_with_grid: bool = True

    # ------------------------------------------------------------------
    # Лог-файл
    # ------------------------------------------------------------------
    def open_log(self, path: Optional[str]) -> None:
        if not path:
            return
        try:
            self._log_handle = open(path, "a", encoding="utf-8", buffering=1)
        except OSError as exc:
            print(f"[monitor] Не удалось открыть лог {path}: {exc}", file=sys.stderr)
        else:
            print(f"[monitor] Лог изменений: {path}")

    def close_log(self) -> None:
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def _write_event(self, event: Event) -> None:
        if self._log_handle is None:
            return
        try:
            self._log_handle.write(event.format() + "\n")
        except OSError:
            pass

    # ------------------------------------------------------------------
    # Захват и анализ
    # ------------------------------------------------------------------
    def _update_fps(self) -> None:
        now = time.monotonic()
        elapsed = now - self._fps_window_started
        if elapsed >= 1.0:
            self.fps = self._fps_window_frames / elapsed
            self._fps_window_started = now
            self._fps_window_frames = 0

    def step(self) -> Optional[Dict[str, LedReading]]:
        """Читает один кадр и возвращает состояния всех LED.

        Возвращает None, если кадр не удалось получить (устройство отвалилось
        или камера занята) — вызывающий код решает, продолжать ли.
        """
        ok, frame = self.camera.read_frame()
        if not ok or frame is None:
            self.dropped += 1
            return None

        self.frames += 1
        self._fps_window_frames += 1
        self._update_fps()
        self._last_frame = frame

        now = time.time()
        readings: Dict[str, LedReading] = {}

        for led in self.config.leds:
            crop = crop_led(frame, led)
            if crop.size == 0:
                readings[led.id] = LedReading(
                    id=led.id,
                    state="NO_DATA",
                    frequency_hz=0.0,
                    color="OFF",
                    brightness=0.0,
                    is_on=False,
                )
                continue

            if self.use_color:
                result = self.detector.detect(crop)
                brightness = float(result["brightness"])
                color_name = str(result["color"])
                is_on = bool(result["is_on"])
            else:
                # Режим подбора экспозиции: важна только яркость, цвет
                # в этот момент ещё не откалиброван и мешает.
                hsv_mean = self._mean_value(crop)
                brightness = hsv_mean
                color_name = "N/A"
                is_on = brightness > 40.0

            self.analyser.update(led.id, brightness, now)
            state = self.analyser.analyze_state(led.id)

            readings[led.id] = LedReading(
                id=led.id,
                state=str(state["state"]),
                frequency_hz=float(state["frequency_hz"]),
                color=color_name,
                brightness=brightness,
                is_on=is_on,
            )

        self._detect_changes(readings)
        self._maybe_save_snapshot()
        return readings

    def _maybe_save_snapshot(self) -> None:
        path = self._snapshot_requested
        if not path:
            return
        self._snapshot_requested = None
        try:
            self.save_annotated(path, with_grid=self._snapshot_with_grid)
        except Exception as exc:  # noqa: BLE001 — снимок не должен ронять цикл
            print(f"[monitor] Ошибка сохранения снимка: {exc}", file=sys.stderr)

    @staticmethod
    def _mean_value(crop: np.ndarray) -> float:
        gray = crop.astype(np.float32).mean(axis=2)
        return float(gray.mean())

    def _detect_changes(self, readings: Dict[str, LedReading]) -> None:
        """Фиксирует смены состояний, игнорируя переходные CALCULATING."""
        for led_id, reading in readings.items():
            if not reading.settled:
                continue
            previous = self._previous.get(led_id)
            current = (reading.state, reading.color)
            if previous == current:
                continue

            if previous is not None:
                event = Event(
                    timestamp=time.time(),
                    led_id=led_id,
                    old_state=previous[0],
                    new_state=reading.state,
                    color=reading.color,
                )
                self._events.append(event)
                self._write_event(event)

            self._previous[led_id] = current

    @property
    def events(self) -> List[Event]:
        return list(self._events)

    def uptime(self) -> float:
        return time.time() - self.started_at

    # ------------------------------------------------------------------
    # Вывод: таблица с перерисовкой
    # ------------------------------------------------------------------
    def format_table(self, readings: Dict[str, LedReading], tty: bool) -> List[str]:
        """Формирует строки таблицы. Последняя строка — сводка."""
        header = (
            f"=== LEDSpector | панель {self.config.panel} | "
            f"{self.camera.actual_width}x{self.camera.actual_height} "
            f"| экспозиция {self._exposure_text()} ==="
        )

        rows = [
            f"{'ID':<10}{'X':>5}{'Y':>5}{'W':>4}{'H':>4}  "
            f"{'STATE':<13}{'Hz':>6}  {'COLOR':<8}{'BRIGHT':>7}"
        ]
        rows.append("-" * 72)

        for led in self.config.leds:
            reading = readings.get(led.id)
            if reading is None:
                continue
            freq = f"{reading.frequency_hz:.2f}" if reading.frequency_hz > 0 else "-"
            # Сначала выравнивание по видимому тексту, потом ANSI-обёртка —
            # иначе коды цвета ломают ширину колонки.
            state_text = f"{reading.state:<13}"
            if tty:
                color = STATE_COLORS.get(reading.state, "\033[97m")
                state_text = f"{color}{state_text}{COLOR_RESET}"
            rows.append(
                f"{led.id:<10}{led.x:>5}{led.y:>5}{led.w:>4}{led.h:>4}  "
                f"{state_text}"
                f"{freq:>6}  {reading.color:<8}{reading.brightness:>7.1f}"
            )

        rows.append("")
        rows.append(self._format_summary())
        return [header, ""] + rows

    def _format_summary(self) -> str:
        return (
            f"кадров {self.frames} | потеряно {self.dropped} | "
            f"FPS {self.fps:.1f} | время {self.uptime():.0f}с | "
            f"смен {len(self._events)}"
        )

    def _exposure_text(self) -> str:
        value = self.camera.get_exposure()
        mode = self.camera.get_auto_exposure()
        if value is None:
            return "н/д"
        return f"{value} ({mode})" if mode else str(value)

    def format_events(self, tty: bool, limit: Optional[int] = None) -> List[str]:
        """Последние смены состояний — колонка событий под таблицей."""
        count = limit if limit is not None else self._max_events_shown
        if not self._events:
            return ["", "Смен состояний пока не было."]
        tail = self._events[-count:]
        lines = ["", f"--- смены состояний (последние {len(tail)}) ---"]
        for event in tail:
            stamp = time.strftime("%H:%M:%S", time.localtime(event.timestamp))
            new_state = f"{event.new_state:<12}"
            if tty and event.new_state in STATE_COLORS:
                new_state = f"{STATE_COLORS[event.new_state]}{new_state}{COLOR_RESET}"
            line = (
                f"[{stamp}] {event.led_id:<8} {event.old_state:<12} -> "
                f"{new_state} {event.color}"
            )
            lines.append(line)
        return lines

    def render_lines(
        self,
        readings: Dict[str, LedReading],
        tty: bool,
        show_events: bool = True,
    ) -> List[str]:
        lines = self.format_table(readings, tty)
        if show_events:
            lines += self.format_events(tty)
        return lines

    # ------------------------------------------------------------------
    # Циклы
    # ------------------------------------------------------------------
    def run_table(
        self,
        tty: bool,
        interval: float = 0.1,
        show_events: bool = True,
        max_iterations: Optional[int] = None,
    ) -> int:
        """Перерисовывает таблицу на месте. Возвращает число смен."""
        if not tty:
            print(
                "[monitor] Вывод не в терминал — перерисовка отключена, "
                "используйте режим 'log'.",
                file=sys.stderr,
            )
            return self.run_log(interval=interval)

        printed = 0
        iterations = 0
        try:
            while max_iterations is None or iterations < max_iterations:
                iterations += 1
                readings = self.step()
                if readings is None:
                    print(
                        f"\033[{printed}A\033[0J"
                        "\033[91m[monitor] кадр потерян\033[0m",
                        end="",
                        flush=True,
                    )
                    printed = 1
                    time.sleep(0.5)
                    continue

                lines = self.render_lines(readings, tty=True, show_events=show_events)
                if printed:
                    sys.stdout.write(f"\033[{printed}A")
                sys.stdout.write("\033[0J")
                sys.stdout.write("\n".join(lines) + "\n")
                sys.stdout.flush()
                printed = len(lines)

                time.sleep(interval)
        except KeyboardInterrupt:
            sys.stdout.write(f"\033[{printed}A\033[0J")
        return len(self._events)

    def run_log(
        self,
        interval: float = 0.1,
        header: bool = True,
        max_iterations: Optional[int] = None,
    ) -> int:
        """Только смены состояний, append-only."""
        if header:
            print(
                f"=== LEDSpector | панель {self.config.panel} | "
                f"{self.camera.actual_width}x{self.camera.actual_height} ==="
            )
            print(
                "Режим: только изменения. "
                "Ctrl+C — остановить. Состояния по умолчанию не выводятся."
            )
            print(f"{'-' * 60}")

        iterations = 0
        logged = 0
        try:
            while max_iterations is None or iterations < max_iterations:
                iterations += 1
                before = len(self._events)
                if self.step() is not None and len(self._events) > before:
                    for event in self._events[before:]:
                        print(event.format(), flush=True)
                        logged += 1
                else:
                    time.sleep(interval)
        except KeyboardInterrupt:
            print(f"\n[monitor] Остановлено. Смен за сессию: {len(self._events)}")
        return len(self._events)

    def snapshot_once(self, tty: bool = False) -> Dict[str, LedReading]:
        """Один прогон анализа без цикла — для отладочных снимков."""
        readings = self.step()
        if readings is None:
            raise RuntimeError("Не удалось получить кадр с камеры")
        return readings

    def save_annotated(self, path: str, with_grid: bool = True) -> bool:
        """Сохраняет последний кадр с разметкой ROI (и опционально сеткой).

        Сделано по флагу, а не всегда: файлы не растут сами по себе, а
        для отладки на сервере достаточно снять один кадр и забрать его.
        """
        if self._last_frame is None:
            print("[monitor] Нет ни одного кадра для сохранения", file=sys.stderr)
            return False
        from pathlib import Path

        from .grid_overlay import annotate_frame

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        annotated = annotate_frame(
            self._last_frame,
            self.config.leds,
            with_grid=with_grid,
        )
        if not cv2.imwrite(str(target), annotated):
            print(f"[monitor] Не удалось сохранить {target}", file=sys.stderr)
            return False
        print(f"[monitor] Аннотированный кадр: {target}")
        return True