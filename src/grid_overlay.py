"""Наложение координатной сетки на кадр — для ручной разметки LED.

Сетка двухслойная:

* мелкая, каждый `minor_step` пикселей (по умолчанию 10) — приглушённая,
  позволяет прицельно попасть в мелкий диод 2-5 px;
* крупная, каждый `major_step` пикселей (по умолчанию 50) — с подписями
  значений X и Y по краям кадра.

Подписи вынесены на рамку, а не в пересечения линий: иначе 26x15 подписей
внутри кадра закрывают саму панель, ради которой сетка и нужна.

Рисование идёт через полупрозрачный оверлей, иначе линии поверх светлого
металла корпуса просто пропадают.
"""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .led_config import LED

__all__ = ["GridOptions", "draw_grid", "draw_led_boxes", "annotate_frame"]

MINOR_STEP = 10
MAJOR_STEP = 50

MINOR_COLOR = (110, 110, 110)
MAJOR_COLOR = (0, 220, 255)
LABEL_COLOR = (0, 255, 255)
LED_BOX_COLOR = (0, 255, 0)
LED_CENTER_COLOR = (0, 0, 255)

MINOR_ALPHA = 0.35
MAJOR_ALPHA = 0.95


class GridOptions:
    """Параметры сетки."""

    def __init__(
        self,
        minor_step: int = MINOR_STEP,
        major_step: int = MAJOR_STEP,
        show_minor: bool = True,
        labels: bool = True,
    ) -> None:
        if minor_step <= 0 or major_step <= 0:
            raise ValueError("Шаг сетки должен быть положительным")
        if major_step < minor_step:
            raise ValueError("Крупный шаг не может быть меньше мелкого")
        self.minor_step = minor_step
        self.major_step = major_step
        self.show_minor = show_minor
        self.labels = labels


def _blend_overlay(frame: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    """Наложение оверлея только на пиксели, где он не нулевой.

    Простой `cv2.addWeighted` по всему кадру затемнил бы фон: оверлей
    чёрный там, где нет линий, и его alpha-доля гасила бы исходный кадр.
    Поэтому смешиваем строго по маске линий.
    """
    mask = np.any(overlay != 0, axis=2)
    if not mask.any():
        return frame
    result = frame.copy()
    blended = overlay.astype(np.float32) * alpha + frame.astype(np.float32) * (1.0 - alpha)
    result[mask] = np.clip(blended, 0, 255).astype(np.uint8)[mask]
    return result


def _draw_minor_lines(
    frame: np.ndarray, overlay: np.ndarray, step: int, exclude: Iterable[int]
) -> None:
    height, width = frame.shape[:2]
    exclude_set = set(exclude)
    for x in range(0, width, step):
        if x in exclude_set:
            continue
        cv2.line(overlay, (x, 0), (x, height - 1), MINOR_COLOR, 1)
    for y in range(0, height, step):
        if y in exclude_set:
            continue
        cv2.line(overlay, (0, y), (width - 1, y), MINOR_COLOR, 1)


def _draw_major_lines(frame: np.ndarray, overlay: np.ndarray, step: int) -> None:
    height, width = frame.shape[:2]
    for x in range(0, width, step):
        cv2.line(overlay, (x, 0), (x, height - 1), MAJOR_COLOR, 1)
    for y in range(0, height, step):
        cv2.line(overlay, (0, y), (width - 1, y), MAJOR_COLOR, 1)


def _draw_labels(frame: np.ndarray, step: int) -> None:
    height, width = frame.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.42
    thickness = 1

    # Подписи X идут по верхней кромке, Y — по левой, читаются крест-накрест.
    for x in range(0, width, step):
        text = str(x)
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        pos_x = min(max(x + 3, 1), max(width - tw - 2, 1))
        cv2.putText(frame, text, (pos_x, th + 2), font, scale, LABEL_COLOR, thickness)

    for y in range(step, height, step):
        text = str(y)
        (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
        pos_y = min(max(y - 3, th + 3), height - 3)
        cv2.putText(frame, text, (2, pos_y), font, scale, LABEL_COLOR, thickness)

    axis_text = f"X:{step} Y:{step}px"
    cv2.putText(
        frame,
        axis_text,
        (width - cv2.getTextSize(axis_text, font, scale, thickness)[0][0] - 4, 14),
        font,
        scale,
        LABEL_COLOR,
        thickness,
    )


def draw_grid(
    frame: np.ndarray,
    minor_step: int = MINOR_STEP,
    major_step: int = MAJOR_STEP,
    show_minor: bool = True,
    labels: bool = True,
) -> np.ndarray:
    """Возвращает копию кадра с наложенной сеткой (исходный кадр не меняется)."""
    if frame is None or frame.size == 0:
        raise ValueError("Пустой кадр")

    options = GridOptions(
        minor_step=minor_step,
        major_step=major_step,
        show_minor=show_minor,
        labels=labels,
    )
    annotated = frame.copy()

    if options.show_minor and options.minor_step < options.major_step:
        minor_overlay = np.zeros_like(annotated)
        _draw_minor_lines(
            annotated,
            minor_overlay,
            options.minor_step,
            exclude=range(0, annotated.shape[1], options.major_step),
        )
        annotated = _blend_overlay(annotated, minor_overlay, MINOR_ALPHA)

    major_overlay = np.zeros_like(annotated)
    _draw_major_lines(annotated, major_overlay, options.major_step)
    annotated = _blend_overlay(annotated, major_overlay, MAJOR_ALPHA)

    if options.labels:
        _draw_labels(annotated, options.major_step)

    return annotated


def draw_led_boxes(
    frame: np.ndarray,
    leds: Sequence[LED],
    with_centers: bool = True,
) -> np.ndarray:
    """Обводит прямоугольники LED из конфига и подписывает их ID."""
    annotated = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for led in leds:
        x, y, w, h = led.bbox
        cv2.rectangle(annotated, (x, y), (x + w, y + h), LED_BOX_COLOR, 1)

        if with_centers:
            cx, cy = led.center
            arm = max(3, min(w, h) // 2 + 2)
            cv2.line(annotated, (cx - arm, cy), (cx + arm, cy), LED_CENTER_COLOR, 1)
            cv2.line(annotated, (cx, cy - arm), (cx, cy + arm), LED_CENTER_COLOR, 1)

        label_y = y - 5 if y > 14 else y + h + 14
        cv2.putText(
            annotated, led.id, (max(x, 0), label_y), font, 0.4, LED_BOX_COLOR, 1
        )

    return annotated


def annotate_frame(
    frame: np.ndarray,
    leds: Optional[Sequence[LED]] = None,
    with_grid: bool = False,
    minor_step: int = MINOR_STEP,
    major_step: int = MAJOR_STEP,
) -> np.ndarray:
    """Кадр + сетка + прямоугольники LED. Порядок: сетка под разметкой."""
    annotated = frame
    if with_grid:
        annotated = draw_grid(
            annotated, minor_step=minor_step, major_step=major_step
        )
    if leds:
        annotated = draw_led_boxes(annotated, leds)
    return annotated


def grid_lines_for(
    width: int,
    height: int,
    major_step: int = MAJOR_STEP,
) -> Tuple[List[int], List[int]]:
    """Возвращает позиции крупных линий — для проверки ожиданий в тестах."""
    return (
        list(range(0, width, major_step)),
        list(range(0, height, major_step)),
    )


def save_png(path: str, frame: np.ndarray) -> None:
    """Сохраняет кадр в PNG.

    Именно PNG, а не JPEG: мелкие диоды в 2-5 px после сжатия дают
    артефакты по цвету, из-за которых последующая калибровка цвета
    будет работать по искажённым данным.
    """
    from pathlib import Path

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), frame):
        raise RuntimeError(f"Не удалось сохранить кадр в {target}")
    print(f"[grid] сохранено: {target} ({frame.shape[1]}x{frame.shape[0]})")