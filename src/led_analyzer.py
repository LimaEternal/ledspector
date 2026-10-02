"""Подбор точки замера светодиода вне зоны пересвета.

Задача модуля — помочь с настройкой без физического доступа к серверу.
Диоды на панели стоят рядом и засвечивают друг друга: у каждого горит
ореол радиусом в десятки пикселей, и ореолы соседей перекрываются. В
центре горевшего диода каналы матрицы срезаются в 255 одновременно, и
цвет теряется безвозвратно — S падает почти до нуля, оттенок уезжает,
детектор объявляет WHITE или UNKNOWN.

Снижение экспозиции тут не помогает: приёмник суммирует поток от всех
диодов сразу, поэтому общий уровень остаётся выше порога клиппинга, а
отношение сигнал/фон не меняется.

Что помогает без правки оптики — сдвинуть точку замера от центра горевшего
диода на край его ореола. Там яркость ниже порога клиппинга, а
насыщенность, наоборот, высокая: рассеянный свет насыщенного цвета на
тёмном фоне сохраняет S, тогда как смешивание с белым убивает его.
Измерения на реальном кадре показывают устойчивый коридор: на радиусе
примерно от трети до двух третей ореола V падает со 255 до ~60-150 при S,
растущем до 100-220, и оттенок держится постоянным.

Радиус у каждого диода свой, поэтому искать его вручную по картинке
неудобно. Здесь он находится автоматически, а результат печатается
готовым блоком для конфига.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .color_detector import CLIPPED_V, OFF_THRESHOLD, ColorDetector

__all__ = [
    "LedCandidate",
    "LedMeasurement",
    "find_led_blobs",
    "measure_led",
    "suggest_measurement_point",
    "analyze_frame",
    "format_config_lines",
    "draw_measurements",
    "DEFAULT_MIN_SATURATION",
    "DEFAULT_MAX_CLIPPED",
    "OFF_LEVEL_V",
]

# Пороги пригодности точки замера. Значения подобраны по кадру 1280x720:
# точка годится, если каналы не срезаны, а цвета хватает для оттенка.
DEFAULT_MAX_CLIPPED: float = 0.02
DEFAULT_MIN_SATURATION: float = 60.0

# Минимальная яркость пригодной точки. Ореол, доведённый почти до тишины,
# классифицируется как OFF, что хуже пересвета с понятной причиной.
# Порог намеренно низкий: у соседних диодов ореолы перекрываются, и зона
# пригодного замера узкая — отсек по яркости здесь убивал бы часть панелей.
OFF_LEVEL_V: float = 45.0

# Радиус поиска ореола вокруг центра диода.
MAX_HALO_RADIUS: int = 70
RADIUS_STEP: int = 2

# Насколько оттенки считаются одним и тем же цветом (в градусах HSV).
HUE_TOLERANCE: float = 12.0


def is_confusable(
    hue: float,
    known_hues: Sequence[float],
    tolerance: float = HUE_TOLERANCE,
) -> bool:
    """Проверяет, не совпадает ли оттенок с уже известным диодом.

    Оттенок сравнивается по кругу: 179 и 1 — это один и тот же красный,
    поэтому наивное сравнение на разность дало бы неверный ответ.
    """
    if not known_hues:
        return False
    for known in known_hues:
        delta = abs((hue - known + 90.0) % 180.0 - 90.0)
        if delta <= tolerance:
            return True
    return False


# Направления, в которых пробуется поставить точку замера. Вверх —
# первым: там панель обычно пустая и ореол соседей задет реже всего.
_MEASUREMENT_DIRECTIONS = ((0, -1), (1, 0), (-1, 0), (0, 1), (1, -1), (-1, 1))


def _candidate_boxes(
    cx: int,
    cy: int,
    radius: int,
    window: int,
    width: int,
    height: int,
) -> List[Tuple[int, int]]:
    """Позиции окна замера на кольце заданного радиуса, в порядке приоритета.

    Точка должна лежать на самом диоде, поэтому смещение всегда ровно на
    радиус: при меньшем смещении она снова попадёт в срезанную зону.
    """
    boxes: List[Tuple[int, int]] = []
    half = window // 2

    for dx, dy in _MEASUREMENT_DIRECTIONS:
        box_x = cx + dx * radius - half
        box_y = cy + dy * radius - half
        if not (0 <= box_x <= width - window and 0 <= box_y <= height - window):
            continue
        candidate = (box_x, box_y)
        if candidate not in boxes:
            boxes.append(candidate)

    return boxes


# Насколько оттенки считаются одним и тем же цветом (в градусах HSV).

# Цвета, которые означают, что точка замера не годится: диод не виден
# либо оттенок нечитаем. WHITE и UNKNOWN отбрасываются, потому что при
# перекрывающихся ореолах они чаще означают подмешивание чужого диода,
# а не настоящий белый светодиод.
_REJECTED_COLORS = frozenset({"OFF", "BLACK", "WHITE", "UNKNOWN", "BLOWN_OUT"})

# Минимальная площадь пятна, чтобы считать его диодом, а не бликом.
MIN_BLOB_AREA: int = 150
BLOB_THRESHOLD_V: int = 180


@dataclass
class LedCandidate:
    """Кандидат в диод: найденное светящееся пятно."""

    cx: int
    cy: int
    area: int
    peak_v: int
    peak_s: int
    peak_h: int

    @property
    def center(self) -> Tuple[int, int]:
        return (self.cx, self.cy)


@dataclass
class LedMeasurement:
    """Результат подбора точки замера для одного диода."""

    cx: int
    cy: int
    status: str
    detail: str
    suggested: Optional[Tuple[int, int, int, int]] = None
    suggested_radius: Optional[int] = None
    center_v: float = 0.0
    center_s: float = 0.0
    center_clipped: float = 0.0
    best_v: float = 0.0
    best_s: float = 0.0
    profile: List[Tuple[int, float, float]] = field(default_factory=list)

    @property
    def needs_move(self) -> bool:
        return self.suggested is not None


def _ring_samples(
    frame: np.ndarray,
    cx: int,
    cy: int,
    radius: int,
    step_deg: int = 10,
) -> Optional[Tuple[float, float, float, int]]:
    """Усредняет V/S/H и считает пересвет по кольцу заданного радиуса.

    Кольцо, а не круг: центр диода всегда срезан, а край кольца на том
    же радиусе — однородная зона. Так замер не зависит от смещения
    относительно геометрии диода.
    """
    height, width = frame.shape[:2]
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    values: List[int] = []
    sats: List[int] = []
    hues: List[int] = []
    clipped = 0
    total = 0

    for deg in range(0, 360, step_deg):
        rad = math.radians(deg)
        x = int(round(cx + radius * math.cos(rad)))
        y = int(round(cy + radius * math.sin(rad)))
        if not (0 <= x < width and 0 <= y < height):
            continue
        pixel_hsv = hsv[y, x]
        values.append(int(pixel_hsv[2]))
        sats.append(int(pixel_hsv[1]))
        hues.append(int(pixel_hsv[0]))
        total += 1
        if pixel_hsv[2] >= CLIPPED_V:
            clipped += 1

    if not total:
        return None

    # Оттенок усредняем по кругу: у зелёного диода H≈80-86, у соседних
    # оттенков вроде 20 или 150 это дало бы мусор, поэтому при отбраковке
    # кольца смотрим на S, а не на среднее H.
    return (
        float(np.mean(values)),
        float(np.mean(sats)),
        float(np.median(hues)),
        clipped / total,
    )


def _center_stats(
    frame: np.ndarray, cx: int, cy: int, size: int = 5
) -> Tuple[float, float, float]:
    """V, S и доля срезанных каналов в квадрате по центру диода."""
    half = size // 2
    height, width = frame.shape[:2]
    x1, y1 = max(0, cx - half), max(0, cy - half)
    x2, y2 = min(width, cx + half + 1), min(height, cy + half + 1)
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return (0.0, 0.0, 0.0)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return (
        float(np.mean(hsv[:, :, 2])),
        float(np.mean(hsv[:, :, 1])),
        float(np.mean(hsv[:, :, 2] >= CLIPPED_V)),
    )


def find_led_blobs(
    frame: np.ndarray,
    threshold_v: int = BLOB_THRESHOLD_V,
    min_area: int = MIN_BLOB_AREA,
) -> List[LedCandidate]:
    """Находит светящиеся пятна — кандидаты в диоды.

    Работает по яркости, а не по цвету: у пересвеченных диодов цвет и
    нечитаем, а свечение остаётся самым заметным признаком.
    """
    if frame is None or frame.size == 0:
        return []

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = (hsv[:, :, 2] >= threshold_v).astype(np.uint8)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 8)

    candidates: List[LedCandidate] = []
    for index in range(1, count):
        area = int(stats[index, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cx = int(centroids[index][0])
        cy = int(centroids[index][1])
        component = labels == index
        v_channel = hsv[:, :, 2][component]
        s_channel = hsv[:, :, 1][component]
        h_channel = hsv[:, :, 0][component]
        candidates.append(
            LedCandidate(
                cx=cx,
                cy=cy,
                area=area,
                peak_v=int(v_channel.max()),
                peak_s=int(np.median(s_channel)),
                peak_h=int(np.median(h_channel)),
            )
        )

    candidates.sort(key=lambda c: (-c.area, c.cy, c.cx))
    return candidates


def suggest_measurement_point(
    frame: np.ndarray,
    cx: int,
    cy: int,
    max_clipped: float = DEFAULT_MAX_CLIPPED,
    min_saturation: float = DEFAULT_MIN_SATURATION,
    min_value: float = OFF_LEVEL_V,
    max_radius: int = MAX_HALO_RADIUS,
    window: int = 5,
    known_hues: Sequence[float] = (),
    hue_tolerance: float = HUE_TOLERANCE,
) -> Tuple[Optional[int], Optional[Tuple[int, int, int, int]], List[Tuple[int, float, float]]]:
    """Ищёт ближайшее к диоду кольцо, пригодное для замера.

    Перебирает радиусы от центра наружу и берёт **первый** пригодный —
    чем ближе к диоду, тем надёжнее замер: дальше по радиусу начинается
    фон соседнего ореола, и диод может «исчезнуть» в шуме.

    Кольцо считается пригодным, если каналы не срезаны, насыщенности
    хватает для оттенка и яркость ещё выше фона. Последнее условие
    обязательно: ореол, доведённый до тишины, даёт S под 255 и
    классифицируется как OFF, что хуже, чем пересвет с понятной причиной.

    Возвращает (радиус, (x, y, w, h), профиль).
    """
    profile: List[Tuple[int, float, float]] = []
    detector = ColorDetector()
    height, width = frame.shape[:2]

    for radius in range(2, max_radius + 1, RADIUS_STEP):
        sampled = _ring_samples(frame, cx, cy, radius)
        if sampled is None:
            continue
        value, saturation, _hue, clipped = sampled
        profile.append((radius, value, saturation))

        if clipped > max_clipped or saturation < min_saturation:
            continue
        if value < min_value:
            continue

        # Решающий голос у самого детектора, а не у порогов: кольцо
        # может формально пройти пороги и всё равно дать BLOWN_OUT,
        # потому что пороги усредняют, а детектор ловит факт клиппинга.
        # Радиусы перебираются от центра наружу, поэтому берётся самый
        # близкий к диоду из действительно годных — там сигнал сильнее.
        #
        # Направление перебирается, а не берётся одно: диоды у нижнего
        # края кадра не имели бы точки при фиксированном смещении вверх.
        for box_x, box_y in _candidate_boxes(cx, cy, radius, window, width, height):
            crop = frame[box_y:box_y + window, box_x:box_x + window]
            if crop.size == 0:
                continue

            verdict = detector.detect(crop)
            # Срез по пересвету строже кольцевого: 5x5 может целиком не
            # попасть в пересвеченное, и одна такая точка среди пригодных
            # сделала бы замер нестабильным от кадра к кадру.
            if verdict["clipped_fraction"] > max_clipped:
                continue
            if verdict["blown_out"] or verdict["color"] in _REJECTED_COLORS:
                continue

            return (radius, (box_x, box_y, window, window), profile)

    return (None, None, profile)

    return (None, None, profile)


def measure_led(
    frame: np.ndarray,
    cx: int,
    cy: int,
    detector: Optional[ColorDetector] = None,
    max_clipped: float = DEFAULT_MAX_CLIPPED,
    min_saturation: float = DEFAULT_MIN_SATURATION,
    window: int = 5,
    known_hues: Sequence[float] = (),
    hue_tolerance: float = HUE_TOLERANCE,
) -> LedMeasurement:
    """Анализирует один диод и возвращает вердикт с рекомендацией."""
    detector = detector or ColorDetector()

    center_v, center_s, center_clipped = _center_stats(frame, cx, cy, window)
    radius, box, profile = suggest_measurement_point(
        frame,
        cx,
        cy,
        max_clipped=max_clipped,
        min_saturation=min_saturation,
        window=window,
        known_hues=known_hues,
        hue_tolerance=hue_tolerance,
    )

    if radius is None or box is None:
        return LedMeasurement(
            cx=cx,
            cy=cy,
            status="NO_HALO",
            detail=(
                "Пригодной точки вне пересвета не найдено: ореол либо "
                "срезан целиком, либо соседи перекрывают его. Нужна правка "
                "оптики (маска на диоды, рассеиватель) или снижение gain."
            ),
            center_v=center_v,
            center_s=center_s,
            center_clipped=center_clipped,
            profile=profile,
        )

    suggested_v, suggested_s, _hue, _clipped = _ring_samples(frame, cx, cy, radius)

    if center_clipped >= detector.blown_out_fraction:
        status = "MOVE"
        detail = (
            f"Центр срезан ({100 * center_clipped:.0f}% каналов в 255), "
            f"цвет недоступен. Замер переносится на r={radius}: "
            f"V={suggested_v:.0f} S={suggested_s:.0f}."
        )
    else:
        status = "OK"
        detail = (
            f"Центр пригоден (V={center_v:.0f} S={center_s:.0f}). "
            f"Запасной радиус на случай пересвета — r={radius}."
        )

    return LedMeasurement(
        cx=cx,
        cy=cy,
        status=status,
        detail=detail,
        suggested=box,
        suggested_radius=radius,
        center_v=center_v,
        center_s=center_s,
        center_clipped=center_clipped,
        best_v=suggested_v,
        best_s=suggested_s,
        profile=profile,
    )


def analyze_frame(
    frame: np.ndarray,
    detector: Optional[ColorDetector] = None,
    min_area: int = MIN_BLOB_AREA,
    max_clipped: float = DEFAULT_MAX_CLIPPED,
    min_saturation: float = DEFAULT_MIN_SATURATION,
    window: int = 5,
    hue_tolerance: float = HUE_TOLERANCE,
) -> List[LedMeasurement]:
    """Находит все диоды в кадре и подбирает для каждого точку замера.

    Диоды обрабатываются независимо друг от друга: на панели законно
    стоят несколько диодов одного цвета, поэтому сверять оттенки между
    ними нельзя.
    """
    detector = detector or ColorDetector()
    results: List[LedMeasurement] = []

    for candidate in find_led_blobs(frame, min_area=min_area):
        results.append(
            measure_led(
                frame,
                candidate.cx,
                candidate.cy,
                detector=detector,
                max_clipped=max_clipped,
                min_saturation=min_saturation,
                window=window,
            )
        )

    return results


def format_config_lines(
    measurements: Sequence[LedMeasurement],
    window_size: int = 5,
    prefix: str = "LED_",
) -> List[str]:
    """Готовит строки конфига из подобранных точек замера.

    Диоды без пригодной точки пропускаются: в конфиг попадает только то,
    что действительно можно замерять.
    """
    lines: List[str] = []
    for number, measurement in enumerate(measurements, start=1):
        if measurement.suggested is None:
            continue
        x, y, w, h = measurement.suggested
        lines.append(f"{prefix}{number}  {x}  {y}  {window_size}  {window_size}")
    return lines


def draw_measurements(
    frame: np.ndarray,
    measurements: Sequence[LedMeasurement],
) -> np.ndarray:
    """Рисует найденные диоды, их центры и предложенные точки замера."""
    annotated = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX

    for number, measurement in enumerate(measurements, start=1):
        # Ореол целиком — контекст, где искалась пригодная точка.
        if measurement.suggested_radius:
            cv2.circle(
                annotated,
                (measurement.cx, measurement.cy),
                measurement.suggested_radius,
                (255, 200, 0),
                1,
            )

        # Центр: красный крест — он срезан, замерять его нельзя.
        cv2.drawMarker(
            annotated,
            (measurement.cx, measurement.cy),
            (0, 0, 255),
            cv2.MARKER_CROSS,
            14,
            1,
        )

        if measurement.suggested is None:
            cv2.putText(
                annotated,
                "NO HALO",
                (measurement.cx + 8, measurement.cy - 8),
                font,
                0.4,
                (0, 0, 255),
                1,
            )
            continue

        x, y, w, h = measurement.suggested
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 1)
        label = f"{prefix_id(number)} {w}x{h} V={measurement.best_v:.0f} S={measurement.best_s:.0f}"
        cv2.putText(
            annotated,
            label,
            (x, max(y - 5, 12)),
            font,
            0.4,
            (0, 255, 0),
            1,
        )

    return annotated


def prefix_id(number: int) -> str:
    """Идентификатор диода по порядковому номеру."""
    return f"LED_{number}"