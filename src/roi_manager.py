"""Менеджер ROI-зон (Region of Interest) для выделения и хранения координат светодиодов."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


_DEFAULT_CAMERA: Dict[str, Any] = {
    "camera_id": 1,
    "width": 1280,
    "height": 720,
    "fps": 30,
}


class ROIManager:
    """Менеджер зон интереса (ROI) для визуального контроля светодиодов.

    Позволяет интерактивно выделять зоны на кадре, сохранять их
    в JSON-конфиг и загружать при повторном запуске.

    Attributes:
        config_path: Путь к файлу конфигурации.
        rois: Список словарей с полями ``id``, ``bbox``, ``description``.
        camera_config: Словарь параметров камеры.
    """

    def __init__(self, config_path: str = "config/settings.json") -> None:
        """Инициализация менеджера и загрузка конфигурации.

        Args:
            config_path: Путь к JSON-файлу конфигурации.
        """
        self.config_path: str = config_path
        self.rois: List[Dict[str, Any]] = []
        self.camera_config: Dict[str, Any] = dict(_DEFAULT_CAMERA)
        self.load_config()

    # ------------------------------------------------------------------
    # Конфигурация
    # ------------------------------------------------------------------

    def load_config(self) -> None:
        """Безопасно загружает конфигурацию из JSON-файла.

        Если файл отсутствует или повреждён — используются значения по
        умолчанию, а список ROI очищается.
        """
        path = Path(self.config_path)

        if not path.is_file():
            print(f"[ROIManager] Конфиг не найден: {path}")
            self.rois = []
            return

        try:
            with open(path, encoding="utf-8") as fh:
                data: Dict[str, Any] = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[ROIManager] Ошибка чтения конфига: {exc}")
            self.rois = []
            return

        self.camera_config = data.get("camera", dict(_DEFAULT_CAMERA))

        raw_rois: List[Dict[str, Any]] = data.get("rois", [])
        self.rois = []
        for entry in raw_rois:
            if not isinstance(entry, dict):
                continue
            roi_id = str(entry.get("id", ""))
            bbox = entry.get("bbox")
            description = str(entry.get("description", ""))
            if roi_id and isinstance(bbox, list) and len(bbox) == 4:
                self.rois.append({
                    "id": roi_id,
                    "bbox": [int(v) for v in bbox],
                    "description": description,
                })

        print(f"[ROIManager] Загружено {len(self.rois)} ROI из {path}")

    def save_config(self) -> None:
        """Сохраняет текущие ROI и параметры камеры в JSON-файл."""
        data: Dict[str, Any] = {
            "camera": self.camera_config,
            "rois": self.rois,
        }

        path = Path(self.config_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)

        print(f"[ROIManager] Конфиг сохранён: {path} ({len(self.rois)} ROI)")

    # ------------------------------------------------------------------
    # Интерактивное выделение
    # ------------------------------------------------------------------

    def clear_rois(self) -> None:
        """Очищает список ROI."""
        self.rois.clear()

    def select_rois_interactive(self, frame: np.ndarray) -> None:
        """Интерактивно выделяет ROI на кадре через ``cv2.selectROI``.

        Пользователь последовательно выделяет прямоугольные зоны.
        После каждой зоны запрашивается имя диода через ``input()``.
        Авто-имя: ``LED_1``, ``LED_2``, ...

        Завершение: нажмите **Enter** (пустой ввод) или **ESC** в диалоге
        выбора ROI.

        Args:
            frame: Текущий кадр с камеры (не изменяется).
        """
        self.clear_rois()
        counter = 0

        while True:
            counter += 1
            default_name = f"LED_{counter}"

            print(f"\n[ROIManager] Выделите зону #{counter} "
                  f"(Enter/ESC — завершить разметку)")

            roi_tuple: Tuple[int, int, int, int] = cv2.selectROI(
                "Select ROI", frame, showCrosshair=True, fromCenter=False,
            )
            x, y, w, h = roi_tuple

            # selectROI возвращает (0, 0, 0, 0) при отмене
            if w == 0 or h == 0:
                print("[ROIManager] Разметка завершена.")
                break

            raw_name = input(f"  Имя диода [{default_name}]: ").strip()
            roi_id = raw_name if raw_name else default_name

            description = input("  Описание (пусто — без описания): ").strip()

            self.rois.append({
                "id": roi_id,
                "bbox": [int(x), int(y), int(w), int(h)],
                "description": description,
            })

            print(f"  -> Добавлено: {roi_id} bbox=[{x}, {y}, {w}, {h}]")

    # ------------------------------------------------------------------
    # Отрисовка
    # ------------------------------------------------------------------

    def draw_rois(self, frame: np.ndarray) -> np.ndarray:
        """Наносит на кадр прямоугольники ROI, имена и перекрестия центров.

        Args:
            frame: Исходный кадр (не модифицируется).

        Returns:
            Копия кадра с отрисованными ROI.
        """
        annotated = frame.copy()

        for roi in self.rois:
            x, y, w, h = roi["bbox"]
            roi_id: str = roi["id"]
            cx, cy = x + w // 2, y + h // 2

            # Прямоугольник
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Перекрестие центра
            cross_size = 8
            cv2.line(annotated, (cx - cross_size, cy),
                     (cx + cross_size, cy), (0, 255, 255), 1)
            cv2.line(annotated, (cx, cy - cross_size),
                     (cx, cy + cross_size), (0, 255, 255), 1)

            # Подпись
            label_y = max(y - 6, 14)
            cv2.putText(annotated, roi_id, (x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return annotated

    # ------------------------------------------------------------------
    # Вырезание ROI
    # ------------------------------------------------------------------

    def get_cropped_rois(self, frame: np.ndarray) -> Dict[str, np.ndarray]:
        """Возвращает вырезанные зоны кадра в виде словаря ``{id: crop}``.

        Координаты зажимаются (clamping) к границам кадра для защиты
        от ``IndexError`` при выходе ROI за пределы изображения.

        Args:
            frame: Исходный кадр.

        Returns:
            Словарь ``{roi_id: numpy_array}`` для каждой ROI.
        """
        frame_h, frame_w = frame.shape[:2]
        crops: Dict[str, np.ndarray] = {}

        for roi in self.rois:
            x, y, w, h = roi["bbox"]
            roi_id: str = roi["id"]

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(frame_w, x + w)
            y2 = min(frame_h, y + h)

            if x2 <= x1 or y2 <= y1:
                crops[roi_id] = np.array([], dtype=frame.dtype)
                continue

            crops[roi_id] = frame[y1:y2, x1:x2].copy()

        return crops
