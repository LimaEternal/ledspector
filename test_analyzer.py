"""Интеграционный скрипт: детекция цвета + анализ частоты мигания.

Горячие клавиши:
    r — сбросить и переразметить зоны заново
    s — принудительно сохранить конфиг
    q — выход
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Tuple

import cv2

from src.camera import USBCamera
from src.color_detector import ColorDetector
from src.frequency_analyser import FrequencyAnalyser
from src.roi_manager import ROIManager

CONFIG_PATH = "config/settings.json"
CAMERA_ID = 1
WINDOW_NAME = "LED Inspector — Analyzer"

# Палитра для отрисовки статуса
_COLORS_BGR: Dict[str, Tuple[int, int, int]] = {
    "OFF":        (80, 80, 80),
    "SOLID_ON":   (0, 200, 0),
    "BLINK_1HZ":  (0, 200, 255),
    "BLINK_4HZ":  (0, 100, 255),
    "UNKNOWN":    (0, 100, 255),
    "CALCULATING": (200, 200, 0),
    "NO_DATA":    (128, 128, 128),
}


def _draw_info_panel(
    frame: np.ndarray,
    roi_id: str,
    state: str,
    freq_hz: float,
    color_name: str,
    y_offset: int,
) -> int:
    """Рисует одну строку информационной панели.

    Args:
        frame: Кадр для отрисовки (модифицируется).
        roi_id: Имя диода.
        state: Состояние мигания.
        freq_hz: Частота мигания.
        color_name: Определённый цвет.
        y_offset: Текущая Y-координата строки.

    Returns:
        Новый ``y_offset`` после этой строки.
    """
    state_color = _COLORS_BGR.get(state, (200, 200, 200))

    if freq_hz > 0:
        text = f"[{roi_id}] {state} ({freq_hz:.1f} Hz) | {color_name}"
    else:
        text = f"[{roi_id}] {state} | {color_name}"

    cv2.putText(frame, text, (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, state_color, 2)

    return y_offset + 28


def run_selection(manager: ROIManager, frame: cv2.typing.MatLike) -> None:
    """Запускает интерактивную разметку и сохраняет результат."""
    manager.select_rois_interactive(frame)
    manager.save_config()
    print(f"[Test] Всего ROI: {len(manager.rois)}")


def main() -> None:
    cam = USBCamera(camera_id=CAMERA_ID, width=1280, height=720, fps=30)
    manager = ROIManager(config_path=CONFIG_PATH)
    detector = ColorDetector()
    analyser = FrequencyAnalyser(window_seconds=2.0)

    try:
        cam.start()
    except RuntimeError as exc:
        print(f"Ошибка камеры: {exc}")
        return

    if not manager.rois:
        ret, frame = cam.read_frame()
        if not ret:
            print("Не удалось получить кадр для разметки")
            cam.release()
            return
        run_selection(manager, frame)

    print("\nГорячие клавиши: r — разметка, s — сохранить, q — выход\n")

    try:
        while True:
            ret, frame = cam.read_frame()
            if not ret:
                print("Ошибка получения кадра")
                break

            crops = manager.get_cropped_rois(frame)
            now = time.time()

            # Анализ каждого ROI
            roi_states: List[Tuple[str, str, float, str]] = []
            for roi_id, crop in crops.items():
                color_result = detector.detect(crop)
                analyser.update(roi_id, color_result["brightness"], now)
                state_result = analyser.analyze_state(roi_id)

                roi_states.append((
                    roi_id,
                    state_result["state"],
                    state_result["frequency_hz"],
                    color_result["color"],
                ))

            # Отрисовка рамок ROI
            annotated = manager.draw_rois(frame)

            # Инфо-панель
            y = 28
            for roi_id, state, freq, color in roi_states:
                y = _draw_info_panel(annotated, roi_id, state, freq, color, y)

            cv2.imshow(WINDOW_NAME, annotated)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("r"):
                analyser.reset()
                ret_r, frame_r = cam.read_frame()
                if ret_r:
                    run_selection(manager, frame_r)
            elif key == ord("s"):
                manager.save_config()
    finally:
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
