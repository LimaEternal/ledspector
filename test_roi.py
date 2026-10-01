"""Тестовый скрипт проверки ROI-менеджера.

Горячие клавиши:
    r — сбросить и переразметить зоны заново
    s — принудительно сохранить конфиг
    q — выход
"""

import cv2

from src.camera_factory import create_camera, current_source
from src.roi_manager import ROIManager

CONFIG_PATH = "config/settings.json"
WINDOW_NAME = "LED Inspector — ROI"


def run_selection(manager: ROIManager, frame: cv2.typing.MatLike) -> None:
    """Запускает интерактивную разметку и сохраняет результат."""
    manager.select_rois_interactive(frame)
    manager.save_config()
    print(f"[Test] Всего ROI: {len(manager.rois)}")


def main() -> None:
    manager = ROIManager(config_path=CONFIG_PATH)
    # Камера выбирается из настроек (webcam/scrcpy)
    cam = create_camera(manager.camera_config, current_source(manager.camera_config))

    try:
        cam.start()
    except RuntimeError as exc:
        print(f"Ошибка камеры: {exc}")
        return

    # Если конфиг пуст — сразу запускаем разметку
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

            annotated = manager.draw_rois(frame)
            cv2.imshow(WINDOW_NAME, annotated)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("r"):
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
