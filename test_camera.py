import cv2
import time
from src.camera import USBCamera

def main():
    # Если на ноуте 0 — это встроенная вебка, то внешняя USB скорее всего будет 1
    CAM_INDEX = 1  # Измени на 1, если откроется не та камера

    cam = USBCamera(camera_id=CAM_INDEX, width=1280, height=720, fps=30)
    
    try:
        cam.start()
    except Exception as e:
        print(f"Ошибка: {e}")
        return

    prev_time = time.time()
    fps_counter = 0
    actual_fps = 0.0

    print("Нажмите 'q' для выхода, 'e' — переключить ручную/авто экспозицию")
    manual_exp = False

    while True:
        ret, frame = cam.read_frame()
        if not ret:
            print("Ошибка получения кадра")
            break

        # Расчет реального FPS
        fps_counter += 1
        now = time.time()
        if now - prev_time >= 1.0:
            actual_fps = fps_counter / (now - prev_time)
            fps_counter = 0
            prev_time = now

        # Отрисовка метрик
        cv2.putText(frame, f"FPS: {actual_fps:.1f}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Manual Exp: {manual_exp}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow("Test USB Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('e' or 'у'):
            manual_exp = not manual_exp
            if manual_exp:
                cam.set_manual_exposure(-6)
                print("Включена ручная экспозиция")
            else:
                # Включение авто-экспозиции обратно (3 = Auto)
                cam.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
                print("Включена авто-экспозиция")

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()