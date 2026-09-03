import cv2
import time

class USBCamera:
    def __init__(self, camera_id=0, width=1280, height=720, fps=30):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self.fps = fps
        self.cap = None

    def start(self):
        """Инициализация камеры с прямым бэкендом DirectShow."""
        self.cap = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        
        if not self.cap.isOpened():
            raise RuntimeError(f"Не удалось открыть USB-камеру с ID={self.camera_id}")

        # Устанавливаем параметры разрешения и кадров
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        # Прогрев матрицы
        time.sleep(1)
        for _ in range(5):
            self.cap.read()

        print(f"[Camera] Подключено: {int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ "
              f"{int(self.cap.get(cv2.CAP_PROP_FPS))} FPS")

    def set_manual_exposure(self, exposure_value=-5):
        """
        Отключение авто-экспозиции.
        Примечание: На разных дешевых вебках диапазон значения экспозиции может различаться
        (например, от -13 до 0 или от 1 до 100).
        """
        if self.cap and self.cap.isOpened():
            # 1 = Ручной режим (Manual Exposure) для большинства UVC-камер
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1) 
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure_value)

    def read_frame(self):
        if not self.cap or not self.cap.isOpened():
            return False, None
        return self.cap.read()

    def release(self):
        if self.cap and self.cap.isOpened():
            self.cap.release()
            print("[Camera] Захваченное устройство освобождено")