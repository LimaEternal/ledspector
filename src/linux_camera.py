"""Камера USB на Linux через бэкенд V4L2 + ручное управление экспозицией.

Отличие от `src/camera.py`: там жёстко зашит `cv2.CAP_DSHOW`, который
существует только под Windows. Здесь используется `cv2.CAP_V4L2` и
разрешение принудительно фиксируется — иначе камера отдаст своё
собственное разрешение по умолчанию (у типовых UVC-камер это 480x320),
и координаты LED из конфига будут указывать на другие пиксели.

Экспозиция управляется через `v4l2-ctl`: OpenCV на бэкенде V4L2
транслирует `CAP_PROP_EXPOSURE` не на всех драйверах, тогда как прямой
ioctl работает всегда. Результат каждой попытки читается обратно, чтобы
печатать фактическое значение, а не запрошенное.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple

import cv2

__all__ = ["LinuxCamera", "ExposureResult", "list_video_devices", "have_v4l2_ctl"]

# Для UVC-камер control auto_exposure: 1 = Aperture Priority Mode (авто),
# 3 = Manual Mode. Значения именно такие, как показывает `fswebcam
# --list-controls` в списке "Auto Exposure | Manual Mode | Aperture
# Priority Mode".
UVC_AUTO_EXPOSURE_MANUAL = 3
UVC_AUTO_EXPOSURE_APERTURE = 1

V4L2_CTL_MIN_ABSOLUTE = 80
V4L2_CTL_MAX_ABSOLUTE = 100000


def have_v4l2_ctl() -> bool:
    """Проверяет наличие утилиты v4l2-ctl (пакет v4l-utils)."""
    return shutil.which("v4l2-ctl") is not None


def list_video_devices() -> List[Path]:
    """Возвращает отсортированный список /dev/video* устройств."""
    return sorted(Path("/dev").glob("video*"))


class ExposureResult:
    """Результат попытки выставить экспозицию."""

    def __init__(
        self,
        ok: bool,
        applied: Optional[int],
        actual: Optional[int],
        method: str,
        detail: str = "",
    ) -> None:
        self.ok = ok
        self.applied = applied
        self.actual = actual
        self.method = method
        self.detail = detail

    def __repr__(self) -> str:
        return (
            f"ExposureResult(ok={self.ok}, applied={self.applied}, "
            f"actual={self.actual}, method={self.method!r}, detail={self.detail!r})"
        )


class LinuxCamera:
    """Обёртка над cv2.VideoCapture с бэкендом V4L2."""

    def __init__(
        self,
        device=0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
    ) -> None:
        # device принимает и индекс (0 -> /dev/video0), и путь строкой
        self.device_arg = device
        self.device = self._resolve_device(device)
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: Optional[cv2.VideoCapture] = None

        self.actual_width = 0
        self.actual_height = 0
        self.actual_fps = 0.0

        self._exposure_method: Optional[str] = None

    @staticmethod
    def _resolve_device(device) -> str:
        if isinstance(device, int):
            return f"/dev/video{device}"
        return str(device)

    # ------------------------------------------------------------------
    # Захват
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Открывает устройство, фиксирует разрешение и прогревает матрицу."""
        self.cap = cv2.VideoCapture(self.device, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(
                f"Не удалось открыть камеру {self.device}. "
                "Проверьте, что устройство существует, права на /dev/video* "
                "и что камера не занята другим процессом (fswebcam, guvc)."
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        time.sleep(1.0)
        for _ in range(10):
            self.cap.read()

        self._refresh_actual()

        print(
            f"[camera] {self.device}: запрошено {self.width}x{self.height}"
            f"@{self.fps}, получено {self.actual_width}x{self.actual_height}"
            f"@{self.actual_fps:.0f}"
        )
        if (self.actual_width, self.actual_height) != (self.width, self.height):
            print(
                f"[camera] ВНИМАНИЕ: камера не смогла выдать запрошенное "
                f"разрешение. Координаты из конфига ({self.width}x"
                f"{self.height}) применять нельзя — либо смените "
                "разрешение камеры, либо исправьте директиву resolution."
            )

    def _refresh_actual(self) -> None:
        if self.cap is None:
            return
        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = float(self.cap.get(cv2.CAP_PROP_FPS))

    def read_frame(self):
        """Возвращает (ok, frame). ok=False означает потерю кадра."""
        if self.cap is None or not self.cap.isOpened():
            return False, None
        ok, frame = self.cap.read()
        return bool(ok), frame

    def release(self) -> None:
        if self.cap is not None and self.cap.isOpened():
            self.cap.release()
            print("[camera] Устройство освобождено")
        self.cap = None

    def __enter__(self) -> "LinuxCamera":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()

    # ------------------------------------------------------------------
    # v4l2-ctl
    # ------------------------------------------------------------------
    def _run_v4l2(self, args: List[str], timeout: float = 5.0):
        if not have_v4l2_ctl():
            return None
        cmd = ["v4l2-ctl", "-d", self.device] + args
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except (subprocess.TimeoutExpired, OSError) as exc:
            return None
        return proc

    def list_controls(self) -> str:
        """Возвращает вывод `v4l2-ctl --list-ctrls` (или '' если нет утилиты)."""
        proc = self._run_v4l2(["--list-ctrls"], timeout=10.0)
        if proc is None:
            return ""
        return (proc.stdout or "") + (proc.stderr or "")

    def get_exposure(self) -> Optional[int]:
        """Читает фактическое значение exposure_absolute."""
        proc = self._run_v4l2(["--get-ctrl=exposure_absolute"])
        if proc is None or proc.returncode != 0:
            return None
        for line in (proc.stdout or "").splitlines():
            if ":" not in line:
                continue
            _, _, value = line.partition(":")
            try:
                return int(value.strip())
            except ValueError:
                continue
        return None

    def get_auto_exposure(self) -> Optional[str]:
        """Читает текущий режим auto_exposure ('manual' / 'aperture')."""
        proc = self._run_v4l2(["--get-ctrl=auto_exposure"])
        if proc is None or proc.returncode != 0:
            return None
        text = (proc.stdout or "").lower()
        if "manual" in text:
            return "manual"
        if "aperture" in text:
            return "aperture"
        return None

    # ------------------------------------------------------------------
    # Экспозиция
    # ------------------------------------------------------------------
    def set_exposure(self, value: int, verbose: bool = True) -> ExposureResult:
        """Выставляет exposure_absolute, переводя камеру в ручной режим.

        Порядок обязателен: сначала auto_exposure=3, иначе камера тут же
        перезапишет exposure_absolute своим значением.
        """
        value = int(value)

        if self._set_via_v4l2(value, verbose):
            self._exposure_method = "v4l2-ctl"
            return ExposureResult(True, value, self.get_exposure(), "v4l2-ctl")

        if self._set_via_opencv(value, verbose):
            self._exposure_method = "opencv"
            actual = self._exposure_from_cap()
            return ExposureResult(True, value, actual, "opencv")

        detail = (
            "Не удалось выставить экспозицию ни через v4l2-ctl, ни через "
            "свойства OpenCV. Установите пакет v4l-utils "
            "(apt install v4l-utils) и запустите под root либо с правами "
            "на /dev/video*."
        )
        if verbose:
            print(f"[camera] {detail}")
        return ExposureResult(False, value, self.get_exposure(), "none", detail)

    def _set_via_v4l2(self, value: int, verbose: bool) -> bool:
        proc = self._run_v4l2(
            [
                f"--set-ctrl=auto_exposure={UVC_AUTO_EXPOSURE_MANUAL}",
                f"--set-ctrl=exposure_absolute={value}",
            ]
        )
        if proc is None or proc.returncode != 0:
            return False

        # Подтверждаем: камера могла проигнорировать запрос или округлить.
        actual = self.get_exposure()
        if actual is None:
            return True
        if actual != value:
            if verbose:
                print(
                    f"[camera] v4l2-ctl: запрошено {value}, камера вернула "
                    f"{actual} — пробую другой путь"
                )
            return False
        return True

    def _set_via_opencv(self, value: int, verbose: bool) -> bool:
        if self.cap is None or not self.cap.isOpened():
            return False
        try:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, float(value))
        except cv2.error:
            return False
        actual = self._exposure_from_cap()
        if verbose:
            print(f"[camera] OpenCV: запрошено {value}, получено {actual}")
        return actual is not None and abs(float(actual) - float(value)) <= 1.0

    def _exposure_from_cap(self) -> Optional[float]:
        if self.cap is None or not self.cap.isOpened():
            return None
        try:
            value = float(self.cap.get(cv2.CAP_PROP_EXPOSURE))
        except cv2.error:
            return None
        return None if value < 0 else value

    def get_exposure_method(self) -> Optional[str]:
        return self._exposure_method

    def enable_auto_exposure(self) -> bool:
        """Возвращает камеру в автоматический режим экспозиции."""
        proc = self._run_v4l2(
            [f"--set-ctrl=auto_exposure={UVC_AUTO_EXPOSURE_APERTURE}"]
        )
        ok = proc is not None and proc.returncode == 0
        if ok:
            print("[camera] Экспозиция: автоматический режим")
        else:
            print("[camera] Не удалось вернуть автоматическую экспозицию")
        return ok

    @staticmethod
    def clamp_exposure(value: int) -> int:
        return max(V4L2_CTL_MIN_ABSOLUTE, min(V4L2_CTL_MAX_ABSOLUTE, int(value)))

    def describe(self) -> str:
        return (
            f"{self.device} {self.actual_width}x{self.actual_height}"
            f"@{self.actual_fps:.0f} "
            f"exposure={self.get_exposure()} mode={self.get_auto_exposure()} "
            f"method={self._exposure_method}"
        )


def open_camera(
    device=0,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
) -> Tuple[LinuxCamera, bool]:
    """Открывает камеру, подбирая первое доступное устройство при device=None."""
    if device is not None:
        cam = LinuxCamera(device=device, width=width, height=height, fps=fps)
        cam.start()
        return cam, True

    devices = list_video_devices()
    if not devices:
        raise RuntimeError("Устройства /dev/video* не найдены — камера не подключена")

    last_error: Optional[Exception] = None
    for candidate in devices:
        cam = LinuxCamera(device=candidate, width=width, height=height, fps=fps)
        try:
            cam.start()
            return cam, True
        except RuntimeError as exc:
            last_error = exc
            print(f"[camera] {candidate}: не открывается, пробую следующее")
    raise RuntimeError(f"Ни одно устройство не открылось: {last_error}")