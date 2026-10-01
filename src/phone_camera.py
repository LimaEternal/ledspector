"""Захват основной (тыловой) камеры Android-телефона через scrcpy.

scrcpy в режиме ``--video-source=camera`` стримит камеру телефона в
отдельное окно (требуется Android 12+ и scrcpy 3.0+). Данный класс
автоматически запускает scrcpy с фиксированными параметрами окна,
находит окно по заголовку и захватывает его область через ``mss``.

За счёт ``--window-borderless`` + ``--render-fit=stretched`` +
фиксированного размера окна каждый кадр имеет одинаковые размеры
(по умолчанию 720x480), поэтому координаты ROI из ``settings.json``
остаются стабильными между сессиями.

Интерфейс совместим с ``USBCamera``: ``start()``, ``read_frame()``,
``release()``.

Требования:
    * scrcpy 3.0+ (в PATH или в папке проекта)
    * Android 12+ с включённой отладкой по USB (ADB), устройство авторизовано
"""

from __future__ import annotations

import ctypes
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import mss
import numpy as np

MIN_VERSION: Tuple[int, int] = (3, 0)


class _RECT(ctypes.Structure):
    _fields_ = [
        ("left", ctypes.c_long),
        ("top", ctypes.c_long),
        ("right", ctypes.c_long),
        ("bottom", ctypes.c_long),
    ]


class _POINT(ctypes.Structure):
    _fields_ = [
        ("x", ctypes.c_long),
        ("y", ctypes.c_long),
    ]


def _set_process_dpi_aware() -> None:
    """Делает процесс DPI-aware: координаты окон в физических пикселях.

    Без этого при масштабировании Windows >100% ``mss`` и координаты
    окна окажутся в разных системах координат.
    """
    if sys.platform != "win32":
        return
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except (AttributeError, OSError):
        pass


def _find_scrcpy() -> Optional[str]:
    """Ищет исполняемый файл scrcpy.

    План поиска:
    1. scrcpy из PATH (``shutil.which``);
    2. локальная копия рядом с проектом: ``<корень_проекта>/scrcpy.exe``,
       ``<корень_проекта>/scrcpy/scrcpy.exe`` или ``<корень_проекта>/scrcpy*/scrcpy.exe``.

    Returns:
        Абсолютный путь к ``scrcpy.exe`` или ``None``, если не найден.
    """
    found = shutil.which("scrcpy")
    if found:
        return os.path.abspath(found)

    project_root = Path(__file__).resolve().parent.parent
    for pattern in ("scrcpy.exe", "scrcpy/scrcpy.exe", "scrcpy*/scrcpy.exe"):
        for match in sorted(project_root.glob(pattern)):
            if match.is_file():
                return str(match)
    return None


def _scrcpy_version(scrcpy_path: Optional[str] = None) -> Optional[Tuple[int, int]]:
    """Определяет версию scrcpy по ``--version``.

    Args:
        scrcpy_path: Путь к scrcpy.exe (если найден локально).
            ``None`` — искать в PATH.

    Returns:
        Кортеж ``(major, minor)`` или ``None``, если определить не удалось.
    """
    binary = scrcpy_path or "scrcpy"
    try:
        result = subprocess.run(
            [binary, "--version"],
            capture_output=True, text=True, timeout=10,
        )
        text = result.stdout or result.stderr or ""
    except (OSError, subprocess.TimeoutExpired):
        return None

    match = re.search(r"(\d+)\.(\d+)", text)
    if match is None:
        return None
    return int(match.group(1)), int(match.group(2))


class ScrcpyPhoneCamera:
    """Камера Android-телефона, захватываемая из окна scrcpy.

    Attributes:
        width: Ширина кадра захвата (размер окна scrcpy).
        height: Высота кадра захвата.
        fps: Целевой FPS захвата.
        camera_id: Нормализованный идентификатор источника (``"scrcpy"``).
    """

    def __init__(
        self,
        window_title: str = "LEDPhoneCam",
        window_w: int = 720,
        window_h: int = 480,
        camera_facing: str = "back",
        camera_max_size: int = 720,
        auto_launch: bool = True,
        launch_timeout_sec: int = 15,
        device_serial: Optional[str] = None,
    ) -> None:
        """Инициализация камеры.

        Args:
            window_title: Заголовок окна scrcpy (по нему ищется окно).
            window_w: Ширина окна/кадра в пикселях.
            window_h: Высота окна/кадра.
            camera_facing: ``"back"`` (основная), ``"front"`` или ``"external"``.
            camera_max_size: Ограничение разрешения камеры (``-m``).
            auto_launch: Запускать ли scrcpy автоматически.
            launch_timeout_sec: Максимум ожидания появления окна.
            device_serial: Серийник устройства ADB (если подключено несколько).
        """
        self.window_title = window_title
        self.window_w = window_w
        self.window_h = window_h
        self.camera_facing = camera_facing
        self.camera_max_size = camera_max_size
        self.auto_launch = auto_launch
        self.launch_timeout_sec = launch_timeout_sec
        self.device_serial = device_serial

        self.width: int = window_w
        self.height: int = window_h
        self.fps: int = 30
        self.camera_id: str = "scrcpy"

        self._proc: Optional[subprocess.Popen] = None
        self._sct: Optional[mss.MSS] = None
        self._hwnd: Optional[int] = None
        self._scrcpy_path: Optional[str] = None
        self._started: bool = False

    # ------------------------------------------------------------------
    # Жизненный цикл
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Запускает scrcpy (при необходимости) и захватывает его окно."""
        if self._started:
            return

        self._scrcpy_path = _find_scrcpy()
        if self._scrcpy_path is None:
            raise RuntimeError(
                "scrcpy не найден. Добавьте его в PATH или положите копию "
                "(папку с scrcpy.exe) в корень проекта — она найдётся "
                "автоматически. Скачать: https://github.com/Genymobile/scrcpy"
            )

        if self.auto_launch:
            self._check_version()
            self._setup_user32()
            self._launch_scrcpy()
        else:
            self._setup_user32()

        _set_process_dpi_aware()
        self._sct = mss.MSS()
        self._hwnd = self._wait_for_window(self.launch_timeout_sec)

        if self._hwnd is None:
            self._cleanup()
            raise RuntimeError(
                f"Окно scrcpy '{self.window_title}' не появилось за "
                f"{self.launch_timeout_sec} сек. Проверьте, что телефон "
                "подключён по USB, включена отладка и устройство авторизовано "
                "(adb devices)."
            )

        self._started = True
        print(f"[PhoneCamera] Окно scrcpy '{self.window_title}' "
              f"подключено ({self.width}x{self.height})")

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Читает текущий кадр из окна scrcpy.

        Returns:
            Кортеж ``(ret, frame)``: ``ret=True`` и BGR-кадр, либо
            ``ret=False`` и ``None``.
        """
        if not self._started or self._sct is None or self._hwnd is None:
            return False, None

        region = self._get_client_rect(self._hwnd)
        if region is None:
            return False, None

        left, top, w, h = region
        if w <= 0 or h <= 0:
            return False, None

        try:
            shot = self._sct.grab({
                "left": left, "top": top, "width": w, "height": h,
            })
        except mss.exception.ScreenShotError:
            return False, None

        bgra = np.asarray(shot)
        if bgra.size == 0:
            return False, None

        # mss на Windows отдаёт BGRA -> конвертируем в BGR
        frame = cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
        return True, frame

    def release(self) -> None:
        """Останавливает захват и закрывает запущенный scrcpy."""
        self._started = False
        self._hwnd = None

        if self._sct is not None:
            self._sct.close()
            self._sct = None

        if self._proc is not None:
            if self._proc.poll() is None:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            self._proc = None
            print("[PhoneCamera] scrcpy остановлен")

    # ------------------------------------------------------------------
    # Внутренняя реализация
    # ------------------------------------------------------------------

    def _check_version(self) -> None:
        """Проверяет версию scrcpy (требуется 3.0+, т.к. camera-режим новее)."""
        version = _scrcpy_version(self._scrcpy_path)
        if version is None:
            print("[PhoneCamera] Не удалось определить версию scrcpy, "
                  "продолжаем (ожидается 3.0+)")
            return
        if version < MIN_VERSION:
            self._cleanup()
            raise RuntimeError(
                f"scrcpy версии {version[0]}.{version[1]} не поддерживает "
                "camera-режим (нужен 3.0+). Обновите scrcpy: "
                "https://github.com/Genymobile/scrcpy"
            )
        print(f"[PhoneCamera] scrcpy v{version[0]}.{version[1]}")

    def _setup_user32(self) -> None:
        """Декларирует типы Win32-функций для корректной работы на 64-bit."""
        if sys.platform != "win32":
            raise RuntimeError(
                "ScrcpyPhoneCamera на Windows только тестировалась; "
                f"текущая ОС: {sys.platform}"
            )
        user32 = ctypes.windll.user32
        user32.FindWindowW.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p]
        user32.FindWindowW.restype = ctypes.c_void_p
        user32.GetClientRect.argtypes = [ctypes.c_void_p, ctypes.POINTER(_RECT)]
        user32.GetClientRect.restype = ctypes.c_int
        user32.ClientToScreen.argtypes = [ctypes.c_void_p, ctypes.POINTER(_POINT)]
        user32.ClientToScreen.restype = ctypes.c_int

    def _launch_scrcpy(self) -> None:
        """Запускает scrcpy в режиме камеры с фиксированным окном."""
        cmd = [
            self._scrcpy_path,
            "--video-source=camera",
            f"--camera-facing={self.camera_facing}",
            "--no-audio",
        ]
        if self.camera_max_size:
            cmd.append(f"-m{self.camera_max_size}")
        cmd.extend([
            f"--window-title={self.window_title}",
            f"--window-width={self.window_w}",
            f"--window-height={self.window_h}",
            "--window-borderless",
            "--render-fit=stretched",
        ])
        if self.device_serial:
            cmd.append(f"--serial={self.device_serial}")

        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self._proc = subprocess.Popen(
            cmd,
            cwd=os.path.dirname(self._scrcpy_path),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            creationflags=flags,
        )
        print(f"[PhoneCamera] Запуск: {' '.join(cmd)}")

    def _wait_for_window(self, timeout_sec: int) -> Optional[int]:
        """Ждёт появления окна scrcpy, попутно отслеживая крах процесса."""
        user32 = ctypes.windll.user32
        deadline = time.monotonic() + timeout_sec

        while time.monotonic() < deadline:
            handle = user32.FindWindowW(None, self.window_title)
            if handle:
                return int(handle)

            if self._proc is not None and self._proc.poll() is not None:
                code = self._proc.returncode
                detail = ""
                if self._proc.stderr is not None:
                    detail = self._proc.stderr.read().decode(
                        errors="replace")[-500:]
                self._cleanup()
                raise RuntimeError(
                    f"scrcpy завершился с кодом {code}. {detail.strip()}"
                )
            time.sleep(0.25)

        return None

    def _get_client_rect(self, hwnd: int) -> Optional[Tuple[int, int, int, int]]:
        """Возвращает область клиентской зоны окна в экранных координатах."""
        user32 = ctypes.windll.user32
        rect = _RECT()
        if not user32.GetClientRect(hwnd, ctypes.byref(rect)):
            return None
        if rect.right <= 0 or rect.bottom <= 0:
            return None

        p1 = _POINT(0, 0)
        p2 = _POINT(rect.right, rect.bottom)
        if not user32.ClientToScreen(hwnd, ctypes.byref(p1)):
            return None
        if not user32.ClientToScreen(hwnd, ctypes.byref(p2)):
            return None

        return p1.x, p1.y, p2.x - p1.x, p2.y - p1.y

    def _cleanup(self) -> None:
        """Закрывает окно mss и убивает процесс scrcpy (best-effort)."""
        if self._sct is not None:
            try:
                self._sct.close()
            except Exception:
                pass
            self._sct = None
        if self._proc is not None:
            if self._proc.poll() is None:
                self._proc.kill()
            self._proc = None


if __name__ == "__main__":
    cam = ScrcpyPhoneCamera()
    try:
        cam.start()
    except RuntimeError as exc:
        print(f"Ошибка: {exc}")
        raise SystemExit(1) from exc

    print("Отображение камеры. Нажмите 'q' для выхода.")
    try:
        while True:
            ret, frame = cam.read_frame()
            if not ret:
                print("Не удалось получить кадр")
                break
            cv2.imshow("Phone camera (scrcpy)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cam.release()
        cv2.destroyAllWindows()