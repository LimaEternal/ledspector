"""Камера USB на Linux через бэкенд V4L2 + ручное управление экспозицией.

Отличие от `src/camera.py`: там жёстко зашит `cv2.CAP_DSHOW`, который
существует только под Windows. Здесь используется `cv2.CAP_V4L2` и
разрешение принудительно фиксируется — иначе камера отдаст своё
собственное разрешение по умолчанию (у типовых UVC-камер это 480x320),
и координаты LED из конфига будут указывать на другие пиксели.

Формат задаётся явно через `CAP_PROP_FOURCC` и выставляется ПЕРЕД
разрешением. Это не косметика: 1280x720 у UVC-камер бывает только в
MJPG, а в несжатом YUYV максимум — 640x480@10. Без fourcc драйвер
выбирает формат по умолчанию и тихо понижает разрешение, из-за чего
координаты LED уезжают на другие пиксели.

Экспозиция управляется через `v4l2-ctl`: OpenCV на бэкенде V4L2
транслирует `CAP_PROP_EXPOSURE` не на всех драйверах, тогда как прямой
ioctl работает всегда. Имя контрола берётся из реального вывода
`--list-ctrls` (у этой камеры `exposure_time_absolute`), а значение
ручного режима `auto_exposure` определяется по текстовой подписи, потому
что нумерация меню у камер различается. Результат каждой попытки читается
обратно, чтобы печатать фактическое значение, а не запрошенное.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

__all__ = [
    "LinuxCamera",
    "ExposureResult",
    "list_video_devices",
    "have_v4l2_ctl",
    "parse_formats_ext",
    "resolve_auto_exposure_values",
    "DEFAULT_FOURCC",
]

# Имя контрола экспозиции у UVC-камер. В выводе `v4l2-ctl --list-ctrls`
# он выглядит как exposure_time_absolute (0x009a0902). Короткое имя
# exposure_absolute из спецификации UVC драйвер не всегда принимает.
EXPOSURE_CONTROL = "exposure_time_absolute"

DEFAULT_FOURCC = "MJPG"

V4L2_CTL_MIN_ABSOLUTE = 80
V4L2_CTL_MAX_ABSOLUTE = 100000

# Запасные значения для auto_exposure, если разбор --list-ctrls не дал
# результата. У разных камер нумерация меню различается, поэтому это лишь
# последний рубеж, а не основной путь.
FALLBACK_AUTO_MANUAL = 1
FALLBACK_AUTO_APERTURE = 3


def have_v4l2_ctl() -> bool:
    """Проверяет наличие утилиты v4l2-ctl (пакет v4l-utils)."""
    return shutil.which("v4l2-ctl") is not None


def list_video_devices() -> List[Path]:
    """Возвращает отсортированный список /dev/video* устройств."""
    return sorted(Path("/dev").glob("video*"))


def parse_formats_ext(text: str) -> Dict[str, List[Tuple[int, int, float]]]:
    """Разбирает вывод `v4l2-ctl --list-formats-ext`.

    Возвращает словарь {формат: [(ширина, высота, fps), ...]}. Нужен, чтобы
    до подключения камеры знать, какие разрешения она вообще умеет: без
    заданного fourcc драйвер молча отдаёт наибольший размер того формата,
    который OpenCV выбрал по умолчанию, и запрошенное разрешение просто
    игнорируется.
    """
    formats: Dict[str, List[Tuple[int, int, float]]] = {}
    current: Optional[str] = None

    for raw_line in (text or "").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("ioctl") or line.startswith("Type:"):
            continue

        header = re.match(r"\[\d+\]:\s*'(\w+)'\s*\((.*)\)", line)
        if header:
            current = header.group(1)
            formats.setdefault(current, [])
            continue

        if current is None:
            continue

        size = re.search(r"Size:\s*Discrete\s*(\d+)x(\d+)", line)
        if size:
            formats[current].append((int(size.group(1)), int(size.group(2)), 0.0))
            continue

        interval = re.search(r"Interval:\s*Discrete\s*([\d.]+)s", line)
        if interval and formats[current]:
            seconds = float(interval.group(1))
            fps = round(1.0 / seconds, 1) if seconds > 0 else 0.0
            last = formats[current][-1]
            formats[current][-1] = (last[0], last[1], fps)

    return {name: sizes for name, sizes in formats.items() if sizes}


def resolve_auto_exposure_values(ctrls_text: str) -> Dict[str, Optional[int]]:
    """Определяет, какое значение auto_exposure включает ручной режим.

    У камер различается и нумерация меню, и подписи: у этой камеры
    `value=1 (Manual Mode)`, но встречается и обратная раскладка. Поэтому
    значение ищется по текстовой подписи, а не берётся из спецификации UVC.

    Подпись берётся из скобок сразу после `value=N`, а не из первых скобок
    в строке: первые обычно содержат тип (`(menu)`), а нужная — в конце.

    Возвращает {'manual': N, 'auto': M, 'current': текущее значение}.
    Любое из полей может быть None, если разбор не удался.
    """
    result: Dict[str, Optional[int]] = {
        "manual": None,
        "auto": None,
        "current": None,
    }

    for raw_line in (ctrls_text or "").splitlines():
        line = raw_line.strip()
        if "auto_exposure" not in line:
            continue

        value = re.search(r"\bvalue=(-?\d+)", line)
        if value:
            result["current"] = int(value.group(1))

        # Подпись режима идёт следом за значением: "value=1 (Manual Mode)".
        labelled = re.search(r"\bvalue=(-?\d+)\s*\(([^)]*)\)", line)
        if labelled:
            current = int(labelled.group(1))
            text = labelled.group(2).strip().lower()
            if "manual" in text:
                result["manual"] = current
            elif "aperture" in text or "auto" in text:
                result["auto"] = current

    return result


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
        fourcc: str = DEFAULT_FOURCC,
    ) -> None:
        # device принимает и индекс (0 -> /dev/video0), и путь строкой
        self.device_arg = device
        self.device = self._resolve_device(device)
        self.width = width
        self.height = height
        self.fps = fps
        self.fourcc = fourcc
        self.cap: Optional[cv2.VideoCapture] = None

        self.actual_width = 0
        self.actual_height = 0
        self.actual_fps = 0.0
        self.actual_fourcc = ""

        self._exposure_method: Optional[str] = None
        self._auto_exposure: Dict[str, Optional[int]] = {
            "manual": None,
            "auto": None,
            "current": None,
        }

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

        # Порядок важен: сначала fourcc, иначе драйвер успеет выбрать
        # формат по умолчанию (обычно YUYV) и разрешение 1280x720 просто
        # не найдётся — камера отдаст наибольший размер несжатого формата.
        if self.fourcc:
            code = cv2.VideoWriter_fourcc(*self.fourcc[:4].upper())
            self.cap.set(cv2.CAP_PROP_FOURCC, code)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

        time.sleep(1.0)
        for _ in range(10):
            self.cap.read()

        self._refresh_actual()

        print(
            f"[camera] {self.device}: запрошено {self.fourcc} "
            f"{self.width}x{self.height}@{self.fps}, получено "
            f"{self.actual_fourcc} {self.actual_width}x{self.actual_height}"
            f"@{self.actual_fps:.0f}"
        )
        if (self.actual_width, self.actual_height) != (self.width, self.height):
            print(
                f"[camera] ВНИМАНИЕ: камера не смогла выдать запрошенное "
                f"разрешение. Координаты из конфига ({self.width}x"
                f"{self.height}) применять нельзя — либо смените "
                "разрешение камеры, либо исправьте директиву resolution.\n"
                f"[camera] Поддерживаемые форматы: {self.list_formats_ext() or 'неизвестно'}"
            )

    def _refresh_actual(self) -> None:
        if self.cap is None:
            return
        self.actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.actual_fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        code = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        if code > 0:
            packed = "".join(
                chr((code >> (8 * i)) & 0xFF) for i in range(4)
            )
            self.actual_fourcc = packed.strip("\x00").strip()

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

    def list_formats_ext(self) -> str:
        """Возвращает разобранные поддерживаемые форматы и разрешения."""
        proc = self._run_v4l2(["--list-formats-ext"], timeout=10.0)
        if proc is None:
            return ""
        text = (proc.stdout or "") + (proc.stderr or "")
        parsed = parse_formats_ext(text)
        if not parsed:
            return ""
        parts = []
        for name in sorted(parsed):
            sizes = ", ".join(
                f"{w}x{h}@{fps:.0f}" if fps else f"{w}x{h}"
                for w, h, fps in parsed[name]
            )
            parts.append(f"{name}: {sizes}")
        return " | ".join(parts)

    def supports(self, width: int, height: int, fourcc: str = DEFAULT_FOURCC) -> bool:
        """Проверяет по списку форматов, умеет ли камера нужный режим."""
        proc = self._run_v4l2(["--list-formats-ext"], timeout=10.0)
        if proc is None:
            return False
        parsed = parse_formats_ext((proc.stdout or "") + (proc.stderr or ""))
        target = fourcc[:4].upper()
        return any(
            (width, height) in [(w, h) for w, h, _ in sizes]
            for name, sizes in parsed.items()
            if name.upper() == target
        )

    def get_exposure(self) -> Optional[int]:
        """Читает фактическое значение экспозиции."""
        proc = self._run_v4l2([f"--get-ctrl={EXPOSURE_CONTROL}"])
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

    def resolve_auto_exposure(self) -> Dict[str, Optional[int]]:
        """Определяет значения auto_exposure по списку контролов камеры."""
        resolved = resolve_auto_exposure_values(self.list_controls())
        if resolved["manual"] is None:
            resolved["manual"] = FALLBACK_AUTO_MANUAL
        if resolved["auto"] is None:
            resolved["auto"] = FALLBACK_AUTO_APERTURE
        self._auto_exposure = resolved
        return resolved

    # ------------------------------------------------------------------
    # Экспозиция
    # ------------------------------------------------------------------
    def set_exposure(self, value: int, verbose: bool = True) -> ExposureResult:
        """Выставляет экспозицию, переводя камеру в ручной режим.

        Порядок обязателен: сначала auto_exposure в Manual Mode, иначе
        камера тут же перезапишет экспозицию своим значением.
        """
        value = int(value)
        if self._auto_exposure.get("manual") is None:
            self.resolve_auto_exposure()

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
        manual = self._auto_exposure.get("manual") or FALLBACK_AUTO_MANUAL
        proc = self._run_v4l2(
            [
                f"--set-ctrl=auto_exposure={manual}",
                f"--set-ctrl={EXPOSURE_CONTROL}={value}",
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
        if self._auto_exposure.get("auto") is None:
            self.resolve_auto_exposure()
        aperture = self._auto_exposure.get("auto") or FALLBACK_AUTO_APERTURE
        proc = self._run_v4l2([f"--set-ctrl=auto_exposure={aperture}"])
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
            f"{self.device} {self.actual_fourcc or self.fourcc} "
            f"{self.actual_width}x{self.actual_height}@{self.actual_fps:.0f} "
            f"exposure={self.get_exposure()} mode={self.get_auto_exposure()} "
            f"method={self._exposure_method}"
        )


def open_camera(
    device=0,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    fourcc: str = DEFAULT_FOURCC,
) -> Tuple[LinuxCamera, bool]:
    """Открывает камеру, подбирая первое доступное устройство при device=None."""
    if device is not None:
        cam = LinuxCamera(
            device=device,
            width=width,
            height=height,
            fps=fps,
            fourcc=fourcc,
        )
        cam.start()
        return cam, True

    devices = list_video_devices()
    if not devices:
        raise RuntimeError("Устройства /dev/video* не найдены — камера не подключена")

    last_error: Optional[Exception] = None
    for candidate in devices:
        cam = LinuxCamera(
            device=candidate,
            width=width,
            height=height,
            fps=fps,
            fourcc=fourcc,
        )
        try:
            cam.start()
            return cam, True
        except RuntimeError as exc:
            last_error = exc
            print(f"[camera] {candidate}: не открывается, пробую следующее")
    raise RuntimeError(f"Ни одно устройство не открылось: {last_error}")