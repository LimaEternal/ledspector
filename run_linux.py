#!/usr/bin/env python3
"""Headless-инструменты для настройки и наблюдения LED на Linux-сервере.

Команды:
- check     — самопроверка окружения (Python, OpenCV, /dev/video*, v4l2-ctl)
- grid      — снять кадр с камерой, сохранить snapshot_raw.png + snapshot_grid.png
- exposure  — интерактивный подбор exposure_absolute
- monitor   — таблица в реальном времени (перерисовка)
- log       — только изменения состояний (append-only)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Sequence

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import cv2

from src.color_detector import ColorDetector
from src.console_monitor import Monitor, crop_led
from src.grid_overlay import draw_grid, draw_led_boxes
from src.led_analyzer import (
    analyze_frame,
    draw_measurements,
    format_config_lines,
    prefix_id,
)
from src.led_config import LedConfig, DEFAULT_CONFIG_PATH, load_config
from src.linux_camera import (
    DEFAULT_FOURCC,
    EXPOSURE_CONTROL,
    LinuxCamera,
    have_v4l2_ctl,
    list_video_devices,
    open_camera,
)


def cmd_check(_: argparse.Namespace) -> None:
    print("=== LEDSpector Linux Check ===")
    print(f"Python {sys.version.split()[0]}")
    print(f"cwd: {os.getcwd()}")
    print(f"opencv: {cv2.__version__}")
    try:
        import numpy as np

        print(f"numpy: {np.__version__}")
    except Exception:
        print("numpy: НЕТ")

    print("\n--- /dev/video* ---")
    devices = list_video_devices()
    if not devices:
        print("не найдено (камера не подключена или udev не даёт доступ)")
    else:
        for d in devices:
            print(d)

    print("\n--- v4l2-ctl ---")
    if have_v4l2_ctl():
        print("найден")
    else:
        print("НЕ НАЙДЕН (пакет v4l-utils)")

    print("\n--- поддерживаемые форматы ---")
    try:
        fmts = LinuxCamera(width=1280, height=720).list_formats_ext()
        print(fmts or "неизвестно (v4l2-ctl недоступен?)")
        cam_probe = LinuxCamera(width=1280, height=720)
        print(
            f"1280x720 в MJPG: "
            f"{'да' if cam_probe.supports(1280, 720, 'MJPG') else 'нет'}; "
            f"в YUYV: "
            f"{'да' if cam_probe.supports(1280, 720, 'YUYV') else 'нет'}"
        )
    except Exception as exc:
        print(f"ОШИБКА: {type(exc).__name__}: {exc}")

    print("\n--- попытка открыть камеру /dev/video0 (MJPG 1280x720) ---")
    try:
        cam = LinuxCamera(width=1280, height=720, fourcc=DEFAULT_FOURCC)
        cam.start()
        print(cam.describe())
        print("режимы auto_exposure:", cam.resolve_auto_exposure())
        print("controls:")
        print(cam.list_controls() or "недоступны")
        cam.release()
        print("\nOK: камера открывается и даёт кадры")
    except Exception as exc:
        print(f"ОШИБКА: {type(exc).__name__}: {exc}")


def cmd_grid(args: argparse.Namespace) -> None:
    print("=== LEDSpector Grid ===")
    led_config: Optional[LedConfig] = None
    try:
        led_config = load_config(args.config)
    except Exception as exc:
        print(f"Конфиг {args.config} не загружен ({exc}). Рисуем только сетку.")

    cam, _ = open_camera(
        device=args.device,
        width=1280,
        height=720,
        fps=args.fps,
        fourcc=args.fourcc,
    )
    try:
        cam.set_exposure(args.exposure)
        print("Снимаем один кадр...")
        ok, frame = cam.read_frame()
        if not ok or frame is None:
            raise RuntimeError("Не удалось получить кадр")
        print(f"raw: {frame.shape[1]}x{frame.shape[0]}")

        if led_config is not None:
            try:
                led_config.check_resolution(frame.shape[1], frame.shape[0])
            except ValueError as exc:
                print(f"ВНИМАНИЕ: {exc}")
                print("Сетка сохранена, но координаты из конфига к этому кадру не применимы.")

        base = args.out[:-4] if args.out.endswith(".png") else args.out
        raw_path = f"{base}_raw.png"
        grid_path = f"{base}_grid.png"

        cv2.imwrite(raw_path, frame)
        print(f"Сохранён RAW: {raw_path}")

        grid = draw_grid(frame, minor_step=args.minor_step, major_step=args.major_step)
        if led_config:
            grid = draw_led_boxes(grid, led_config.leds)
        cv2.imwrite(grid_path, grid)
        print(f"Сохранён GRID: {grid_path}")
    finally:
        cam.release()


def cmd_exposure(args: argparse.Namespace) -> None:
    print("=== LEDSpector Exposure ===")
    led_config: Optional[LedConfig] = None
    try:
        led_config = load_config(args.config)
    except Exception as exc:
        print(f"Внимание: не удалось загрузить конфиг {args.config}: {exc}")
        led_config = None

    cam, _ = open_camera(
        device=args.device,
        width=1280,
        height=720,
        fps=args.fps,
        fourcc=args.fourcc,
    )
    try:
        current = cam.get_exposure() or 500
        while True:
            val = input(f"exposure_absolute [{current}] (q чтобы выйти, a авто): ").strip()
            if not val:
                val = str(current)
            if val.lower() in ("q", "exit", "quit"):
                break
            if val.lower() in ("a", "auto"):
                cam.enable_auto_exposure()
                current = cam.get_exposure() or current
                continue
            try:
                exp = cam.clamp_exposure(int(val))
            except ValueError:
                print("Введите число")
                continue
            res = cam.set_exposure(exp)
            current = res.actual if res.actual is not None else exp
            print(f"ok={res.ok} method={res.method} applied={res.applied} actual={current}")
            print(cam.describe())

            if led_config:
                ok, frame = cam.read_frame()
                if ok and frame is not None:
                    det = ColorDetector()
                    for led in led_config.leds:
                        crop = crop_led(frame, led)
                        if crop.size == 0:
                            print(f"  {led.id}: вне кадра")
                            continue
                        r = det.detect(crop)
                        print(f"  {led.id}: V={r['brightness']:.1f} color={r['color']} on={r['is_on']}")
            print()
    finally:
        cam.release()


def _install_snapshot_signal(
    mon: Monitor, path: str, with_grid: bool = True
) -> None:
    """Вешает обработчик сигнала для снимка кадра по требованию.

    Ctrl+\\ (SIGQUIT) или `kill -QUIT <pid>` сохраняют текущий кадр с
    разметкой — удобно, когда нужно посмотреть, что именно камера видит,
    не останавливая наблюдение.
    """
    import signal

    def _handler(_signum, _frame):
        # Запись файла в обработчике сигнала небезопасна, поэтому только
        # ставим флаг, а сам снимок делаем в основном потоке.
        mon._snapshot_requested = path
        mon._snapshot_with_grid = with_grid

    try:
        signal.signal(signal.SIGQUIT, _handler)
    except (AttributeError, ValueError):
        pass


def _parse_set_ctrl(values: Optional[Sequence[str]]) -> Dict[str, int]:
    """Разбирает --set-ctrl в словарь {контрол: значение}.

    Принимается и перечисление через запятую (`gain=1,saturation=256`),
    и повторение флага — формат v4l2-ctl тот же, чтобы не путаться.
    """
    settings: Dict[str, int] = {}
    for chunk in values or ():
        for pair in chunk.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                print(f"[run_linux] Пропущено '{pair}': ожидается имя=значение")
                continue
            name, _, raw = pair.partition("=")
            name = name.strip().lower()
            try:
                settings[name] = int(raw.strip())
            except ValueError:
                # Меню-контролы принимают и подписи, а не только числа.
                settings[name] = raw.strip()  # type: ignore[assignment]
    return settings


def _apply_startup_controls(cam: LinuxCamera, args: argparse.Namespace) -> None:
    """Применяет --exposure и --set-ctrl, затем проверяет состояние камеры."""
    settings = _parse_set_ctrl(getattr(args, "set_ctrl", None))

    exposure = getattr(args, "exposure", None)
    if exposure is not None:
        if settings:
            settings.setdefault(EXPOSURE_CONTROL, int(exposure))
        else:
            cam.set_exposure(int(exposure))

    if settings:
        cam.apply_controls(settings)

    cam.warn_degraded_controls()


def cmd_analyze(args: argparse.Namespace) -> None:
    """Разбирает готовый снимок: находит диоды и подбирает точки замера."""
    print("=== LEDSpector Analyze ===")

    if args.image:
        frame = cv2.imread(args.image)
        if frame is None:
            raise SystemExit(f"Не удалось открыть изображение: {args.image}")
        print(f"Кадр: {args.image} ({frame.shape[1]}x{frame.shape[0]})")
    else:
        cam, _ = open_camera(
            device=args.device,
            width=1280,
            height=720,
            fps=args.fps,
            fourcc=args.fourcc,
        )
        try:
            _apply_startup_controls(cam, args)
            ok, frame = cam.read_frame()
            if not ok or frame is None:
                raise SystemExit("Не удалось получить кадр с камеры")
        finally:
            cam.release()

    detector = ColorDetector()
    measurements = analyze_frame(
        frame,
        detector=detector,
        window=args.window_size,
    )

    if not measurements:
        print("Светящихся пятен не найдено. Проверь экспозицию и gain.")
        return

    print()
    print(
        f"{'ID':<8}{'центр':<12}{'вердикт':<10}{'центр V/S':<12}"
        f"{'пересвет':<10}новая точка"
    )
    print("-" * 74)

    movable = 0
    for number, m in enumerate(measurements, start=1):
        if m.suggested is None:
            spot = "нет годной точки"
        else:
            x, y, w, h = m.suggested
            spot = f"({x},{y}) {w}x{h} r={m.suggested_radius}"
            if m.status == "MOVE":
                movable += 1

        print(
            f"{prefix_id(number):<8}({m.cx},{m.cy})".ljust(20)
            + f"{m.status:<10}"
            + f"{m.center_v:.0f}/{m.center_s:.0f}".ljust(12)
            + f"{100 * m.center_clipped:.0f}%".ljust(10)
            + spot
        )

    print()
    if movable:
        print(f"Диодов с пересветом в центре: {movable} из {len(measurements)}.")
        print(
            "Пересвет в центре означает, что цвет там недоступен физически, "
            "и понижение экспозиции не помогает: приёмник суммирует поток "
            "от соседних диодов. Помогает замер на краю ореола — точки "
            "подобраны ниже."
        )
    else:
        print("Пересвета в центре нет: центры диодов можно замерять напрямую.")

    if args.out:
        annotated = draw_measurements(frame, measurements)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(args.out, annotated)
        print(f"\nРазметка сохранена: {args.out}")

    lines = format_config_lines(measurements, window_size=args.window_size)
    if lines:
        print("\nГотовый блок для config/leds.tsv:")
        for line in lines:
            print(f"  {line}")
        print(
            "\nЗамени содержимое блока ID..(последняя строка) в конфиге этими "
            "строками. Диоды со статусом «нет годной точки» в блок не попадают."
        )


def cmd_find(args: argparse.Namespace) -> None:
    """Короткий вариант analyze: только координаты новых точек замера."""
    # Разметка не сохраняется — команда существует ради блока координат,
    # поэтому лишний артефакт на диске просто не нужен.
    if not args.out:
        args.out = ""
    cmd_analyze(args)


def cmd_monitor(args: argparse.Namespace) -> None:
    print("=== LEDSpector Monitor ===")
    led_config = load_config(args.config)

    cam, _ = open_camera(
        device=args.device,
        width=1280,
        height=720,
        fps=args.fps,
        fourcc=args.fourcc,
    )
    mon: Optional[Monitor] = None
    try:
        led_config.check_resolution(cam.actual_width, cam.actual_height)
        _apply_startup_controls(cam, args)
        mon = Monitor(led_config, cam, window_seconds=args.window, use_color=not args.brightness_only)
        if args.log:
            mon.open_log(args.log)
        if args.annotated:
            _install_snapshot_signal(mon, args.annotated, with_grid=not args.annotated_no_grid)
            print(
                f"[monitor] Снимок с разметкой по Ctrl+\\ -> {args.annotated}"
            )
        if args.log_only or not sys.stdout.isatty():
            mon.run_log(interval=args.interval)
        else:
            mon.run_table(tty=True, interval=args.interval, show_events=not args.no_events)
    finally:
        if mon is not None:
            mon.close_log()
        cam.release()


def cmd_log(args: argparse.Namespace) -> None:
    args.log_only = True
    cmd_monitor(args)


def _add_common_args(parser: argparse.ArgumentParser, log_default: Optional[str]) -> None:
    """Общий набор опций для monitor и log — чтобы не расходились."""
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--fourcc",
        default=DEFAULT_FOURCC,
        help="Формат камеры: mjpg даёт 1280x720@20, yuyv — только 640x480@10",
    )
    parser.add_argument("--exposure", type=int)
    parser.add_argument("--window", type=float, default=2.0)
    parser.add_argument("--interval", type=float, default=0.1)
    parser.add_argument("--log", default=log_default)
    parser.add_argument("--log-only", action="store_true")
    parser.add_argument("--no-events", action="store_true")
    parser.add_argument(
        "--brightness-only", action="store_true", help="Только яркость (без HSV-цвета)"
    )
    parser.add_argument(
        "--set-ctrl",
        action="append",
        metavar="NAME=VALUE",
        help=(
            "Контрол камеры через v4l2-ctl, можно перечислить через запятую. "
            "Повторяемый флаг. Например: --set-ctrl gain=1,saturation=256"
        ),
    )
    parser.add_argument(
        "--annotated",
        help="Сохранять кадр с разметкой по Ctrl+\\ в этот файл",
    )
    parser.add_argument(
        "--annotated-no-grid",
        action="store_true",
        help="Сохранять аннотированный кадр без координатной сетки",
    )


def _add_camera_args(parser: argparse.ArgumentParser) -> None:
    """Опции камеры для команд, работающих и со снимком, и с устройством."""
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--fourcc",
        default=DEFAULT_FOURCC,
        help="Формат камеры: mjpg даёт 1280x720@20, yuyv — только 640x480@10",
    )
    parser.add_argument("--exposure", type=int)
    parser.add_argument(
        "--set-ctrl",
        action="append",
        metavar="NAME=VALUE",
        help="Контролы камеры, например --set-ctrl gain=1,saturation=256",
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="run_linux.py", description="LEDspector Linux CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("check", help="Диагностика окружения и камеры")
    s.set_defaults(func=cmd_check)

    s = sub.add_parser("grid", help="Снять кадр с сеткой и сохранить PNG")
    s.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    s.add_argument("--device", type=int, default=0)
    s.add_argument("--fps", type=int, default=30)
    s.add_argument(
        "--fourcc",
        default=DEFAULT_FOURCC,
        help="Формат камеры: mjpg даёт 1280x720@20, yuyv — только 640x480@10",
    )
    s.add_argument("--exposure", type=int, default=500)
    s.add_argument("--out", default="snapshot.png")
    s.add_argument("--minor-step", type=int, default=10)
    s.add_argument("--major-step", type=int, default=50)
    s.set_defaults(func=cmd_grid)

    s = sub.add_parser("exposure", help="Подбор экспозиции в интерактивном режиме")
    s.add_argument("--config", default=DEFAULT_CONFIG_PATH)
    s.add_argument("--device", type=int, default=0)
    s.add_argument("--fps", type=int, default=30)
    s.add_argument(
        "--fourcc",
        default=DEFAULT_FOURCC,
        help="Формат камеры: mjpg даёт 1280x720@20, yuyv — только 640x480@10",
    )
    s.add_argument(
        "--set-ctrl",
        action="append",
        metavar="NAME=VALUE",
        help="Контролы камеры, например --set-ctrl gain=1,saturation=256",
    )
    s.set_defaults(func=cmd_exposure)

    s = sub.add_parser(
        "analyze",
        help="Найти диоды на снимке и подобрать точки замера вне пересвета",
    )
    s.add_argument(
        "--image",
        help="Разобрать готовый снимок (например docs/dim2_raw.png) вместо съёмки",
    )
    _add_camera_args(s)
    s.add_argument("--out", default="analyze_result.png", help="Куда сохранить разметку")
    s.add_argument(
        "--window-size", type=int, default=5, help="Размер ROI для замера, px"
    )
    s.set_defaults(func=cmd_analyze)

    s = sub.add_parser(
        "find",
        help="Как analyze, но кратко: только координаты новых точек замера",
    )
    s.add_argument("--image")
    _add_camera_args(s)
    s.add_argument("--out", default="")
    s.add_argument("--window-size", type=int, default=5)
    s.set_defaults(func=cmd_find)

    s = sub.add_parser("monitor", help="Таблица в реальном времени + лог изменений")
    _add_common_args(s, log_default=None)
    s.set_defaults(func=cmd_monitor)

    s = sub.add_parser("log", help="Только лог изменений (append-only)")
    _add_common_args(s, log_default="ledspector_changes.log")
    s.set_defaults(func=cmd_log)

    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()