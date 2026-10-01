"""Единая точка входа в проект LEDSpector.

Запускает интерактивное меню выбора режима работы:

    python run.py

GUI-режимы (камера/ROI/анализ/эмулятор) стартуют как отдельные процессы
и не блокируют меню — можно запускать несколько модулей одновременно.
Завершение процессов выполняется вручную (пункт 5) или автоматически при
выходе из лаунчера.

Режимы:
    1. Тест камеры        — проверка захвата камеры (FPS, экспозиция)
    2. Назначение зон ROI  — интерактивная разметка светодиодов на кадре
    3. Анализ (трекинг)   — детекция цвета и частоты мигания светодиодов
    4. Эмулятор LED       — имитация панели сервера для отладки анализатора
    5. Остановить модуль  — принудительно завершить запущенный процесс
    6. Источник камеры    — выбор между USB-вебкой и телефоном (scrcpy)
    0. Выход
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

# Корректный вывод кириллицы в Windows-консолях (кодировка cp1251/866)
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stdin.reconfigure(encoding="utf-8")
except (AttributeError, ValueError):
    pass

CONFIG_PATH: str = "config/settings.json"
PROJECT_ROOT: Path = Path(__file__).resolve().parent

MODES: List[Tuple[int, str, Callable[[], None]]] = []

# Тело задачи: (уникальный id, подпись, процесс)
TASKS: List[Tuple[int, str, subprocess.Popen]] = []
_next_task_id = 1

# Номер режима -> скрипт, запускаемый в отдельном процессе.
RUNNABLE: Dict[int, Path] = {
    1: PROJECT_ROOT / "test_camera.py",
    2: PROJECT_ROOT / "test_roi.py",
    3: PROJECT_ROOT / "test_analyzer.py",
    4: PROJECT_ROOT / "src" / "led_emulator.py",
}


def _run_camera_test() -> None:
    _launch_task(1, "Тест камеры")


def _run_roi_selection() -> None:
    _launch_task(2, "Назначение зон (ROI)")


def _run_analyzer() -> None:
    _launch_task(3, "Анализ цвета и частоты")


def _run_emulator() -> None:
    _launch_task(4, "Эмулятор светодиодной панели")


def _init_modes() -> None:
    if MODES:
        return
    MODES.extend([
        (1, "Тест камеры", _run_camera_test),
        (2, "Назначение зон (ROI)", _run_roi_selection),
        (3, "Анализ цвета и частоты", _run_analyzer),
        (4, "Эмулятор светодиодной панели", _run_emulator),
        (5, "Остановить запущенный модуль", _stop_task_prompt),
        (6, "Источник камеры", _switch_camera_source),
    ])


def _launch_task(mode_num: int, label: str) -> None:
    """Запускает модуль в отдельном процессе и возвращает управление меню."""
    global _next_task_id

    _prune_tasks()
    script = RUNNABLE[mode_num]
    proc = subprocess.Popen([sys.executable, str(script)], cwd=str(PROJECT_ROOT))

    TASKS.append((_next_task_id, label, proc))
    print(f"Запущено: {label} (задача #{_next_task_id}, PID {proc.pid})")
    _next_task_id += 1


def _prune_tasks() -> None:
    """Удаляет из списка уже завершившиеся (собственно) процессы."""
    alive = [t for t in TASKS if t[2].poll() is None]
    TASKS[:] = alive


def _terminate(proc: subprocess.Popen) -> None:
    """Мягко завершает процесс, при подвисании — принудительно."""
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(timeout=3)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=3)


def _stop_task_prompt() -> None:
    """Интерактивная остановка выбранного запущенного модуля."""
    _prune_tasks()

    if not TASKS:
        print("Нет запущенных модулей.")
        return

    print("\nЗапущенные модули:")
    for task_id, label, proc in TASKS:
        print(f"  {task_id}. {label} (PID {proc.pid})")
    print("  0. Назад")

    raw = input("Остановить модуль: ").strip()
    if raw == "0":
        return
    try:
        task_id = int(raw)
    except ValueError:
        print("Некорректный ввод.")
        return

    for i, (tid, label, proc) in enumerate(TASKS):
        if tid == task_id:
            _terminate(proc)
            del TASKS[i]
            print(f"Остановлено: {label} (задача #{tid})")
            return

    print(f"Задача #{task_id} не найдена.")


def _stop_all_tasks() -> None:
    """Завершает все запущенные модули (выход из лаунчера / Ctrl+C)."""
    if not TASKS:
        return
    for task_id, label, proc in TASKS:
        _terminate(proc)
        print(f"Остановлено: {label} (задача #{task_id})")
    TASKS.clear()


def _running_tasks_line() -> str:
    _prune_tasks()
    if not TASKS:
        return "Активные задачи: нет"
    parts = ", ".join(f"#{tid}: {label} (PID {proc.pid})" for tid, label, proc in TASKS)
    return f"Активные задачи: {parts}"


def _current_source() -> str:
    """Возвращает активный источник камеры из конфига."""
    from src.camera_factory import current_source, read_camera_config
    return current_source(read_camera_config(CONFIG_PATH))


def _source_label(source: str) -> str:
    """Человекочитаемая подпись источника."""
    return {
        "webcam": "webcam (USB)",
        "scrcpy": "scrcpy (телефон)",
    }.get(source, source)


def _print_menu() -> None:
    _init_modes()
    print("\n=== LEDSpector ===")
    print(f"  Камера: {_source_label(_current_source())}")
    print(f"  {_running_tasks_line()}")
    for num, label, _ in MODES:
        print(f"  {num}. {label}")
    print("  0. Выход")


def _choose_mode() -> int:
    while True:
        raw = input("Выбор: ").strip()
        if raw == "0":
            return 0
        try:
            choice = int(raw)
        except ValueError:
            print("Некорректный ввод, попробуйте ещё раз.")
            continue
        if any(num == choice for num, _, _ in MODES):
            return choice
        print(f"Режим {choice} не существует.")


def _persist_source(source: str) -> None:
    """Сохраняет выбранный источник камеры в ``settings.json``."""
    from src.camera_factory import SOURCES

    path = Path(CONFIG_PATH)
    data: dict = {}
    if path.is_file():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            data = {}

    camera = data.setdefault("camera", {})
    camera["source"] = source if source in SOURCES else "webcam"

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8",
    )


def _switch_camera_source() -> None:
    """Интерактивный выбор источника камеры."""
    from src.camera_factory import SOURCE_SCRCPY, SOURCE_WEB, current_source, read_camera_config

    config = read_camera_config(CONFIG_PATH)
    current = current_source(config)

    print("\nИсточник камеры:")
    print(f"  1. Web-камера (USB)      [текущий: {'да' if current == SOURCE_WEB else 'нет'}]")
    print(f"  2. Телефон через scrcpy  [текущий: {'да' if current == SOURCE_SCRCPY else 'нет'}]")
    print("  0. Назад")

    choice = input("Выбор: ").strip()
    target = {SOURCE_WEB: "1", SOURCE_SCRCPY: "2"}.get(current)

    if choice == "0" or choice == target:
        return

    if choice == "1":
        _persist_source(SOURCE_WEB)
        print("Источник камеры: webcam (USB)")
    elif choice == "2":
        _persist_source(SOURCE_SCRCPY)
        print("Источник камеры: scrcpy (телефон)")
    else:
        print("Некорректный выбор.")


def main() -> None:
    """Главный цикл меню."""
    _init_modes()

    while True:
        _print_menu()
        choice = _choose_mode()

        if choice == 0:
            _stop_all_tasks()
            print("До свидания!")
            break

        handler = next((fn for num, _, fn in MODES if num == choice), None)
        if handler is None:
            continue

        try:
            handler()
        except Exception as exc:  # noqa: BLE001 — единая точка логирования
            print(f"\n[Ошибка] {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        _stop_all_tasks()
        sys.exit(0)