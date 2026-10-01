"""Фабрика источников камеры.

Выбирает конкретную реализацию камеры по конфигурации
(``source: "webcam"`` или ``source: "scrcpy"``) из ``config/settings.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from src.camera import USBCamera
from src.phone_camera import ScrcpyPhoneCamera

SOURCE_WEB = "webcam"
SOURCE_SCRCPY = "scrcpy"

SOURCES: Tuple[str, str] = (SOURCE_WEB, SOURCE_SCRCPY)

# Базовые значения по умолчанию для секции "camera"
DEFAULT_CAMERA: Dict[str, Any] = {
    "source": SOURCE_WEB,
    "camera_id": 1,
    "width": 1280,
    "height": 720,
    "fps": 30,
    "scrcpy": {
        "window_title": "LEDPhoneCam",
        "window_width": 720,
        "window_height": 480,
        "camera_facing": "back",
        "camera_max_size": 720,
        "auto_launch": True,
        "launch_timeout_sec": 15,
    },
}

# Значения по умолчанию для scrcpy (если блок в конфиге отсутствует)
_DEFAULT_SCRCPY: Dict[str, Any] = DEFAULT_CAMERA["scrcpy"]


def read_camera_config(path: str = "config/settings.json") -> Dict[str, Any]:
    """Читает секцию ``"camera"`` из конфига с подстановкой дефолтов.

    Args:
        path: Путь к ``settings.json``.

    Returns:
        Словарь камеры, пригодный для :func:`create_camera`.
    """
    config_path = Path(path)
    if not config_path.is_file():
        return dict(DEFAULT_CAMERA)

    try:
        with open(config_path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return dict(DEFAULT_CAMERA)

    raw = data.get("camera") or {}
    merged = {**DEFAULT_CAMERA, **raw}
    merged["scrcpy"] = {**_DEFAULT_SCRCPY, **(raw.get("scrcpy") or {})}
    return merged


def create_camera(
    config: Dict[str, Any],
    source: str = SOURCE_WEB,
) -> USBCamera:
    """Создаёт экземпляр камеры по типу источника.

    Args:
        config: Словарь камеры из ``settings.json`` (``"camera"``).
        source: Тип источника: ``"webcam"`` или ``"scrcpy"``.

    Returns:
        ``USBCamera`` (веб-камера) или ``ScrcpyPhoneCamera`` (телефон).

    Raises:
        ValueError: для неизвестного типа источника.
    """
    if source == SOURCE_WEB:
        return USBCamera(
            camera_id=int(config.get("camera_id", 1)),
            width=int(config.get("width", 1280)),
            height=int(config.get("height", 720)),
            fps=int(config.get("fps", 30)),
        )

    if source == SOURCE_SCRCPY:
        scrcpy_cfg = {**_DEFAULT_SCRCPY, **(config.get("scrcpy") or {})}
        return ScrcpyPhoneCamera(
            window_title=str(scrcpy_cfg["window_title"]),
            window_w=int(scrcpy_cfg["window_width"]),
            window_h=int(scrcpy_cfg["window_height"]),
            camera_facing=str(scrcpy_cfg["camera_facing"]),
            camera_max_size=int(scrcpy_cfg["camera_max_size"]),
            auto_launch=bool(scrcpy_cfg["auto_launch"]),
            launch_timeout_sec=int(scrcpy_cfg["launch_timeout_sec"]),
        )

    raise ValueError(f"Неизвестный источник камеры: {source!r}")


def current_source(config: Dict[str, Any]) -> str:
    """Возвращает активный источник из конфига, если он валиден."""
    source = str(config.get("source", SOURCE_WEB))
    return source if source in SOURCES else SOURCE_WEB


__all__: List[str] = [
    "SOURCE_WEB",
    "SOURCE_SCRCPY",
    "SOURCES",
    "DEFAULT_CAMERA",
    "read_camera_config",
    "create_camera",
    "current_source",
]