"""Парсер конфигурации раскладки светодиодов (текстовый формат TSV).

Конфиг описывает, в каких координатах кадра какой светодиод расположен.
Формат рассчитан на ручное редактирование: сначала идут директивы шапки,
затем таблица, разделённая пробелами или табуляцией.

Пример файла::

    # LEDSpector — карта панели
    panel: VEGMAN-R220-front
    resolution: 1280 720

    # ID   X    Y    W    H   описание
    PWR   482  231  5    5   питание
    HDD   512  231  4    4   накопитель 1

Директивы распознаются по наличию двоеточия в начале строки, поэтому
комментарии и пустые строки безопасно оставлять в любом месте файла.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "LED",
    "LedConfig",
    "load_config",
    "save_config",
    "DEFAULT_CONFIG_PATH",
]

DEFAULT_CONFIG_PATH = "config/leds.tsv"

DIRECTIVE_PREFIX = "#!"


@dataclass
class LED:
    """Один светодиод: идентификатор и прямоугольник считывания в кадре."""

    id: str
    x: int
    y: int
    w: int = 5
    h: int = 5
    description: str = ""

    @property
    def bbox(self) -> Tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)

    @property
    def center(self) -> Tuple[int, int]:
        return (self.x + self.w // 2, self.y + self.h // 2)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "description": self.description,
        }


@dataclass
class LedConfig:
    """Разобранная раскладка панели."""

    panel: str = "unnamed"
    resolution: Tuple[int, int] = (1280, 720)
    leds: List[LED] = field(default_factory=list)
    path: Optional[Path] = None

    @property
    def width(self) -> int:
        return self.resolution[0]

    @property
    def height(self) -> int:
        return self.resolution[1]

    def get(self, led_id: str) -> Optional[LED]:
        for led in self.leds:
            if led.id == led_id:
                return led
        return None

    def check_resolution(self, width: int, height: int) -> None:
        """Проверяет, что кадр совпадает с разрешением из шапки конфига.

        Расхождение означает, что координаты из конфига указывают на другие
        пиксели, поэтому анализировать такой кадр бессмысленно.
        """
        if (width, height) == self.resolution:
            return
        raise ValueError(
            f"Разрешение кадра {width}x{height} не совпадает с конфигом "
            f"{self.width}x{self.height} (панель {self.panel!r}). "
            "Координаты LED указывают на другие пиксели — "
            "перенастройте камеру или исправьте директиву resolution."
        )


def _split_fields(line: str) -> List[str]:
    return line.split()


def _parse_resolution(value: str) -> Tuple[int, int]:
    parts = value.replace("x", " ").replace("*", " ").split()
    if len(parts) != 2:
        raise ValueError(f"Ожидалось 'ширина высота', получено {value!r}")
    return (int(parts[0]), int(parts[1]))


KNOWN_DIRECTIVES = ("panel", "resolution")


def _is_directive_line(text: str) -> bool:
    """Проверяет, что строка 'ключ: значение' — известная директива."""
    if ":" not in text:
        return False
    key = text.partition(":")[0].strip().lower()
    return key in KNOWN_DIRECTIVES


def _parse_directive(key: str, value: str) -> Tuple[str, Any]:
    key = key.strip().lower()
    if key == "panel":
        return "panel", value.strip()
    if key == "resolution":
        return "resolution", _parse_resolution(value)
    raise ValueError(f"Неизвестная директива {key!r}")


def load_config(path: str = DEFAULT_CONFIG_PATH) -> LedConfig:
    """Загружает конфиг раскладки.

    Битые строки пропускаются с предупреждением и номером строки, а не
    приводят к падению: во время первоначальной разметки конфиг правится
    руками, и одна опечатка не должна ронять наблюдение в проде.
    """
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Конфиг не найден: {config_path}. "
            "Создайте его по образцу config/leds.tsv."
        )

    panel = "unnamed"
    resolution: Tuple[int, int] = (1280, 720)
    leds: List[LED] = []
    seen_ids: Dict[str, int] = {}

    for lineno, raw_line in enumerate(
        config_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue

        # Комментарий, если это не директива. Директива распознаётся и с
        # префиксом '#!' (явно), и как известный ключ с двоеточием после
        # необязательного '#' — иначе обычный комментарий вида
        # '# примечание: ...' молча превратился бы в директиву.
        body = line
        if body.startswith(DIRECTIVE_PREFIX):
            body = body[len(DIRECTIVE_PREFIX):].strip()
        elif body.startswith("#") and _is_directive_line(body.lstrip("#").strip()):
            body = body.lstrip("#").strip()
        elif body.startswith("#"):
            continue

        if ":" in body:
            key, _, value = body.partition(":")
            try:
                parsed_key, parsed_value = _parse_directive(key, value)
            except ValueError as exc:
                print(f"[led_config] Строка {lineno}: {exc} — пропущено")
                continue
            if parsed_key == "panel":
                panel = str(parsed_value)
            else:
                resolution = parsed_value
            continue

        fields = _split_fields(body)
        if len(fields) < 3:
            print(
                f"[led_config] Строка {lineno}: нужно минимум 'ID X Y', "
                f"получено {len(fields)} полей — пропущено"
            )
            continue

        led_id = fields[0]
        try:
            x, y = int(fields[1]), int(fields[2])
            w = int(fields[3]) if len(fields) > 3 else 5
            h = int(fields[4]) if len(fields) > 4 else 5
        except ValueError:
            print(
                f"[led_config] Строка {lineno}: координаты для {led_id!r} "
                "не числа — пропущено"
            )
            continue

        if w <= 0 or h <= 0:
            print(
                f"[led_config] Строка {lineno}: размер {w}x{h} для "
                f"{led_id!r} неположительный — пропущено"
            )
            continue

        if led_id in seen_ids:
            print(
                f"[led_config] Строка {lineno}: дубликат ID {led_id!r} "
                f"(уже был на строке {seen_ids[led_id]}) — пропущено"
            )
            continue

        description = " ".join(fields[5:]) if len(fields) > 5 else ""
        seen_ids[led_id] = lineno
        leds.append(LED(id=led_id, x=x, y=y, w=w, h=h, description=description))

    return LedConfig(
        panel=panel,
        resolution=resolution,
        leds=leds,
        path=config_path,
    )


def _format_resolution(resolution: Tuple[int, int]) -> str:
    return f"{resolution[0]} {resolution[1]}"


def save_config(config: LedConfig, path: Optional[str] = None) -> Path:
    """Записывает конфиг обратно в файл (атомарно)."""
    target = Path(path) if path else config.path
    if target is None:
        raise ValueError("Не указан путь для сохранения конфига")

    lines: List[str] = [
        "# LEDSpector — карта панели",
        f"{DIRECTIVE_PREFIX}panel: {config.panel}",
        f"{DIRECTIVE_PREFIX}resolution: {_format_resolution(config.resolution)}",
        "",
        "# ID   X    Y    W    H   описание",
    ]
    for led in config.leds:
        tail = f"   {led.description}" if led.description else ""
        lines.append(f"{led.id:<6}{led.x:<5}{led.y:<5}{led.w:<5}{led.h:<5}{tail}")

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target.with_suffix(target.suffix + ".tmp")
    tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp_path.replace(target)
    return target


def to_json(config: LedConfig) -> str:
    """Отладочный дамп конфига в JSON — для отправки в разбор."""
    payload = {
        "panel": config.panel,
        "resolution": list(config.resolution),
        "leds": [led.to_dict() for led in config.leds],
    }
    return json.dumps(payload, indent=2, ensure_ascii=False)