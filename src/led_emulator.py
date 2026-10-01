"""Эмулятор светодиодной индикации сервера (tkinter).

Позволяет создать шаблон передней панели сервера с произвольным
расположением светодиодов, их цветом и режимом мигания (OFF, SOLID,
BLINK_1HZ, BLINK_4HZ, BLINK_CUSTOM). Удобен для отладки анализатора —
на эмулятор направляется USB-камера.

Раскладка сохраняется/загружается из JSON (``config/emulator_layouts.json``).
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import messagebox

CONFIG_PATH: str = "config/emulator_layouts.json"
TICK_MS: int = 20

# Размеры панели и диодов (пиксели)
DEFAULT_PANEL_W: int = 1280
DEFAULT_PANEL_H: int = 480
DEFAULT_LED_SIZE: int = 64
MIN_LED_SIZE: int = 16
MAX_LED_SIZE: int = 240

# Минимальный период мигания (сек) — ограничение частоты сверху
MIN_PERIOD: float = 0.05

# Пресеты частоты мигания: (подпись кнопки, частота в Гц)
FREQ_PRESETS: Tuple[Tuple[str, float], ...] = (
    ("0.5 Гц", 0.5),
    ("1 Гц", 1.0),
    ("2 Гц", 2.0),
    ("4 Гц", 4.0),
    ("8 Гц", 8.0),
)

# Палитра: имя -> (цвет включённого, цвет выключенного) в hex #RRGGBB
COLOR_PALETTE: Dict[str, Tuple[str, str]] = {
    "GREEN": ("#00FF00", "#004A00"),
    "RED": ("#FF0000", "#4A0000"),
    "BLUE": ("#0066FF", "#001A4A"),
    "AMBER": ("#FFAA00", "#4A3200"),
    "WHITE": ("#FFFFFF", "#4A4A4A"),
    "ORANGE": ("#FF6600", "#4A1F00"),
}

PATTERNS: Tuple[str, ...] = (
    "OFF", "SOLID", "BLINK_1HZ", "BLINK_4HZ", "BLINK_CUSTOM",
)


def _freq_to_period(hz: float) -> float:
    """Переводит частоту мигания (Гц) в период (сек).

    Args:
        hz: Частота в герцах. Значения <= 0 трактуются как максимум.

    Returns:
        Период в секундах, не меньше :data:`MIN_PERIOD`.
    """
    if hz <= 0:
        return MIN_PERIOD
    return max(MIN_PERIOD, 1.0 / hz)


def _pattern_is_on(
    pattern: str, t: float, custom_period: float,
) -> bool:
    """Вычисляет состояние светодиода в момент времени ``t``.

    Args:
        pattern: Имя режима из :data:`PATTERNS`.
        t: Момент времени (монотонные часы).
        custom_period: Период для ``BLINK_CUSTOM``.

    Returns:
        ``True`` если диод должен светиться.
    """
    if pattern == "OFF":
        return False
    if pattern == "SOLID":
        return True
    if pattern == "BLINK_1HZ":
        period = 1.0
    elif pattern == "BLINK_4HZ":
        period = 0.25
    else:  # BLINK_CUSTOM
        period = max(custom_period, MIN_PERIOD)
    return (t % period) < (period / 2)


@dataclass
class LEDElement:
    """Модель светодиода на панели эмулятора.

    ``canvas_key`` — внутренний ключ для привязки к элементам Canvas,
    в JSON не сериализуется (генерируется заново при загрузке).

    Attributes:
        id: Пользовательское имя диода.
        x: X-координата центра (пиксели Canvas).
        y: Y-координата центра.
        size: Диаметр диода в пикселях.
        on_color: Цвет в включённом состоянии (hex ``#RRGGBB``).
        off_color: Цвет в выключенном состоянии.
        pattern: Режим из :data:`PATTERNS`.
        custom_period: Период для ``BLINK_CUSTOM``.
        phase: Фазовый сдвиг (сек) — чтобы соседние диоды мигали вразнобой.
    """

    id: str
    x: int
    y: int
    size: int = DEFAULT_LED_SIZE
    on_color: str = "#00FF00"
    off_color: str = "#004A00"
    pattern: str = "SOLID"
    custom_period: float = 1.0
    phase: float = 0.0
    canvas_key: str = field(
        default_factory=lambda: uuid.uuid4().hex, repr=False,
    )

    def to_dict(self) -> Dict[str, Any]:
        """Сериализация в словарь для JSON."""
        return {
            "id": self.id,
            "x": self.x,
            "y": self.y,
            "size": self.size,
            "on_color": self.on_color,
            "off_color": self.off_color,
            "pattern": self.pattern,
            "custom_period": self.custom_period,
            "phase": self.phase,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional["LEDElement"]:
        """Восстановление модели из словаря (с валидацией полей).

        Args:
            data: Десериализованный JSON-объект.

        Returns:
            ``LEDElement`` или ``None``, если данные некорректны.
        """
        try:
            x, y = int(data["x"]), int(data["y"])
            size = int(data.get("size", DEFAULT_LED_SIZE))
        except (KeyError, TypeError, ValueError):
            return None

        size = max(MIN_LED_SIZE, min(MAX_LED_SIZE, size))
        pattern = str(data.get("pattern", "SOLID"))
        if pattern not in PATTERNS:
            pattern = "SOLID"

        try:
            period = max(MIN_PERIOD, float(data.get("custom_period", 1.0)))
        except (TypeError, ValueError):
            period = 1.0
        try:
            phase = float(data.get("phase", 0.0))
        except (TypeError, ValueError):
            phase = 0.0

        return cls(
            id=str(data.get("id", "LED")),
            x=x,
            y=y,
            size=size,
            on_color=str(data.get("on_color", "#00FF00")),
            off_color=str(data.get("off_color", "#004A00")),
            pattern=pattern,
            custom_period=period,
            phase=phase,
        )


class LEDEmulator(tk.Tk):
    """Главное окно эмулятора: холст панели + панель свойств выбранного диода."""

    def __init__(self, config_path: str = CONFIG_PATH) -> None:
        """Инициализация окна и загрузка раскладки.

        Args:
            config_path: Путь к JSON-файлу раскладки.
        """
        super().__init__()
        self.config_path: str = config_path

        # --- Модель ---
        self.leds: List[LEDElement] = []
        self.panel_name: str = "Server 1U — Demo"
        self.panel_w: int = DEFAULT_PANEL_W
        self.panel_h: int = DEFAULT_PANEL_H
        self.panel_bg: str = "#222222"

        self.selected_key: Optional[str] = None
        self._elems_by_key: Dict[str, LEDElement] = {}
        self._canvas_items: Dict[str, Tuple[int, int]] = {}
        self._item_to_key: Dict[int, str] = {}
        self._last_states: Dict[str, bool] = {}
        self._drag_key: Optional[str] = None
        self._drag_off_x: int = 0
        self._drag_off_y: int = 0
        self._populating: bool = False
        self._running: bool = True
        self._after_id: Optional[str] = None
        self._free_place: bool = False
        self._freq_buttons: List[tk.Button] = []

        self.title("LED Inspector — Эмулятор панели сервера")
        self.resizable(False, False)

        self._build_toolbar()
        self._build_main()
        self._bind_variables()

        self._fit_window()

        self.load_config()
        self._populate_panel()
        self._animate()

        self.protocol("WM_DELETE_WINDOW", self.on_close)

    def _fit_window(self) -> None:
        """Подгоняет размер окна под содержимое (холст + панель свойств)."""
        self.update_idletasks()
        width = self.winfo_reqwidth()
        height = self.winfo_reqheight()
        if width > 1 and height > 1:
            self.geometry(f"{width}x{height}")

    # ------------------------------------------------------------------
    # Построение UI
    # ------------------------------------------------------------------

    def _build_toolbar(self) -> None:
        bar = tk.Frame(self, padx=4, pady=4)
        bar.pack(side=tk.TOP, fill=tk.X)

        tk.Button(bar, text="Добавить LED", command=self.add_led).pack(
            side=tk.LEFT, padx=2,
        )
        tk.Button(bar, text="Удалить", command=self.delete_selected).pack(
            side=tk.LEFT, padx=2,
        )
        tk.Button(bar, text="Сохранить", command=self.save_config).pack(
            side=tk.LEFT, padx=2,
        )
        tk.Button(bar, text="Загрузить", command=self.load_config).pack(
            side=tk.LEFT, padx=2,
        )

    def _build_main(self) -> None:
        main = tk.Frame(self)
        main.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=4)

        # --- Холст панели ---
        canvas_frame = tk.Frame(main, bd=1, relief=tk.SUNKEN)
        canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        self.canvas = tk.Canvas(
            canvas_frame, width=self.panel_w, height=self.panel_h,
            bg=self.panel_bg, highlightthickness=0,
        )
        self.canvas.pack(padx=4, pady=4)

        self.canvas.bind("<Button-1>", self._on_press)
        self.canvas.bind("<B1-Motion>", self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        # --- Панель свойств ---
        self._build_properties(main)

    def _build_properties(self, parent: tk.Frame) -> None:
        frame = tk.Frame(parent, padx=8, pady=8, width=270)
        frame.pack(side=tk.RIGHT, fill=tk.Y)
        frame.pack_propagate(False)

        tk.Label(frame, text="Свойства диода", font=("TkDefaultFont", 10, "bold")
                 ).pack(anchor=tk.W, pady=(0, 6))

        # Имя
        tk.Label(frame, text="Имя:").pack(anchor=tk.W)
        self.var_id = tk.StringVar()
        self.entry_id = tk.Entry(frame, textvariable=self.var_id)
        self.entry_id.pack(fill=tk.X, pady=(0, 4))

        # Цвет
        tk.Label(frame, text="Цвет:").pack(anchor=tk.W)
        self.var_color = tk.StringVar()
        color_values = tuple(COLOR_PALETTE.keys())
        self.color_menu = tk.OptionMenu(
            frame, self.var_color, *color_values,
        )
        self.color_menu.pack(fill=tk.X, pady=(0, 4))

        # Режим мигания
        tk.Label(frame, text="Режим:").pack(anchor=tk.W)
        self.var_pattern = tk.StringVar()
        self.pattern_menu = tk.OptionMenu(
            frame, self.var_pattern, *PATTERNS,
        )
        self.pattern_menu.pack(fill=tk.X, pady=(0, 4))

        # Размер
        tk.Label(frame, text="Размер:").pack(anchor=tk.W)
        self.var_size = tk.IntVar(value=DEFAULT_LED_SIZE)
        self.scale_size = tk.Scale(
            frame, from_=MIN_LED_SIZE, to=MAX_LED_SIZE, orient=tk.HORIZONTAL,
            variable=self.var_size,
        )
        self.scale_size.pack(fill=tk.X, pady=(0, 4))

        # Позиция
        pos = tk.Frame(frame)
        pos.pack(fill=tk.X, pady=(0, 4))
        tk.Label(pos, text="X:").pack(side=tk.LEFT)
        self.var_x = tk.StringVar()
        self.entry_x = tk.Entry(pos, textvariable=self.var_x, width=6)
        self.entry_x.pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(pos, text="Y:").pack(side=tk.LEFT)
        self.var_y = tk.StringVar()
        self.entry_y = tk.Entry(pos, textvariable=self.var_y, width=6)
        self.entry_y.pack(side=tk.LEFT)

        # Частота мигания: пресеты + период для BLINK_CUSTOM
        tk.Label(frame, text="Частота мигания:").pack(anchor=tk.W)

        presets = tk.Frame(frame)
        presets.pack(fill=tk.X, pady=(0, 4))
        for i, (label, hz) in enumerate(FREQ_PRESETS):
            btn = tk.Button(
                presets, text=label, width=5,
                command=lambda h=hz: self._apply_freq_preset(h),
            )
            btn.grid(row=0, column=i, padx=1)
            self._freq_buttons.append(btn)

        tk.Label(frame, text="Период (BLINK_CUSTOM, сек):").pack(anchor=tk.W)
        self.var_period = tk.StringVar(value="1.0")
        self.entry_period = tk.Entry(frame, textvariable=self.var_period)
        self.entry_period.pack(fill=tk.X, pady=(0, 4))

        tk.Label(frame, text="Сдвиг фазы (сек):").pack(anchor=tk.W)
        self.var_phase = tk.StringVar(value="0.0")
        self.entry_phase = tk.Entry(frame, textvariable=self.var_phase)
        self.entry_phase.pack(fill=tk.X, pady=(0, 4))

        # Размер панели
        tk.Frame(frame, height=1, bd=1, relief=tk.SUNKEN).pack(
            fill=tk.X, pady=8,
        )
        tk.Label(frame, text="Панель (ширина x высота):",
                 font=("TkDefaultFont", 10, "bold")).pack(anchor=tk.W, pady=(0, 4))

        psize = tk.Frame(frame)
        psize.pack(fill=tk.X, pady=(0, 4))
        self.var_pw = tk.StringVar(value=str(self.panel_w))
        self.entry_pw = tk.Entry(psize, textvariable=self.var_pw, width=6)
        self.entry_pw.pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(psize, text="x").pack(side=tk.LEFT)
        self.var_ph = tk.StringVar(value=str(self.panel_h))
        self.entry_ph = tk.Entry(psize, textvariable=self.var_ph, width=6)
        self.entry_ph.pack(side=tk.LEFT, padx=(8, 0))

        # Поле для виджетов, которые отключаются без выбранного диода
        self._panel_widgets: List[tk.Widget] = [
            self.entry_id, self.color_menu, self.pattern_menu,
            self.scale_size, self.entry_x, self.entry_y,
            self.entry_period, self.entry_phase, *self._freq_buttons,
        ]

        # Статус-бар
        self.status_var = tk.StringVar(value="")
        tk.Label(
            self, textvariable=self.status_var, anchor=tk.W,
            bd=1, relief=tk.SUNKEN, padx=6, pady=2,
        ).pack(side=tk.BOTTOM, fill=tk.X)

    def _bind_variables(self) -> None:
        """Подписка на изменения переменных свойств диода."""
        for var in (self.var_id, self.var_color, self.var_pattern,
                    self.var_size, self.var_x, self.var_y,
                    self.var_period, self.var_phase):
            var.trace_add("write", self._on_property_change)
        # Размеры панели
        self.var_pw.trace_add("write", self._on_panel_change)
        self.var_ph.trace_add("write", self._on_panel_change)

    # ------------------------------------------------------------------
    # Модель <-> View
    # ------------------------------------------------------------------

    def _rebuild_canvas(self) -> None:
        """Полностью перерисовывает холст из текущей модели."""
        self.canvas.delete("all")
        self.canvas.configure(
            width=self.panel_w, height=self.panel_h, bg=self.panel_bg,
        )
        self._canvas_items.clear()
        self._item_to_key.clear()
        self._elems_by_key.clear()
        self._last_states.clear()
        self.selected_key = None
        self._drag_key = None
        self._free_place = False

        for elem in self.leds:
            elem.canvas_key = uuid.uuid4().hex
            self._elems_by_key[elem.canvas_key] = elem
            self._draw_element(elem)

    def _draw_element(self, elem: LEDElement) -> None:
        """Создаёт canvas-элементы для диода (тело + рамка выделения)."""
        key = elem.canvas_key
        r = elem.size / 2.0

        body = self.canvas.create_oval(
            elem.x - r, elem.y - r, elem.x + r, elem.y + r,
            fill=elem.off_color, outline="#808080", width=1,
        )
        sel = self.canvas.create_rectangle(
            elem.x - r - 4, elem.y - r - 4, elem.x + r + 4, elem.y + r + 4,
            outline="#00FFFF", width=2, state="hidden",
        )
        self.canvas.addtag_withtag(f"led_body:{key}", body)
        self.canvas.addtag_withtag(f"led_sel:{key}", sel)

        self._canvas_items[key] = (body, sel)
        self._item_to_key[body] = key
        self._item_to_key[sel] = key
        self._last_states[key] = False

    def _update_element_shape(self, elem: LEDElement) -> None:
        """Перемещает/перерисовывает диод на холсте по текущим координатам."""
        key = elem.canvas_key
        r = elem.size / 2.0
        self.canvas.coords(
            f"led_body:{key}", elem.x - r, elem.y - r, elem.x + r, elem.y + r,
        )
        self.canvas.coords(
            f"led_sel:{key}", elem.x - r - 4, elem.y - r - 4,
            elem.x + r + 4, elem.y + r + 4,
        )
        # Применяем текущий цвет сразу (а не ждём тик анимации)
        on = _pattern_is_on(
            elem.pattern, time.monotonic() - elem.phase, elem.custom_period,
        )
        self._last_states[key] = on
        self.canvas.itemconfigure(
            f"led_body:{key}", fill=elem.on_color if on else elem.off_color,
        )

    # ------------------------------------------------------------------
    # Выбор и перетаскивание
    # ------------------------------------------------------------------

    def _led_at(self, x: int, y: int) -> Optional[str]:
        """Возвращает ключ диода под точкой (x, y) либо ``None``.

        ``find_closest`` в tkinter всегда возвращает ближайший элемент
        независимо от расстояния, поэтому попадание проверяется явно —
        иначе клик по пустому месту панели «выбирал» произвольный диод.

        Args:
            x: X-координата клика в системе Canvas.
            y: Y-координата клика в системе Canvas.

        Returns:
            ``canvas_key`` диода, если точка попала внутрь его круга.
        """
        nearest = self.canvas.find_closest(x, y)
        if not nearest:
            return None

        key = self._item_to_key.get(int(nearest[0]))
        if key is None:
            return None

        elem = self._elems_by_key.get(key)
        if elem is None:
            return None

        dx, dy = x - elem.x, y - elem.y
        if dx * dx + dy * dy <= (elem.size / 2.0) ** 2:
            return key
        return None

    def _on_press(self, event: tk.Event) -> None:
        """Нажатие ЛКМ: размещение нового диода либо выбор существующего."""
        key = self._led_at(event.x, event.y)

        if self._free_place and key is None:
            # Режим размещения: клик по пустому месту создаёт диод
            self._free_place = False
            self._create_led(
                x=max(0, min(self.panel_w, event.x)),
                y=max(0, min(self.panel_h, event.y)),
            )
            return

        # Клик по существующему диоду отменяет режим размещения
        self._free_place = False

        if key is None:
            self.select_element(None)
            return

        self.select_element(key)
        elem = self._elems_by_key[key]
        self._drag_key = key
        self._drag_off_x = event.x - elem.x
        self._drag_off_y = event.y - elem.y

    def _on_drag(self, event: tk.Event) -> None:
        """Перемещение выбранного диода мышью."""
        if self._drag_key is None:
            return
        elem = self._elems_by_key[self._drag_key]

        nx = max(0, min(self.panel_w, event.x - self._drag_off_x))
        ny = max(0, min(self.panel_h, event.y - self._drag_off_y))
        dx, dy = nx - elem.x, ny - elem.y

        elem.x, elem.y = nx, ny
        self.canvas.move(f"led_body:{elem.canvas_key}", dx, dy)
        self.canvas.move(f"led_sel:{elem.canvas_key}", dx, dy)

        # Синхронизация панели свойств (защита от re-enter)
        self._populating = True
        try:
            self.var_x.set(str(nx))
            self.var_y.set(str(ny))
        finally:
            self._populating = False

    def _on_release(self, event: tk.Event) -> None:
        self._drag_key = None
        if self.selected_key:
            self._status(f"LED '{self._elems_by_key[self.selected_key].id}'")

    def select_element(self, key: Optional[str]) -> None:
        """Выделяет диод, показывает/скрывает рамки выделения."""
        self.selected_key = key
        for k in self._elems_by_key:
            state = "normal" if k == key else "hidden"
            self.canvas.itemconfigure(f"led_sel:{k}", state=state)
        self._populate_panel()

    # ------------------------------------------------------------------
    # Панель свойств
    # ------------------------------------------------------------------

    def _populate_panel(self) -> None:
        """Заполняет элементы панели свойств данными выбранного диода."""
        self._populating = True
        try:
            elem = (self._elems_by_key.get(self.selected_key)
                    if self.selected_key else None)

            if elem is None:
                self.var_id.set("")
                self.var_color.set(next(iter(COLOR_PALETTE)))
                self.var_pattern.set(PATTERNS[0])
                self.var_size.set(DEFAULT_LED_SIZE)
                self.var_x.set("0")
                self.var_y.set("0")
                self.var_period.set("1.0")
                self.var_phase.set("0.0")
                for w in self._panel_widgets:
                    w.configure(state="disabled")
                return

            color_name = self._color_name_for(elem.on_color)
            self.var_id.set(elem.id)
            self.var_color.set(color_name)
            self.var_pattern.set(elem.pattern)
            self.var_size.set(elem.size)
            self.var_x.set(str(elem.x))
            self.var_y.set(str(elem.y))
            self.var_period.set(str(elem.custom_period))
            self.var_phase.set(str(elem.phase))
            for w in self._panel_widgets:
                w.configure(state="normal")
        finally:
            self._populating = False

        self._sync_blink_controls()

    def _sync_blink_controls(self) -> None:
        """Активирует поле периода и пресеты только для ``BLINK_CUSTOM``."""
        elem = (self._elems_by_key.get(self.selected_key)
                if self.selected_key else None)
        enabled = elem is not None and elem.pattern == "BLINK_CUSTOM"
        state = "normal" if enabled else "disabled"
        self.entry_period.configure(state=state)
        for btn in self._freq_buttons:
            btn.configure(state=state)

    def _apply_freq_preset(self, hz: float) -> None:
        """Применяет пресет частоты к выбранному диоду.

        Переводит диод в режим ``BLINK_CUSTOM`` и выставляет период,
        соответствующий частоте ``hz``.

        Args:
            hz: Частота мигания в герцах.
        """
        if self._populating or self.selected_key is None:
            return
        elem = self._elems_by_key.get(self.selected_key)
        if elem is None:
            self._status("Нет выбранного диода")
            return

        elem.pattern = "BLINK_CUSTOM"
        elem.custom_period = _freq_to_period(hz)

        self._populating = True
        try:
            self.var_pattern.set(elem.pattern)
            self.var_period.set(f"{elem.custom_period:g}")
        finally:
            self._populating = False

        self._update_element_shape(elem)
        self._sync_blink_controls()
        self._status(f"LED '{elem.id}': {hz:g} Гц")

    def _on_property_change(self, *args: Any) -> None:
        """Обработчик изменений свойств: модель <- панель, вид <- модель."""
        if self._populating or self.selected_key is None:
            return

        elem = self._elems_by_key.get(self.selected_key)
        if elem is None:
            return

        color_pair = COLOR_PALETTE.get(self.var_color.get())
        if color_pair is not None:
            elem.on_color, elem.off_color = color_pair

        if self.var_pattern.get() in PATTERNS:
            elem.pattern = self.var_pattern.get()

        elem.size = max(MIN_LED_SIZE, min(MAX_LED_SIZE, self.var_size.get()))

        try:
            elem.x = max(0, min(self.panel_w, int(self.var_x.get())))
        except ValueError:
            pass
        try:
            elem.y = max(0, min(self.panel_h, int(self.var_y.get())))
        except ValueError:
            pass
        try:
            elem.custom_period = max(MIN_PERIOD, float(self.var_period.get()))
        except ValueError:
            pass
        try:
            elem.phase = float(self.var_phase.get())
        except ValueError:
            pass

        new_id = self.var_id.get().strip()
        if new_id:
            elem.id = new_id

        if key_in_canvas := self.canvas.find_withtag(
                f"led_body:{elem.canvas_key}"):
            self._update_element_shape(elem)
        self._sync_blink_controls()
        self._status(f"LED '{elem.id}'")

    def _on_panel_change(self, *args: Any) -> None:
        """Изменение размеров панели."""
        if self._populating:
            return
        try:
            w = max(100, int(self.var_pw.get()))
            h = max(50, int(self.var_ph.get()))
        except ValueError:
            return
        self.panel_w, self.panel_h = w, h
        self.canvas.configure(width=w, height=h)

    # ------------------------------------------------------------------
    # Анимация
    # ------------------------------------------------------------------

    def _animate(self) -> None:
        """Тик анимации: обновляет только диоды, сменившие состояние."""
        if not self._running:
            return
        now = time.monotonic()

        for elem in self.leds:
            on = _pattern_is_on(
                elem.pattern, now - elem.phase, elem.custom_period,
            )
            key = elem.canvas_key
            if self._last_states.get(key) != on:
                self._last_states[key] = on
                if self.canvas.find_withtag(f"led_body:{key}"):
                    self.canvas.itemconfigure(
                        f"led_body:{key}",
                        fill=elem.on_color if on else elem.off_color,
                    )

        self._after_id = self.after(TICK_MS, self._animate)

    # ------------------------------------------------------------------
    # Действия
    # ------------------------------------------------------------------

    def _create_led(
        self, x: int, y: int, size: int = DEFAULT_LED_SIZE,
    ) -> LEDElement:
        """Создаёт диод, регистрирует его на холсте и выделяет.

        Args:
            x: X-координата центра (пиксели Canvas).
            y: Y-координата центра.
            size: Диаметр диода в пикселях.

        Returns:
            Созданный элемент.
        """
        elem = LEDElement(
            id=self._next_led_id(),
            x=x,
            y=y,
            size=size,
            on_color=COLOR_PALETTE["GREEN"][0],
            off_color=COLOR_PALETTE["GREEN"][1],
            pattern="SOLID",
        )
        self.leds.append(elem)
        self._elems_by_key[elem.canvas_key] = elem
        self._draw_element(elem)
        self.select_element(elem.canvas_key)
        self._status(f"Добавлен '{elem.id}' — выберите цвет и частоту")
        return elem

    def add_led(self) -> None:
        """Включает режим размещения нового диода кликом по панели."""
        self._free_place = True
        self._status("Кликните по свободному месту панели для нового LED")

    def delete_selected(self) -> None:
        """Удаляет выбранный диод."""
        if self.selected_key is None:
            self._status("Нет выбранного диода")
            return
        key = self.selected_key
        elem = self._elems_by_key[key]
        self.canvas.delete(f"led_body:{key}")
        self.canvas.delete(f"led_sel:{key}")
        self.leds.remove(elem)
        del self._elems_by_key[key]
        self.selected_key = None
        self._last_states.pop(key, None)
        self._populate_panel()
        self._status(f"Удалён '{elem.id}'")

    # ------------------------------------------------------------------
    # Сохранение / загрузка
    # ------------------------------------------------------------------

    def save_config(self) -> None:
        """Сохраняет раскладку в JSON (атомарная запись через tmp-файл)."""
        data: Dict[str, Any] = {
            "version": 1,
            "panel": {
                "name": self.panel_name,
                "width": self.panel_w,
                "height": self.panel_h,
                "bg_color": self.panel_bg,
            },
            "leds": [e.to_dict() for e in self.leds],
        }

        path = Path(self.config_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        try:
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
            os.replace(tmp, path)
            self._status(f"Сохранено: {path}")
        except OSError as exc:
            messagebox.showerror("Ошибка", f"Не удалось сохранить: {exc}")

    def load_config(self) -> None:
        """Загружает раскладку из JSON. При отсутствии — создаёт дефолтную."""
        path = Path(self.config_path)
        if not path.is_file():
            self._seed_default()
            self.save_config()
            self._rebuild_canvas()
            self._status(f"Создан дефолтный конфиг: {path}")
            return

        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError) as exc:
            messagebox.showerror("Ошибка", f"Повреждён конфиг: {exc}")
            return

        panel = data.get("panel", {})
        self.panel_name = str(panel.get("name", self.panel_name))
        self.panel_w = max(100, int(panel.get("width", self.panel_w)))
        self.panel_h = max(50, int(panel.get("height", self.panel_h)))
        self.panel_bg = str(panel.get("bg_color", "#222222"))

        self.var_pw.set(str(self.panel_w))
        self.var_ph.set(str(self.panel_h))

        leds: List[LEDElement] = []
        for item in data.get("leds", []):
            elem = LEDElement.from_dict(item) if isinstance(item, dict) else None
            if elem is not None:
                leds.append(elem)
        self.leds = leds

        self._rebuild_canvas()
        self._status(f"Загружено {len(self.leds)} LED из {path}")

    def _seed_default(self) -> None:
        """Заполняет модель дефолтной раскладкой при отсутствии файла."""
        if self.leds:
            return
        spec = [
            ("PWR", 120, 120, "GREEN", "SOLID"),
            ("HDD", 320, 120, "RED", "BLINK_1HZ"),
            ("NET", 520, 120, "BLUE", "BLINK_4HZ"),
            ("FAULT", 720, 120, "AMBER", "OFF"),
            ("CUSTOM_2HZ", 920, 120, "WHITE", "BLINK_CUSTOM"),
        ]
        for name, x, y, color, pattern in spec:
            on_c, off_c = COLOR_PALETTE[color]
            self.leds.append(LEDElement(
                id=name, x=x, y=y, size=DEFAULT_LED_SIZE,
                on_color=on_c, off_color=off_c, pattern=pattern,
                custom_period=_freq_to_period(2.0) if pattern == "BLINK_CUSTOM" else 1.0,
            ))

    # ------------------------------------------------------------------
    # Вспомогательное
    # ------------------------------------------------------------------

    def _next_led_id(self) -> str:
        """Генерирует уникальное имя вида ``LED_N``."""
        used = {e.id for e in self.leds}
        n = 1
        while f"LED_{n}" in used:
            n += 1
        return f"LED_{n}"

    def _color_name_for(self, hex_color: str) -> str:
        """Возвращает имя палитры по hex-цвету (или первое совпадающее)."""
        for name, (on_c, _) in COLOR_PALETTE.items():
            if on_c.lower() == hex_color.lower():
                return name
        return next(iter(COLOR_PALETTE))

    def _status(self, text: str) -> None:
        self.status_var.set(text)

    def on_close(self) -> None:
        """Корректное завершение: остановка анимации и закрытие окна."""
        self._running = False
        if self._after_id is not None:
            try:
                self.after_cancel(self._after_id)
            except tk.TclError:
                pass
        self.destroy()


def main(config_path: str = CONFIG_PATH) -> None:
    """Точка входа в эмулятор светодиодной панели.

    Args:
        config_path: Путь к JSON-файлу раскладки.
    """
    app = LEDEmulator(config_path=config_path)
    app.mainloop()


if __name__ == "__main__":
    main()