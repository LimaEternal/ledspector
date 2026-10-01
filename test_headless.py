"""Проверка headless-модулей без камеры: парсер конфига, сетка, анализ.

Запуск без внешних зависимостей и без реального устройства:

    python test_headless.py

Камера подменяется заглушкой с синтетическими кадрами, поэтому тест
проходит и на Windows, и на headless-сервере.
"""

import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.console_monitor import Monitor, crop_led
from src.grid_overlay import annotate_frame, draw_grid
from src.led_config import load_config, save_config, LedConfig, LED
from src.linux_camera import LinuxCamera, list_video_devices

fails = []


def check(name, cond, detail=""):
    if cond:
        print(f"  OK   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        fails.append(name)


print("== 1. парсер конфига ==")
cfg = load_config("config/leds.tsv")
check("панель из шапки", cfg.panel == "VEGMAN-R220-front", cfg.panel)
check("разрешение 1280x720", cfg.resolution == (1280, 720), cfg.resolution)
check("9 диодов", len(cfg.leds) == 9, len(cfg.leds))
check("первый PWR 482x231", cfg.leds[0].bbox == (482, 231, 5, 5), cfg.leds[0].bbox)
check("описание сохранено", cfg.leds[0].description == "питание", cfg.leds[0].description)
check("описание с пробелом", cfg.get("LED_3").description == "демо-диод 3", cfg.get("LED_3").description)
check("get() по id", cfg.get("NET") is not None)
check("get() отсутствующего -> None", cfg.get("NOPE") is None)
check("центр PWR", cfg.leds[0].center == (484, 233), cfg.leds[0].center)

print("== 2. устойчивость к мусору ==")
bad = """panel: test
resolution: 640 480
PWR 10 20 5 5 норм
мусор
LED_B 1 2
DUP 1 1 1 1 первый
DUP 2 2 2 2 второй
NEG 5 5 0 5 нулевой_размер
TXT 5 5 abc 5 не_числа
OK2 100 200
"""
with tempfile.TemporaryDirectory() as d:
    p = Path(d) / "leds.tsv"
    p.write_text(bad, encoding="utf-8")
    c2 = load_config(str(p))
    # LED_B 1 2 валиден (id + x + y, w/h по умолчанию). Мусор/NEG/TXT/DUP-дубль отброшены.
    check("остались только валидные", len(c2.leds) == 4, [x.id for x in c2.leds])
    check("валидные id", [x.id for x in c2.leds] == ["PWR", "LED_B", "DUP", "OK2"], [x.id for x in c2.leds])
    check("DUP оставлен первый", c2.get("DUP").x == 1)
    check("W/H по умолчанию 5", c2.get("OK2").bbox == (100, 200, 5, 5), c2.get("OK2").bbox)
    check("разрешение распарсено", c2.resolution == (640, 480))

print("== 3. round-trip сохранения ==")
with tempfile.TemporaryDirectory() as d:
    p = str(Path(d) / "out.tsv")
    save_config(cfg, p)
    c3 = load_config(p)
    check("панель сохранена", c3.panel == cfg.panel)
    check("разрешение сохранено", c3.resolution == cfg.resolution)
    check("все диоды сохранены", [(x.id, x.bbox) for x in c3.leds] == [(x.id, x.bbox) for x in cfg.leds])
    check("описания сохранены", [x.description for x in c3.leds] == [x.description for x in cfg.leds])

print("== 4. проверка разрешения ==")
try:
    cfg.check_resolution(480, 320)
    check("должна быть ошибка на 480x320", False)
except ValueError as e:
    check("ошибка на другом разрешении", "не совпадает" in str(e))
cfg.check_resolution(1280, 720)
check("1280x720 проходит", True)

print("== 5. сетка на синтетическом кадре 1280x720 ==")
frame = np.full((720, 1280, 3), 90, dtype=np.uint8)
cv2.rectangle(frame, (100, 200), (700, 400), (170, 170, 170), -1)
for cx, cy, col in [(482, 231, (0, 255, 0)), (512, 231, (0, 0, 255)), (540, 231, (0, 200, 255))]:
    cv2.circle(frame, (cx, cy), 3, col, -1)

grid = draw_grid(frame, minor_step=10, major_step=50)
check("размер кадра не изменился", grid.shape == frame.shape, grid.shape)
check("исходный кадр не мутирован", np.array_equal(frame[0, 0], np.full(3, 90, np.uint8)))
check("сетка что-то нарисовала", not np.array_equal(grid, frame))
check("подписи различают кадр", np.count_nonzero(np.any(grid != frame, axis=2)) > 5000)

# линии ровно там, где заказано
col_line = np.all(grid[300, :, :] == np.all(grid[300, :, :]) * np.ones(3, np.uint8) * 0 + grid[300, :, :], axis=-1)
x_diff = np.count_nonzero(np.any(grid[300, :] != frame[300, :], axis=1))
check("много вертикальных линий", x_diff > 25, x_diff)

annotated = annotate_frame(frame, cfg.leds, with_grid=True)
check("annotate с LED работает", annotated.shape == frame.shape)

Path("junk").mkdir(exist_ok=True)
cv2.imwrite("junk/test_grid.png", annotated)
print("  -> junk/test_grid.png (сетка + LED на синтетическом кадре)")

print("== 5b. регрессия: сетка не затемняет фон ==")
# Оверлей чёрный вне линий; наивный addWeighted гасил бы весь кадр.
flat = np.full((720, 1280, 3), 90, np.uint8)
g2 = draw_grid(flat, minor_step=10, major_step=50)
check("фон между линиями не изменён", np.array_equal(g2[305, 35], flat[305, 35]),
      f"{flat[305,35].tolist()} -> {g2[305,35].tolist()}")
row = np.where(np.any(g2[305] != flat[305], axis=1))[0]
check("линии только на кратных 10", set(int(x) for x in row) == set(range(0, 1280, 10)))
check("крупная линия ярче мелкой", int(g2[305, 50, 1]) > int(g2[305, 30, 1]))
# Ниже 20px лежит только сетка: выше — рамка с подписями осей, буквы
# которой попадают на этот же столбец и не являются линиями.
col = np.where(np.any(g2[20:, 305] != flat[20:, 305], axis=1))[0] + 20
check("горизонтальные линии на кратных 10",
      set(int(y) for y in col) == set(range(20, 720, 10)))

print("== 6. crop_led ==")
led_in = LED("in", 100, 100, 5, 5)
crop = crop_led(frame, led_in)
check("внутренний ROI 5x5", crop.shape == (5, 5, 3), crop.shape)
led_edge = LED("edge", 1278, 718, 5, 5)
crop = crop_led(frame, led_edge)
check("ROI у края обрезан", crop.shape == (2, 2, 3), crop.shape)
led_out = LED("out", 5000, 5000, 5, 5)
crop = crop_led(frame, led_out)
check("вне кадра -> пустой", crop.size == 0)

print("== 7. Monitor на синтетическом камере ==")


class FakeCamera:
    actual_width = 1280
    actual_height = 720
    dropped_frames = 0

    def __init__(self, frames):
        self.frames = list(frames)
        self.i = 0

    def read_frame(self):
        f = self.frames[self.i % len(self.frames)]
        self.i += 1
        return True, f.copy()

    def get_exposure(self):
        return 500

    def get_auto_exposure(self):
        return "manual"


def solid(color_bgr):
    f = np.zeros((720, 1280, 3), dtype=np.uint8)
    cv2.rectangle(f, (478, 227), (486, 235), color_bgr, -1)
    return f


mon_cfg = LedConfig(panel="t", resolution=(1280, 720), leds=[LED("PWR", 478, 227, 9, 9)])
green = solid((0, 200, 0))
mon = Monitor(mon_cfg, FakeCamera([green for _ in range(40)]))
last = None
for _ in range(30):
    last = mon.step()
r = last["PWR"]
check("зелёный определён как GREEN", r.color == "GREEN", r.color)
check("is_on True", r.is_on is True)
check("состояние дошло до SOLID_ON", r.state == "SOLID_ON", r.state)
check("яркость ~200", 195 < r.brightness < 205, r.brightness)

# BGR-порядок: красный = (0,0,200), синий = (200,0,0)
red = solid((0, 0, 200))
mon2 = Monitor(mon_cfg, FakeCamera([red for _ in range(40)]))
for _ in range(30):
    last = mon2.step()
check("красный определён как RED", last["PWR"].color == "RED", last["PWR"].color)

blue = solid((200, 0, 0))
mon3 = Monitor(mon_cfg, FakeCamera([blue for _ in range(40)]))
for _ in range(30):
    last = mon3.step()
check("синий определён как BLUE", last["PWR"].color == "BLUE", last["PWR"].color)

off = np.zeros((720, 1280, 3), dtype=np.uint8)
mon4 = Monitor(mon_cfg, FakeCamera([off for _ in range(40)]))
for _ in range(30):
    last = mon4.step()
check("чёрный -> OFF", last["PWR"].state == "OFF", last["PWR"].state)

print("== 8. детекция смен состояний (с реальным временем) ==")
# Окно анализатора 2.0с. Чтобы зелёный вышел из окна, нужно >2.0с чёрных кадров.
# Шаг 0.02с: 30 зелёных (0.6с) + 110 чёрных (2.2с) -> в окне только чёрные -> OFF.
mon5 = Monitor(mon_cfg, FakeCamera([green] * 30 + [off] * 110))
for i in range(140):
    mon5.step()
    time.sleep(0.02)
check("зафиксированы смены", len(mon5.events) >= 1, len(mon5.events))
off_events = [e for e in mon5.events if e.new_state == "OFF"]
check("есть событие -> OFF", len(off_events) >= 1, [f"{e.old_state}->{e.new_state}" for e in mon5.events])
if mon5.events:
    e = mon5.events[0]
    check("формат события", e.format().startswith("["), e.format())

print("== 9. формат таблицы ==")
readings = {"PWR": __import__("src.console_monitor", fromlist=["LedReading"]).LedReading(
    "PWR", "BLINK_1HZ", 1.02, "RED", 190.4, True)}
lines = mon5.format_table(readings, tty=False)
check("таблица без TTY без ANSI", all("\033" not in ln for ln in lines))
check("есть заголовок", any("LEDSpector" in ln for ln in lines))
check("есть ID", any("PWR" in ln for ln in lines))
lines_tty = mon5.format_table(readings, tty=True)
check("таблица с ANSI при TTY", any("\033" in ln for ln in lines_tty))
ev = mon5.format_events(tty=False)
check("события форматируются", len(ev) > 1)

print("== 10. потеря кадра не роняет цикл ==")


class DeadCamera(FakeCamera):
    def read_frame(self):
        return False, None


mon6 = Monitor(mon_cfg, DeadCamera([]))
check("step() -> None на потере", mon6.step() is None)
check("счётчик потерянных вырос", mon6.dropped == 1)

print("== 11. LinuxCamera без устройства ==")
devices = list_video_devices()
check("на Windows /dev/video* пуст (ожидаемо)", devices == [], devices)
cam = LinuxCamera(width=1280, height=720)
check("device -> /dev/video0", cam.device == "/dev/video0", cam.device)
try:
    cam.start()
    check("start() должен упасть на Windows", False)
except RuntimeError as e:
    check("start() падает с понятной ошибкой", "Не удалось открыть" in str(e))
check("clamp 50 -> 80", cam.clamp_exposure(50) == 80)
check("clamp 999999 -> 100000", cam.clamp_exposure(999999) == 100000)
check("clamp 500 -> 500", cam.clamp_exposure(500) == 500)

print("== 12. сохранение аннотированного кадра ==")
with tempfile.TemporaryDirectory() as d:
    out = str(Path(d) / "nested" / "shot.png")
    mon7 = Monitor(mon_cfg, FakeCamera([green]))
    check("до кадра сохранять нечего", mon7.save_annotated(out) is False)
    mon7.step()
    check("save_annotated вернул True", mon7.save_annotated(out) is True)
    check("файл создан", Path(out).is_file())
    saved = cv2.imread(out)
    check("размер совпадает с кадром", saved.shape == (720, 1280, 3), None if saved is None else saved.shape)

    # Сигнальный путь: флаг выставляется обработчиком, кадр пишется в step().
    out2 = str(Path(d) / "sig.png")
    mon7._snapshot_requested = out2
    mon7._snapshot_with_grid = False
    mon7.step()
    check("сигнальный снимок сохранён", Path(out2).is_file())
    check("флаг сброшен после снимка", mon7._snapshot_requested is None)

print()
if fails:
    print(f"ПРОВАЛЕНО {len(fails)}: {fails}")
    sys.exit(1)
print("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")