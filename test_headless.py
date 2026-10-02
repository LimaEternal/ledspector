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

from src.color_detector import ColorDetector
from src.console_monitor import Monitor, crop_led
from src.grid_overlay import annotate_frame, draw_grid
from src.led_analyzer import (
    analyze_frame,
    draw_measurements,
    find_led_blobs,
    format_config_lines,
    is_confusable,
    measure_led,
)
from src.led_config import load_config, save_config, LedConfig, LED
from src.linux_camera import (
    DEFAULT_FOURCC,
    EXPOSURE_CONTROL,
    LinuxCamera,
    detect_degraded_controls,
    list_video_devices,
    parse_controls,
    parse_formats_ext,
    resolve_auto_exposure_values,
)

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
check("8 диодов в конфиге", len(cfg.leds) == 8, len(cfg.leds))
check("первый LED_1 на сдвинутой точке",
      cfg.leds[0].bbox == (566, 389, 5, 5), cfg.leds[0].bbox)
check("описание сохранено", cfg.leds[0].description == "синий (центр 568,447)",
      cfg.leds[0].description)
check("описание с пробелом", cfg.get("LED_3").description == "красный (центр 897,559)",
      cfg.get("LED_3").description)
check("get() по id", cfg.get("LED_5") is not None)
check("get() отсутствующего -> None", cfg.get("NOPE") is None)
check("центр первого диода", cfg.leds[0].center == (568, 391), cfg.leds[0].center)
check("все точки в границах кадра",
      all(0 <= led.x < 1280 and 0 <= led.y < 720 for led in cfg.leds))

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

print("== 3b. round-trip с длинными ID ==")
# LED_10 не помещается в колонку шириной 6: при фиксированной ширине
# координаты прилипали к ID («LED_10886») и строка переставала читаться.
long_cfg = LedConfig(
    panel="t",
    resolution=(1280, 720),
    leds=[
        LED("A", 10, 20, 5, 5, "короткий"),
        LED("LED_10", 886, 29, 5, 5, "красный (центр 908,11)"),
        LED("ОЧЕНЬ_ДЛИННЫЙ_ID", 100, 200, 5, 5, "с пробелами и скобками"),
    ],
)
with tempfile.TemporaryDirectory() as d:
    p = str(Path(d) / "long.tsv")
    save_config(long_cfg, p)
    back = load_config(p)
    check("длинные ID переживают сохранение", len(back.leds) == 3, [x.id for x in back.leds])
    check("ID не склеиваются с координатами",
          [x.bbox for x in back.leds] == [(10, 20, 5, 5), (886, 29, 5, 5), (100, 200, 5, 5)],
          [x.bbox for x in back.leds])
    check("описания со скобками сохранены",
          [x.description for x in back.leds][1] == "красный (центр 908,11)",
          [x.description for x in back.leds])

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
check("подписи различают кадр", np.count_nonzero(np.any(grid != frame, axis=2)) > 1000)

# Линии сетки идут по кратным 10, а фон между ними остаётся нетронутым.
# Раньше здесь стояла проверка на большую площадь изменений — она ловила
# артефакт затемнения фона от наивного addWeighted, который уже исправлен.
row305 = np.where(np.any(grid[305] != frame[305], axis=1))[0]
check("вертикальные линии только на кратных 10",
      set(int(x) for x in row305) == set(range(0, 1280, 10)))

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

print("== 13. разбор реального вывода v4l2-ctl (VEGMAN-R220) ==")
FORMATS_EXT = """ioctl: VIDIOC_ENUM_FMT
        Type: Video Capture

        [0]: 'MJPG' (Motion-JPEG, compressed)
                Size: Discrete 480x320
                        Interval: Discrete 0.040s (25.000 fps)
                Size: Discrete 640x480
                        Interval: Discrete 0.040s (25.000 fps)
                Size: Discrete 1280x720
                        Interval: Discrete 0.050s (20.000 fps)
        [1]: 'YUYV' (YUYV 4:2:2)
                Size: Discrete 320x240
                        Interval: Discrete 0.040s (25.000 fps)
                Size: Discrete 640x480
                        Interval: Discrete 0.100s (10.000 fps)
"""
CTRLS = """User Controls

                     brightness 0x00980900 (int)    : min=-127 max=127 step=1 default=0 value=0 flags=has-min-max
                      contrast 0x00980901 (int)    : min=0 max=511 step=1 default=256 value=0 flags=has-min-max
                      saturation 0x00980902 (int)    : min=0 max=511 step=1 default=256 value=0 flags=has-min-max
                          gamma 0x00980910 (int)    : min=10 max=30 step=10 default=20 value=10 flags=has-min-max
                           gain 0x00980913 (int)    : min=1 max=7 step=1 default=4 value=4 flags=has-min-max
           white_balance_temperature 0x0098091a (int)    : min=0 max=6500 step=1 default=4500 value=4500 flags=has-min-max
                      sharpness 0x0098091b (int)    : min=0 max=256 step=1 default=128 value=0 flags=has-min-max

Camera Controls

                  auto_exposure 0x009a0901 (menu)   : min=0 max=3 default=3 value=1 (Manual Mode)
         exposure_time_absolute 0x009a0902 (int)    : min=80 max=100000 step=1 default=80 value=500 flags=has-min-max
"""
parsed = parse_formats_ext(FORMATS_EXT)
check("форматы разобраны", set(parsed) == {"MJPG", "YUYV"}, set(parsed))
check("MJPG 1280x720@20", (1280, 720, 20.0) in parsed["MJPG"], parsed.get("MJPG"))
check("MJPG 480x320@25", (480, 320, 25.0) in parsed["MJPG"])
check("MJPG 640x480@25", (640, 480, 25.0) in parsed["MJPG"])
check("YUYV 640x480 только @10", (640, 480, 10.0) in parsed["YUYV"], parsed.get("YUYV"))
check("YUYV не даёт 720p", not any(w == 1280 for w, _, _ in parsed["YUYV"]))
check(
    "720p достижим только через MJPG (причина выбора fourcc)",
    [f for f, s in parsed.items() if any(w == 1280 for w, _, _ in s)] == ["MJPG"],
)
check("FPS не завышен в 100 раз", all(fps < 100 for sizes in parsed.values() for _, _, fps in sizes))

auto = resolve_auto_exposure_values(CTRLS)
check("auto_exposure: manual = 1", auto["manual"] == 1, auto)
check("auto_exposure: текущее = 1", auto["current"] == 1, auto)
check("auto_exposure: тип (menu) не спутан с подписью", auto["auto"] is None, auto)
check("пустой formats не падает", parse_formats_ext("") == {})
check("None не падает", parse_formats_ext(None) == {})
check("нет auto_exposure -> None",
      resolve_auto_exposure_values("brightness 0x1 (int) : value=5")["manual"] is None)
check("обратная нумерация камеры",
      resolve_auto_exposure_values(
          "auto_exposure (menu) : min=0 max=3 default=1 value=3 (Aperture Priority Mode)"
      )["auto"] == 3)
check("Manual Mode при value=2",
      resolve_auto_exposure_values(
          "auto_exposure (menu) : min=0 max=3 default=3 value=2 (Manual Mode)"
      )["manual"] == 2)
check("имя контрола экспозиции = exposure_time_absolute",
      EXPOSURE_CONTROL == "exposure_time_absolute", EXPOSURE_CONTROL)
check("fourcc по умолчанию MJPG", cam.fourcc == "MJPG", cam.fourcc)
check("fourcc по умолчанию хранится в конструкторе",
      LinuxCamera(width=1280, height=720).fourcc == "MJPG")

print("== 14. детектор пересвета ==")
det = ColorDetector()


def flat(bgr):
    """Однородный кроп заданного цвета."""
    return np.full((9, 9, 3), bgr, np.uint8)


r = det.detect(flat((0, 200, 0)))
check("чистый зелёный -> GREEN", r["color"] == "GREEN", r["color"])
check("чистый зелёный не пересвечен", r["blown_out"] is False)
check("пересвет 0% у чистого", r["clipped_fraction"] == 0.0)

r = det.detect(flat((250, 255, 253)))
check("срезанные каналы -> BLOWN_OUT", r["color"] == "BLOWN_OUT", r["color"])
check("blown_out True", r["blown_out"] is True)
check("clipped_fraction ~100%", r["clipped_fraction"] > 0.99, r["clipped_fraction"])
check("пересвет НЕ путается с WHITE", r["color"] != "WHITE")

# Частичный пересвет: 2 из 9 строк срезаны -> ниже порога 30%.
part = np.full((9, 9, 3), (0, 200, 0), np.uint8)
part[0:2, :, :] = (250, 255, 253)
r = det.detect(part)
check("частичный пересвет (22%) не BLOWN", r["blown_out"] is False, r["clipped_fraction"])
check("частичный пересвет виден в метрике", 0.1 < r["clipped_fraction"] < 0.3, r["clipped_fraction"])
check("частичный пересвет сохраняет цвет", r["color"] == "GREEN", r["color"])

r = det.detect(flat((255, 255, 255)))
check("белый -> BLOWN_OUT (каналы срезаны)", r["color"] == "BLOWN_OUT", r["color"])
r = det.detect(np.zeros((9, 9, 3), np.uint8))
check("чёрный -> OFF", r["color"] == "OFF" and r["blown_out"] is False)
r = det.detect(np.array([], np.uint8))
check("пустой -> OFF, не падает", r["color"] == "OFF" and r["blown_out"] is False)
check("у результата есть hue/saturation", "hue" in r and "saturation" in r)

print("== 15. разбор контролов камеры ==")
controls = parse_controls(CTRLS)
# 7 из блока User Controls + 2 из Camera Controls.
check("найдены все контролы", len(controls) == 9, len(controls))
check("saturation прочитан как 0", controls["saturation"]["value"] == 0)
check("default saturation = 256", controls["saturation"]["default"] == 256)
check("gain прочитан", controls["gain"]["value"] == 4)
check("диапазон gain", (controls["gain"]["min"], controls["gain"]["max"]) == (1, 7))
check("пустой вывод не падает", parse_controls("") == {})

suspect = detect_degraded_controls(controls)
names = sorted(name for name, _v, _d in suspect)
check("saturation помечен как сброшенный", "saturation" in names, names)
check("contrast помечен как сброшенный", "contrast" in names, names)
check("gain не помечен", "gain" not in names)
check("здоровые контролы не ругаются",
      detect_degraded_controls({"saturation": {"value": 256, "default": 256}}) == [])

print("== 16. подбор точки замера вне пересвета ==")


def glow(cx, cy, radius, color=(0, 200, 0), core_radius=18):
    """Синтетический диод: срезанное ядро и затухающий ореол.

    Ядро моделируется как ровно 255 в своём канале — так ведёт себя
    реальный горевший диод, когда поток всех трёх каналов упирается в
    предел АЦП. Ореол падает линейно и держит насыщенность.
    """
    frame = np.full((720, 1280, 3), 15, np.uint8)
    yy, xx = np.mgrid[0:720, 0:1280]
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)

    # Ореол: от полной яркости у ядра к фону на радиусе radius.
    halo = np.clip(1.0 - (dist - core_radius) / float(radius - core_radius), 0, 1)
    for ch in range(3):
        frame[:, :, ch] = np.clip(
            15 + 240 * halo * (color[ch] / 255.0), 0, 255
        ).astype(np.uint8)

    # Ядро: каналы срезаны в 255 — радиусная маска с жёстким краем.
    core = dist <= core_radius
    for ch in range(3):
        if color[ch] > 0:
            frame[:, :, ch] = np.where(core, 255, frame[:, :, ch])
        else:
            frame[:, :, ch] = np.where(core, 15, frame[:, :, ch])

    return frame


glow_frame = glow(600, 400, 90)
blobs = find_led_blobs(glow_frame)
check("диод найден", len(blobs) >= 1, len(blobs))
# Ярче всего у центра, поэтому первым идёт именно диод, а не ореол.
core = blobs[0]
check("найден именно диод (максимум яркости)", core.peak_v >= 250, core.peak_v)
check("центр совпал", abs(core.cx - 600) < 6 and abs(core.cy - 400) < 6,
      (core.cx, core.cy))

m = measure_led(glow_frame, core.cx, core.cy)
check("центр пересвечен", m.center_clipped > 0.5, m.center_clipped)
check("вердикт MOVE", m.status == "MOVE", m.status)
check("точка подобрана", m.suggested is not None, m.detail)
if m.suggested:
    x, y, w, h = m.suggested
    check("точка не в центре", abs((x + w // 2) - 600) + abs((y + h // 2) - 400) > 10)
    crop = glow_frame[y:y + h, x:x + w]
    verdict = det.detect(crop)
    check("в новой точке нет пересвета", verdict["blown_out"] is False, verdict["color"])
    check("в новой точке есть насыщенность", verdict["saturation"] > 60, verdict["saturation"])
    check("в новой точке читается GREEN", verdict["color"] == "GREEN", verdict["color"])

check("профиль непустой", len(m.profile) > 3, len(m.profile))
check("профиль убывает по радиусу",
      all(m.profile[i][1] >= m.profile[i + 1][1] for i in range(len(m.profile) - 1))
      or len(m.profile) < 2)

# Яркое срезанное ядро диаметром в несколько пикселей: ореола за ним
# нет, уходить некуда -> NO_HALO. Это и есть настоящий признак того, что
# одной подстройкой не обойтись.
pin_frame = np.full((720, 1280, 3), 10, np.uint8)
cv2.circle(pin_frame, (600, 400), 3, (0, 60, 0), -1)
cv2.circle(pin_frame, (600, 400), 2, (255, 255, 255), -1)
m2 = measure_led(pin_frame, 600, 400)
check("срезанное ядро без ореола -> NO_HALO", m2.status == "NO_HALO", m2.status)
check("NO_HALO объясняет причину", "gain" in m2.detail or "оптик" in m2.detail, m2.detail)
check("NO_HALO не предлагает точку", m2.suggested is None)
check("NO_HALO попадает в needs_move=False", m2.needs_move is False)

# Тусклое, но не срезанное пятно: точку найти можно, статус OK.
dim_frame = np.full((720, 1280, 3), 10, np.uint8)
cv2.circle(dim_frame, (600, 400), 5, (0, 60, 0), -1)
m_dim = measure_led(dim_frame, 600, 400)
check("тусклое без пересвета -> OK", m_dim.status == "OK", m_dim.status)

check("пустой кадр не падает", find_led_blobs(np.array([], np.uint8)) == [])
check("тёмный кадр без диодов", find_led_blobs(np.zeros((720, 1280, 3), np.uint8)) == [])

print("== 17. формат вывода analyze ==")
check("строки конфига генерируются", len(format_config_lines([m])) == 1, format_config_lines([m]))
lines = format_config_lines([m, m2])
check("NO_HALO пропускается", len(lines) == 1, lines)
annotated = draw_measurements(glow_frame, [m, m2])
check("разметка строится", annotated.shape == glow_frame.shape)
check("разметка не мутирует кадр", not np.array_equal(annotated, glow_frame))

print("== 18. сверка оттенков ==")
check("одинаковые оттенки считаются одним цветом", is_confusable(60.0, [62.0]))
check("разные оттенки не путаются", not is_confusable(60.0, [120.0]))
check("сравнение по кругу: 179 и 1 это красный", is_confusable(179.0, [1.0]))
check("пустой список эталонов не отвергает", not is_confusable(60.0, []))

print()
if fails:
    print(f"ПРОВАЛЕНО {len(fails)}: {fails}")
    sys.exit(1)
print("ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ")