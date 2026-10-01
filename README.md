# ledspector
Non-invasive, real-time computer vision system for automated server front-panel assembly QA and LED status monitoring.

## Windows (разработка и настройка)

`python run.py` — графическое меню: тест камеры, разметка ROI мышью, анализ в окне, эмулятор светодиодной панели.

## Linux-сервер (headless, без графики)

Настройка и наблюдение ведутся в консоли: `python3 run_linux.py`.

```
python3 run_linux.py check      # самопроверка окружения и камеры
python3 run_linux.py grid       # снимок с координатной сеткой для разметки LED
python3 run_linux.py exposure   # подбор ручной экспозиции камеры
python3 run_linux.py monitor    # таблица состояний в реальном времени
python3 run_linux.py log        # только изменения состояний, для длительного наблюдения
```

Подробности, установка, разбор конфигурации и разметки — в [docs/linux.md](docs/linux.md).

Проверка логики без камеры: `python3 test_headless.py`.