# Face2Voice

Инструмент для сопоставления изображений лиц с голосовыми характеристиками.

## Установка проекта

1) Создать и активировать виртуальное окружение python 3.10:

```bash
python3 -m venv venv
source venv/bin/activate
```

2) Установить зависимости:

```bash
pip install -r requirements.txt
```

## Установка dlib

### Linux

Установить системные зависимости:

```bash
sudo apt-get update
sudo apt-get install build-essential cmake libopenblas-dev liblapack-dev libx11-dev
```

Установить dlib:

```bash
pip install dlib
```

### macOS

Установить инструменты сборки:

```bash
xcode-select --install
brew install cmake
```

Установить dlib:

```bash
pip install dlib
```

### Windows

Установить CMake и компилятор MSVC (Visual Studio Build Tools).
Установить dlib:

```bash
pip install dlib
```

## Запуск проекта

```bash
python main.py
```
