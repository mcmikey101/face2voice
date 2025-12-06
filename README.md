# Face2Voice

Инструмент для сопоставления изображений лиц с голосовыми характеристиками.

## Установка проекта

1) Создать и активировать виртуальное окружение python 3.10:

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

3) Установить папку с моделями и внести в папку face2voice

https://drive.google.com/drive/folders/1gROmF3W4diL1dotvtP9nQi29DK0_6f3y?usp=drive_link

4) Установить любым способом ffmpeg
https://www.wikihow.com/Install-FFmpeg-on-Windows

Запустить тестовую генерацию можно командой

```bash
python -m face2voice.inference.inference
```

## Запуск проекта

1) Активация сервера

```bash
uvicorn backend.main:app --reload
```

2) Запуск файла с фронтендом frontend/index.html
