"""
Скрипт для выборки и транскрипции аудио файлов из VoxCeleb датасета с прогресс-баром.

Выбирает по 5-6 случайных WAV-файлов на каждого спикера (id10270-10309),
транскрибирует их с помощью Whisper и сохраняет результаты в CSV.
"""

import os
import random
import csv
import warnings
import time
from pathlib import Path
from typing import List, Optional, Dict, Any
import argparse
from tqdm import tqdm

# Подавляем предупреждение о symlinks на Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
warnings.filterwarnings("ignore", message=".*symlinks.*")

try:
    from faster_whisper import WhisperModel
except ImportError:
    print("Установите faster-whisper: pip install faster-whisper")
    raise

def load_speaker_metadata(meta_csv_path: Path) -> dict:
    """Загружает метаданные спикеров из CSV файла и возвращает словарь {speaker_id: {name, gender, nationality}}."""
    speaker_map = {}
    if not meta_csv_path.exists():
        print(f"⚠️  Файл метаданных {meta_csv_path} не найден. Метаданные спикеров не будут добавлены.")
        return speaker_map
    
    try:
        with open(meta_csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                speaker_id = row.get("VoxCeleb1 ID", "").strip()
                speaker_name = row.get("VGGFace1 ID", "").strip()
                gender = row.get("Gender", "").strip()
                nationality = row.get("Nationality", "").strip()
                if speaker_id:
                    speaker_map[speaker_id] = {
                        "name": speaker_name,
                        "gender": gender,
                        "nationality": nationality
                    }
        print(f"✅ Загружено {len(speaker_map)} записей из метаданных")
    except Exception as e:
        print(f"⚠️  Ошибка при загрузке метаданных: {e}. Метаданные спикеров не будут добавлены.")
    
    return speaker_map


def find_wav_files(speaker_dir: Path) -> List[Path]:
    """Рекурсивно находит все WAV файлов в директории спикера."""
    return [wav_file for wav_file in speaker_dir.rglob("*.wav") if wav_file.is_file()]


def select_random_files(wav_files: List[Path], num_files: int = 5) -> List[Path]:
    """Выбирает случайные файлы из списка."""
    return wav_files if len(wav_files) <= num_files else random.sample(wav_files, num_files)


def transcribe_audio(model: WhisperModel, audio_path: Path, language: str = "en") -> str:
    """Транскрибирует аудио файл с помощью Whisper."""
    try:
        segments, info = model.transcribe(
            str(audio_path),
            language=language,
            beam_size=5,
            vad_filter=True
        )
        transcript = " ".join([segment.text for segment in segments])
        return transcript.strip()
    except Exception as e:
        print(f"Ошибка при транскрипции {audio_path}: {e}")
        return ""


def load_model_with_progress(model_size: str, device: str = "cpu", cache_dir: str = "./checkpoints/whisper"):
    """Загружает модель с индикацией прогресса и таймером."""
    start_time = time.time()
    
    print(f"🔄 Загрузка модели Whisper ({model_size})...")
    print("   ⏳ Это может занять 5-15 минут при первом запуске (скачивание модели)")
    print("   💡 При следующих запусках будет быстрее (модель уже в кеше)")
    print()
    
    # Создаем директорию для кеша
    os.makedirs(cache_dir, exist_ok=True)
    
    # Проверяем, есть ли модель в кеше
    model_name_map = {
        "tiny": "faster-whisper-tiny",
        "base": "faster-whisper-base",
        "small": "faster-whisper-small",
        "medium": "faster-whisper-medium",
        "large": "faster-whisper-large-v2",
        "large-v2": "faster-whisper-large-v2",
        "large-v3": "faster-whisper-large-v3"
    }
    
    model_cache_name = model_name_map.get(model_size, f"faster-whisper-{model_size}")
    cache_path = os.path.join(cache_dir, model_cache_name)
    
    if os.path.exists(cache_path):
        print(f"   ✅ Модель найдена в кеше: {cache_path}")
        print("   ⚡ Загрузка из кеша (должно быть быстро)...")
    else:
        print(f"   📥 Модель не найдена в кеше, начинаем скачивание...")
        print(f"   📦 Размер модели ~{get_model_size_mb(model_size)} МБ")
    
    try:
        # Показываем прогресс каждые 5 секунд
        print("   ⏳ Идет загрузка/инициализация модели...")
        
        # Загружаем модель
        model = WhisperModel(
            model_size,
            device=device,
            compute_type="int8",
            download_root=cache_dir,
            local_files_only=False  # Разрешить скачивание
        )
        
        elapsed = time.time() - start_time
        print(f"   ✅ Модель загружена успешно за {elapsed:.1f} секунд!")
        return model
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"   ❌ Ошибка загрузки после {elapsed:.1f}с: {e}")
        print("   💡 Попробуйте:")
        print("      - Проверить интернет-соединение")
        print("      - Использовать меньшую модель: --model tiny")
        print("      - Проверить свободное место на диске")
        raise


def get_model_size_mb(model_size: str) -> str:
    """Возвращает примерный размер модели в МБ."""
    sizes = {
        "tiny": "75",
        "base": "150",
        "small": "500",
        "medium": "1500",
        "large": "3000",
        "large-v2": "3000",
        "large-v3": "3000"
    }
    return sizes.get(model_size, "?")


def process_speakers(
    wav_base_dir: Path,
    output_csv: Path,
    speaker_ids: List[str],
    samples_per_speaker: int = 5,
    model_size: str = "base",
    device: str = "cpu",
    language: str = "en",
    speaker_metadata: Optional[Dict[str, Any]] = None,
    cache_dir: str = "./checkpoints/whisper"
):
    """Обрабатывает всех спикеров и создает CSV с транскрипциями с прогресс-баром."""
    
    if speaker_metadata is None:
        speaker_metadata = {}
    
    # Загружаем модель с прогресс-баром
    model = load_model_with_progress(model_size, device, cache_dir)
    print()
    
    # Собираем все задачи заранее для точного прогресс-бара
    print("📋 Сканирование файлов...")
    all_tasks = []
    for speaker_id in speaker_ids:
        speaker_dir = wav_base_dir / speaker_id
        if not speaker_dir.exists():
            continue
        
        wav_files = find_wav_files(speaker_dir)
        if not wav_files:
            continue
        
        selected_files = select_random_files(wav_files, samples_per_speaker)
        for audio_path in selected_files:
            relative_path = audio_path.relative_to(wav_base_dir)
            all_tasks.append({
                "speaker_id": speaker_id,
                "audio_path": audio_path,
                "relative_path": str(relative_path).replace("\\", "/")
            })
    
    total_files = len(all_tasks)
    print(f"✅ Найдено {total_files} файлов для транскрипции\n")
    
    if total_files == 0:
        print("⚠️  Не найдено файлов для обработки!")
        return
    
    # Обрабатываем с прогресс-баром и таймером
    results = []
    start_time = time.time()
    
    print("🎙️  Начинаем транскрипцию...\n")
    
    for task in tqdm(all_tasks, desc="Транскрипция", unit="файл", ncols=100):
        transcript = transcribe_audio(model, task["audio_path"], language)
        if transcript:
            speaker_info = speaker_metadata.get(task["speaker_id"], {})
            # Обработка нового формата (dict) и старого формата (str) для обратной совместимости
            if isinstance(speaker_info, dict):
                speaker_name = speaker_info.get("name", "")
                gender = speaker_info.get("gender", "")
                nationality = speaker_info.get("nationality", "")
            else:
                # Старый формат: speaker_metadata[speaker_id] = "name"
                speaker_name = speaker_info if isinstance(speaker_info, str) else ""
                gender = ""
                nationality = ""
            results.append({
                "speaker_id": task["speaker_id"],
                "speaker_name": speaker_name,
                "gender": gender,
                "nationality": nationality,
                "audio_path": task["relative_path"],
                "transcript": transcript
            })
        
        # Промежуточное сохранение каждые 10 файлов
        if len(results) % 10 == 0 and len(results) > 0:
            save_results(output_csv, results)
    
    # Финальное сохранение
    save_results(output_csv, results)
    
    elapsed = time.time() - start_time
    avg_time = elapsed / total_files if total_files > 0 else 0
    
    print(f"\n✅ Обработано {len(results)}/{total_files} файлов")
    print(f"⏱️  Время транскрипции: {elapsed:.1f}с ({elapsed/60:.1f} мин)")
    print(f"📊 Среднее время на файл: {avg_time:.1f}с")
    print(f"💾 Результаты сохранены в {output_csv}")


def save_results(output_csv: Path, results: List[dict]):
    """Сохраняет результаты в CSV файл."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["speaker_id", "speaker_name", "gender", "nationality", "audio_path", "transcript"])
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(description="Транскрипция аудио файлов из VoxCeleb")
    parser.add_argument("--wav_dir", type=str, default="data/wav", help="Базовая директория с папками спикеров")
    parser.add_argument("--output", type=str, default="transcripts.csv", help="Путь к выходному CSV файлу")
    parser.add_argument("--samples", type=int, default=8, help="Количество случайных файлов на спикера (по умолчанию: 8 для ~300 сэмплов)")
    parser.add_argument("--model", type=str, default="medium", choices=["tiny","base","small","medium","large","large-v2","large-v3"], help="Размер модели Whisper (по умолчанию: medium - хорошая точность и скорость)")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu","cuda"], help="Устройство для обработки")
    parser.add_argument("--language", type=str, default="en", help="Язык для транскрипции")
    parser.add_argument("--seed", type=int, default=42, help="Seed для случайного выбора файлов")
    parser.add_argument("--cache_dir", type=str, default="./checkpoints/whisper", help="Директория для кеша моделей")
    parser.add_argument("--meta_csv", type=str, default="data/vox1_meta.csv", help="Путь к CSV файлу с метаданными спикеров")
    args = parser.parse_args()
    
    random.seed(args.seed)
    speaker_ids = [f"id{i:05d}" for i in range(10270, 10310)]
    
    # Определяем пути
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    wav_base_dir = Path(args.wav_dir) if Path(args.wav_dir).is_absolute() else project_root / args.wav_dir
    
    # Определяем путь к файлу метаданных (относительно директории скрипта или абсолютный)
    meta_csv_path = script_dir / args.meta_csv if not Path(args.meta_csv).is_absolute() else Path(args.meta_csv)
    
    # Определяем путь к выходному CSV файлу (по умолчанию в папке scripts)
    if Path(args.output).is_absolute():
        output_csv = Path(args.output)
    else:
        # Если путь относительный, сохраняем в папке scripts
        output_csv = script_dir / args.output
    
    # Загружаем метаданные спикеров
    print("📋 Загрузка метаданных спикеров...")
    speaker_metadata = load_speaker_metadata(meta_csv_path)
    print()
    
    if not wav_base_dir.exists():
        print(f"Ошибка: директория {wav_base_dir} не найдена")
        return
    
    process_speakers(
        wav_base_dir=wav_base_dir,
        output_csv=output_csv,
        speaker_ids=speaker_ids,
        samples_per_speaker=args.samples,
        model_size=args.model,
        device=args.device,
        language=args.language,
        speaker_metadata=speaker_metadata,
        cache_dir=args.cache_dir
    )


if __name__ == "__main__":
    main()