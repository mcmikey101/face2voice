"""
Скрипт для генерации аудио файлов с помощью XTTS-v2 модели.

Читает транскрипции из transcripts.csv, находит соответствующие лица в data/voxceleb_processed
и генерирует аудио файлы с помощью XTTS-v2 модели из checkpoints/xtts.
"""

import os
import csv
import argparse
from pathlib import Path
from typing import Optional
import time
from tqdm import tqdm
import torch
import soundfile as sf

try:
    from TTS.api import TTS
except ImportError:
    print("[WARNING] Установите библиотеку TTS: pip install TTS")
    raise


def load_xtts_model(model_path: Path, device: str = "cpu"):
    """Загружает XTTS-v2 модель из локальной директории."""
    print(f"[INFO] Загрузка XTTS-v2 модели из {model_path}...")
    start_time = time.time()
    
    # Проверяем, существует ли локальная модель
    if not model_path.exists():
        raise FileNotFoundError(f"Директория модели не найдена: {model_path}")
    
    if not (model_path / "model.pth").exists():
        raise FileNotFoundError(f"Файл модели не найден: {model_path / 'model.pth'}")
    
    if not (model_path / "config.json").exists():
        raise FileNotFoundError(f"Файл конфигурации не найден: {model_path / 'config.json'}")
    
    try:
        # Загружаем локальную модель напрямую
        from TTS.tts.configs.xtts_config import XttsConfig
        from TTS.tts.models.xtts import Xtts
        
        print("   [INFO] Загружаем конфигурацию...")
        config = XttsConfig()
        config.load_json(str(model_path / "config.json"))
        
        print("   [INFO] Инициализируем модель...")
        model = Xtts.init_from_config(config)
        
        print("   [INFO] Загружаем веса модели...")
        # Исправляем проблему с PyTorch 2.6 и weights_only
        # Патчим torch.load чтобы всегда использовать weights_only=False
        original_torch_load = torch.load
        def patched_torch_load(*args, **kwargs):
            kwargs['weights_only'] = False
            return original_torch_load(*args, **kwargs)
        
        # Временно заменяем torch.load
        torch.load = patched_torch_load
        
        try:
            model.load_checkpoint(
                config, 
                checkpoint_dir=str(model_path), 
                eval=True,
                use_deepspeed=False
            )
        finally:
            # Восстанавливаем оригинальный torch.load
            torch.load = original_torch_load
        
        if device == "cuda" and torch.cuda.is_available():
            model = model.to(device)
            print("   [INFO] Модель загружена на GPU")
        else:
            model = model.to("cpu")
            print("   [INFO] Модель загружена на CPU")
        
        elapsed = time.time() - start_time
        print(f"[INFO] Локальная модель загружена за {elapsed:.1f} секунд")
        
        # Создаем обертку для совместимости
        class XTTSWrapper:
            def __init__(self, model, config, device):
                self.model = model
                self.config = config
                self.device = device
            
            def tts_to_file(self, text, file_path, speaker_wav=None, language="en"):
                """Генерирует аудио и сохраняет в файл."""
                try:
                    # Генерируем аудио
                    if speaker_wav and os.path.exists(speaker_wav):
                        # С референсным аудио
                        wav = self.model.synthesize(
                            text,
                            self.config,
                            speaker_wav=speaker_wav,
                            language=language,
                            gpt_cond_len=30,
                            gpt_cond_chunk_len=4,
                            max_ref_len=30,
                            sound_norm_refs=False
                        )
                    else:
                        # Без референсного аудио (используем предобученного спикера)
                        # В новых версиях XTTS speaker_wav обязателен, используем None
                        try:
                            wav = self.model.synthesize(
                                text,
                                self.config,
                                speaker_wav=None,  # None для использования предобученного спикера
                                language=language,
                                gpt_cond_len=30,
                                gpt_cond_chunk_len=4,
                                max_ref_len=30,
                                sound_norm_refs=False
                            )
                        except TypeError:
                            # Если None не работает, пробуем пустую строку
                            wav = self.model.synthesize(
                                text,
                                self.config,
                                speaker_wav="",  # Пустая строка
                                language=language,
                                gpt_cond_len=30,
                                gpt_cond_chunk_len=4,
                                max_ref_len=30,
                                sound_norm_refs=False
                            )
                    
                    # Обрабатываем результат
                    # wav может быть словарем с ключом "wav" или просто массивом numpy
                    if isinstance(wav, dict):
                        audio_data = wav.get("wav", wav.get("audio", None))
                        if audio_data is None:
                            # Пробуем найти любой массив в словаре
                            for key, value in wav.items():
                                if hasattr(value, 'shape') or isinstance(value, (list, tuple)):
                                    audio_data = value
                                    break
                    else:
                        audio_data = wav
                    
                    if audio_data is None:
                        raise ValueError("Не удалось извлечь аудио данные из результата синтеза")
                    
                    # Преобразуем в numpy массив если нужно
                    if torch.is_tensor(audio_data):
                        audio_data = audio_data.cpu().numpy()
                    elif isinstance(audio_data, (list, tuple)):
                        # Преобразуем list/tuple в numpy массив
                        import numpy as np
                        audio_data = np.array(audio_data)
                    
                    # Убеждаемся, что это numpy массив с атрибутом shape
                    import numpy as np
                    if not isinstance(audio_data, np.ndarray):
                        audio_data = np.array(audio_data)
                    
                    # Убеждаемся, что это одномерный массив
                    if len(audio_data.shape) > 1:
                        audio_data = audio_data.squeeze()
                    
                    # Получаем sample rate
                    sample_rate = getattr(self.config.audio, 'output_sample_rate', 
                                        getattr(self.config.audio, 'sample_rate', 24000))
                    
                    # Сохраняем аудио
                    sf.write(file_path, audio_data, sample_rate)
                    return file_path
                except Exception as e:
                    print(f"   [ERROR] Ошибка в tts_to_file: {e}")
                    raise
        
        return XTTSWrapper(model, config, device)
        
    except Exception as e:
        print(f"[ERROR] Ошибка загрузки локальной модели: {e}")
        print("\n[INFO] Возможные решения:")
        print("   1. Убедитесь, что модель находится в правильной директории")
        print("   2. Проверьте, что все файлы модели присутствуют (model.pth, config.json)")
        print("   3. Если проблема с PyTorch 2.6, попробуйте:")
        print("      pip install torch==2.5.0")
        raise


def find_face_image(face_dir: Path, speaker_name: str) -> Optional[Path]:
    """Находит файл изображения лица по имени спикера."""
    # Пробуем разные варианты расширений
    for ext in [".jpg", ".jpeg", ".png"]:
        face_path = face_dir / f"{speaker_name}{ext}"
        if face_path.exists():
            return face_path
    
    return None


def generate_audio_with_xtts(
    tts,
    text: str,
    speaker_wav: Optional[str] = None,
    language: str = "en",
    output_path: Optional[str] = None
) -> Optional[str]:
    """
    Генерирует аудио с помощью XTTS-v2.
    
    Args:
        tts: Загруженная модель TTS
        text: Текст для синтеза
        speaker_wav: Путь к референсному аудио файлу (опционально)
        language: Язык синтеза
        output_path: Путь для сохранения аудио
    
    Returns:
        Путь к сгенерированному аудио файлу
    """
    if output_path is None:
        raise ValueError("output_path не может быть None")
    
    try:
        # XTTS-v2 может работать с референсным аудио или без него
        if speaker_wav and os.path.exists(speaker_wav):
            # Используем референсное аудио для клонирования голоса
            tts.tts_to_file(
                text=text,
                file_path=output_path,
                speaker_wav=speaker_wav,
                language=language
            )
        else:
            # Используем предобученного спикера (без референса)
            tts.tts_to_file(
                text=text,
                file_path=output_path,
                language=language
            )
        
        return output_path
    except Exception as e:
        print(f"[ERROR] Ошибка генерации аудио: {e}")
        return None


def process_transcripts(
    transcripts_csv: Path,
    face_dir: Path,
    xtts_model_path: Path,
    output_dir: Path,
    wav_base_dir: Optional[Path] = None,
    device: str = "cpu",
    language: str = "en"
):
    """
    Обрабатывает транскрипции и генерирует аудио файлы.
    
    Args:
        transcripts_csv: Путь к CSV файлу с транскрипциями
        face_dir: Директория с изображениями лиц
        xtts_model_path: Путь к модели XTTS-v2
        output_dir: Директория для сохранения сгенерированных аудио
        wav_base_dir: Базовая директория с оригинальными WAV файлами (для референсного аудио)
        device: Устройство для обработки (cpu/cuda)
        language: Язык синтеза
    """
    # Создаем выходную директорию
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Загружаем модель
    tts = load_xtts_model(xtts_model_path, device)
    print()
    
    # Читаем транскрипции
    print("📋 Загрузка транскрипций...")
    tasks = []
    with open(transcripts_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            speaker_name = row.get("speaker_name", "").strip()
            transcript = row.get("transcript", "").strip()
            audio_path = row.get("audio_path", "").strip()
            speaker_id = row.get("speaker_id", "").strip()
            
            if not speaker_name or not transcript:
                continue
            
            # Ищем изображение лица
            face_image = find_face_image(face_dir, speaker_name)
            
            # Ищем референсное аудио (если указана базовая директория)
            reference_audio = None
            if wav_base_dir and audio_path:
                ref_audio_path = wav_base_dir / audio_path
                if ref_audio_path.exists():
                    reference_audio = str(ref_audio_path)
            
            tasks.append({
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
                "transcript": transcript,
                "face_image": face_image,
                "reference_audio": reference_audio,
                "original_audio_path": audio_path
            })
    
    total_tasks = len(tasks)
    print(f"[INFO] Найдено {total_tasks} задач для обработки\n")
    
    if total_tasks == 0:
        print("[WARNING] Не найдено задач для обработки!")
        return
    
    # Обрабатываем с прогресс-баром
    results = []
    start_time = time.time()
    
    print("[INFO] Начинаем генерацию аудио...\n")
    
    for i, task in enumerate(tqdm(tasks, desc="Генерация аудио", unit="файл", ncols=100)):
        speaker_name = task["speaker_name"]
        transcript = task["transcript"]
        speaker_id = task["speaker_id"]
        original_audio_path = task["original_audio_path"]
        
        # Формируем имя выходного файла
        # Используем speaker_id и индекс для уникальности
        output_filename = f"{speaker_id}_{speaker_name}_{i:04d}.wav"
        output_path = output_dir / output_filename
        
        # Генерируем аудио
        # XTTS-v2 может работать с референсным аудио или без него
        reference_audio = task.get("reference_audio")
        
        try:
            generated_path = generate_audio_with_xtts(
                tts=tts,
                text=transcript,
                speaker_wav=reference_audio,
                language=language,
                output_path=str(output_path)
            )
            
            if generated_path and os.path.exists(generated_path):
                results.append({
                    "speaker_id": speaker_id,
                    "speaker_name": speaker_name,
                    "transcript": transcript,
                    "generated_audio": str(output_path),
                    "original_audio": original_audio_path,
                    "face_image": str(task["face_image"]) if task["face_image"] else None,
                    "reference_used": reference_audio is not None
                })
        except Exception as e:
            print(f"\n[WARNING] Ошибка при обработке {speaker_name}: {e}")
            continue
        
        # Промежуточное сохранение результатов каждые 10 файлов
        if len(results) % 10 == 0 and len(results) > 0:
            save_results_csv(output_dir / "generation_results.csv", results)
    
    # Финальное сохранение результатов
    save_results_csv(output_dir / "generation_results.csv", results)
    
    elapsed = time.time() - start_time
    avg_time = elapsed / total_tasks if total_tasks > 0 else 0
    
    print(f"\n[INFO] Обработано {len(results)}/{total_tasks} файлов")
    print(f"[INFO] Время генерации: {elapsed:.1f}с ({elapsed/60:.1f} мин)")
    print(f"[INFO] Среднее время на файл: {avg_time:.1f}с")
    print(f"[INFO] Результаты сохранены в {output_dir}")


def save_results_csv(output_csv: Path, results: list):
    """Сохраняет результаты генерации в CSV файл."""
    if not results:
        return
    
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    
    fieldnames = ["speaker_id", "speaker_name", "transcript", "generated_audio", 
                  "original_audio", "face_image", "reference_used"]
    
    file_exists = output_csv.exists()
    
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


def main():
    parser = argparse.ArgumentParser(description="Генерация аудио с помощью XTTS-v2")
    parser.add_argument(
        "--transcripts",
        type=str,
        default="data/transcripts.csv",
        help="Путь к CSV файлу с транскрипциями"
    )
    parser.add_argument(
        "--face_dir",
        type=str,
        default="data/voxceleb_processed",
        help="Директория с изображениями лиц"
    )
    parser.add_argument(
        "--xtts_model",
        type=str,
        default="checkpoints/xtts",
        help="Путь к модели XTTS-v2"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/generated_audio",
        help="Директория для сохранения сгенерированных аудио"
    )
    parser.add_argument(
        "--wav_dir",
        type=str,
        default="data/wav",
        help="Базовая директория с оригинальными WAV файлами (для референсного аудио, опционально)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Устройство для обработки"
    )
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        help="Язык синтеза (en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, hu, ko, ja, hi)"
    )
    
    args = parser.parse_args()
    
    # Определяем пути
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    transcripts_csv = project_root / args.transcripts if not Path(args.transcripts).is_absolute() else Path(args.transcripts)
    face_dir = project_root / args.face_dir if not Path(args.face_dir).is_absolute() else Path(args.face_dir)
    xtts_model_path = project_root / args.xtts_model if not Path(args.xtts_model).is_absolute() else Path(args.xtts_model)
    output_dir = project_root / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    wav_base_dir = project_root / args.wav_dir if args.wav_dir and not Path(args.wav_dir).is_absolute() else (Path(args.wav_dir) if args.wav_dir else None)
    
    # Проверяем существование файлов и директорий
    if not transcripts_csv.exists():
        print(f"[ERROR] Файл транскрипций не найден: {transcripts_csv}")
        return
    
    if not face_dir.exists():
        print(f"[WARNING] Директория с лицами не найдена: {face_dir}")
        print("   Продолжаем без проверки лиц...")
    
    # Проверка модели необязательна, так как можно использовать предобученную
    if not xtts_model_path.exists():
        print(f"[WARNING] Локальная модель XTTS-v2 не найдена в {xtts_model_path}")
        print("[INFO] Будет использована предобученная модель (будет скачана при первом запуске)")
        print()
    
    # Запускаем обработку
    process_transcripts(
        transcripts_csv=transcripts_csv,
        face_dir=face_dir,
        xtts_model_path=xtts_model_path,
        output_dir=output_dir,
        wav_base_dir=wav_base_dir,
        device=args.device,
        language=args.language
    )


if __name__ == "__main__":
    main()

