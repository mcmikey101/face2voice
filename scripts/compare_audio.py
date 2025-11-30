"""
Скрипт для сравнения оригинальных и сгенерированных аудио файлов.

Читает generation_results.csv, загружает пары оригинальных и сгенерированных аудио,
вычисляет метрики сравнения и создает визуализации.
"""

import os
import csv
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Используем non-interactive backend

try:
    import librosa
    import soundfile as sf
    from scipy.spatial.distance import cosine
    from scipy.signal import correlate
except ImportError as e:
    print(f"[ERROR] Не установлены необходимые библиотеки: {e}")
    print("[INFO] Установите: pip install librosa soundfile scipy matplotlib")
    raise


def load_audio(file_path: Path, target_sr: int = 22050) -> Tuple[np.ndarray, float]:
    """
    Загружает аудио файл и приводит к единой sample rate.
    
    Args:
        file_path: Путь к аудио файлу
        target_sr: Целевая частота дискретизации
    
    Returns:
        Tuple (audio_data, sample_rate)
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Аудио файл не найден: {file_path}")
    
    try:
        audio, sr = librosa.load(str(file_path), sr=target_sr, mono=True)
        return audio, sr
    except Exception as e:
        raise ValueError(f"Ошибка загрузки аудио {file_path}: {e}")


def compute_mfcc(audio: np.ndarray, sr: Union[int, float], n_mfcc: int = 13) -> np.ndarray:
    """Вычисляет MFCC признаки."""
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfcc


def compute_spectral_features(audio: np.ndarray, sr: Union[int, float]) -> Dict[str, np.ndarray]:
    """Вычисляет спектральные характеристики."""
    # Спектрограмма
    stft = librosa.stft(audio)
    magnitude = np.abs(stft)
    
    # Mel-спектрограмма
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
    
    # Chroma
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
    
    # Spectral contrast
    spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
    
    # Zero crossing rate
    zcr = librosa.feature.zero_crossing_rate(audio)
    
    return {
        'magnitude': magnitude,
        'mel_spec': mel_spec,
        'chroma': chroma,
        'spectral_contrast': spectral_contrast,
        'zcr': zcr
    }


def compute_audio_metrics(original: np.ndarray, generated: np.ndarray, 
                          sr: Union[int, float] = 22050) -> Dict[str, float]:
    """
    Вычисляет метрики сравнения между оригинальным и сгенерированным аудио.
    
    Args:
        original: Оригинальное аудио
        generated: Сгенерированное аудио
        sr: Sample rate
    
    Returns:
        Словарь с метриками
    """
    metrics = {}
    
    # 1. Длина аудио
    metrics['original_duration'] = len(original) / sr
    metrics['generated_duration'] = len(generated) / sr
    metrics['duration_diff'] = abs(metrics['original_duration'] - metrics['generated_duration'])
    
    # 2. Приводим к одинаковой длине для сравнения
    min_len = min(len(original), len(generated))
    orig_aligned = original[:min_len]
    gen_aligned = generated[:min_len]
    
    # 3. MSE (Mean Squared Error)
    metrics['mse'] = np.mean((orig_aligned - gen_aligned) ** 2)
    
    # 4. MAE (Mean Absolute Error)
    metrics['mae'] = np.mean(np.abs(orig_aligned - gen_aligned))
    
    # 5. RMSE (Root Mean Squared Error)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    
    # 6. Корреляция
    if len(orig_aligned) > 1:
        correlation = np.corrcoef(orig_aligned, gen_aligned)[0, 1]
        metrics['correlation'] = correlation if not np.isnan(correlation) else 0.0
    else:
        metrics['correlation'] = 0.0
    
    # 7. MFCC сравнение
    mfcc_orig = compute_mfcc(orig_aligned, sr)
    mfcc_gen = compute_mfcc(gen_aligned, sr)
    
    # Приводим к одинаковому размеру
    min_frames = min(mfcc_orig.shape[1], mfcc_gen.shape[1])
    mfcc_orig = mfcc_orig[:, :min_frames]
    mfcc_gen = mfcc_gen[:, :min_frames]
    
    # Среднее по времени для каждого MFCC коэффициента
    mfcc_orig_mean = np.mean(mfcc_orig, axis=1)
    mfcc_gen_mean = np.mean(mfcc_gen, axis=1)
    
    # Cosine similarity для MFCC
    mfcc_cosine = 1 - cosine(mfcc_orig_mean, mfcc_gen_mean)
    metrics['mfcc_cosine_similarity'] = mfcc_cosine if not np.isnan(mfcc_cosine) else 0.0
    
    # MSE для MFCC
    metrics['mfcc_mse'] = np.mean((mfcc_orig_mean - mfcc_gen_mean) ** 2)
    
    # 8. Спектральные характеристики
    spec_orig = compute_spectral_features(orig_aligned, sr)
    spec_gen = compute_spectral_features(gen_aligned, sr)
    
    # Mel-спектрограмма сравнение
    mel_orig = spec_orig['mel_spec']
    mel_gen = spec_gen['mel_spec']
    min_mel_frames = min(mel_orig.shape[1], mel_gen.shape[1])
    mel_orig = mel_orig[:, :min_mel_frames]
    mel_gen = mel_gen[:, :min_mel_frames]
    
    # Среднее по времени
    mel_orig_mean = np.mean(mel_orig, axis=1)
    mel_gen_mean = np.mean(mel_gen, axis=1)
    
    mel_cosine = 1 - cosine(mel_orig_mean, mel_gen_mean)
    metrics['mel_cosine_similarity'] = mel_cosine if not np.isnan(mel_cosine) else 0.0
    metrics['mel_mse'] = np.mean((mel_orig_mean - mel_gen_mean) ** 2)
    
    # 9. Chroma сравнение
    chroma_orig = spec_orig['chroma']
    chroma_gen = spec_gen['chroma']
    min_chroma_frames = min(chroma_orig.shape[1], chroma_gen.shape[1])
    chroma_orig = chroma_orig[:, :min_chroma_frames]
    chroma_gen = chroma_gen[:, :min_chroma_frames]
    
    chroma_orig_mean = np.mean(chroma_orig, axis=1)
    chroma_gen_mean = np.mean(chroma_gen, axis=1)
    
    chroma_cosine = 1 - cosine(chroma_orig_mean, chroma_gen_mean)
    metrics['chroma_cosine_similarity'] = chroma_cosine if not np.isnan(chroma_cosine) else 0.0
    
    # 10. Energy (RMS)
    rms_orig = librosa.feature.rms(y=orig_aligned)[0]
    rms_gen = librosa.feature.rms(y=gen_aligned)[0]
    min_rms = min(len(rms_orig), len(rms_gen))
    rms_correlation = np.corrcoef(rms_orig[:min_rms], rms_gen[:min_rms])[0, 1]
    metrics['rms_correlation'] = rms_correlation if not np.isnan(rms_correlation) else 0.0
    
    return metrics


def create_comparison_plot(original: np.ndarray, generated: np.ndarray, 
                          sr: Union[int, float], output_path: Path, metrics: Dict[str, Union[float, str]],
                          speaker_name: str, transcript: str):
    """
    Создает визуализацию сравнения оригинального и сгенерированного аудио.
    
    Args:
        original: Оригинальное аудио
        generated: Сгенерированное аудио
        sr: Sample rate
        output_path: Путь для сохранения графика
        metrics: Словарь с метриками
        speaker_name: Имя спикера
        transcript: Транскрипция
    """
    # Приводим к одинаковой длине
    min_len = min(len(original), len(generated))
    orig_aligned = original[:min_len]
    gen_aligned = generated[:min_len]
    
    time_axis = np.arange(min_len) / sr
    
    # Создаем фигуру с несколькими subplots
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 2, hspace=0.3, wspace=0.3)
    
    # 1. Волновые формы
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(time_axis, orig_aligned, label='Оригинал', alpha=0.7, linewidth=0.5)
    ax1.plot(time_axis, gen_aligned, label='Сгенерировано', alpha=0.7, linewidth=0.5)
    ax1.set_xlabel('Время (сек)')
    ax1.set_ylabel('Амплитуда')
    ax1.set_title(f'Волновые формы: {speaker_name}')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Спектрограммы
    # Оригинал
    ax2 = fig.add_subplot(gs[1, 0])
    stft_orig = librosa.stft(orig_aligned)
    magnitude_orig = np.abs(stft_orig)
    librosa.display.specshow(
        librosa.amplitude_to_db(magnitude_orig, ref=np.max),
        y_axis='hz', x_axis='time', sr=sr, ax=ax2
    )
    ax2.set_title('Спектрограмма: Оригинал')
    ax2.set_ylabel('Частота (Hz)')
    
    # Сгенерировано
    ax3 = fig.add_subplot(gs[1, 1])
    stft_gen = librosa.stft(gen_aligned)
    magnitude_gen = np.abs(stft_gen)
    librosa.display.specshow(
        librosa.amplitude_to_db(magnitude_gen, ref=np.max),
        y_axis='hz', x_axis='time', sr=sr, ax=ax3
    )
    ax3.set_title('Спектрограмма: Сгенерировано')
    ax3.set_ylabel('Частота (Hz)')
    
    # 3. Mel-спектрограммы
    # Оригинал
    ax4 = fig.add_subplot(gs[2, 0])
    mel_orig = librosa.feature.melspectrogram(y=orig_aligned, sr=sr)
    librosa.display.specshow(
        librosa.power_to_db(mel_orig, ref=np.max),
        y_axis='mel', x_axis='time', sr=sr, ax=ax4
    )
    ax4.set_title('Mel-спектрограмма: Оригинал')
    ax4.set_ylabel('Mel-шкала')
    
    # Сгенерировано
    ax5 = fig.add_subplot(gs[2, 1])
    mel_gen = librosa.feature.melspectrogram(y=gen_aligned, sr=sr)
    librosa.display.specshow(
        librosa.power_to_db(mel_gen, ref=np.max),
        y_axis='mel', x_axis='time', sr=sr, ax=ax5
    )
    ax5.set_title('Mel-спектрограмма: Сгенерировано')
    ax5.set_ylabel('Mel-шкала')
    
    # 4. MFCC сравнение
    ax6 = fig.add_subplot(gs[3, :])
    mfcc_orig = compute_mfcc(orig_aligned, sr)
    mfcc_gen = compute_mfcc(gen_aligned, sr)
    min_frames = min(mfcc_orig.shape[1], mfcc_gen.shape[1])
    
    # Средние значения MFCC
    mfcc_orig_mean = np.mean(mfcc_orig[:, :min_frames], axis=1)
    mfcc_gen_mean = np.mean(mfcc_gen[:, :min_frames], axis=1)
    
    x = np.arange(len(mfcc_orig_mean))
    width = 0.35
    ax6.bar(x - width/2, mfcc_orig_mean, width, label='Оригинал', alpha=0.7)
    ax6.bar(x + width/2, mfcc_gen_mean, width, label='Сгенерировано', alpha=0.7)
    ax6.set_xlabel('MFCC коэффициент')
    ax6.set_ylabel('Значение')
    ax6.set_title('Сравнение MFCC признаков')
    ax6.legend()
    ax6.grid(True, alpha=0.3, axis='y')
    
    # Добавляем метрики в заголовок (извлекаем только числовые значения)
    mse_val = metrics.get('mse', 0.0)
    mae_val = metrics.get('mae', 0.0)
    corr_val = metrics.get('correlation', 0.0)
    mfcc_cosine_val = metrics.get('mfcc_cosine_similarity', 0.0)
    
    # Убеждаемся, что значения - это числа
    if isinstance(mse_val, str):
        mse_val = 0.0
    if isinstance(mae_val, str):
        mae_val = 0.0
    if isinstance(corr_val, str):
        corr_val = 0.0
    if isinstance(mfcc_cosine_val, str):
        mfcc_cosine_val = 0.0
    
    metrics_text = (
        f"MSE: {float(mse_val):.6f} | "
        f"MAE: {float(mae_val):.6f} | "
        f"Correlation: {float(corr_val):.4f} | "
        f"MFCC Cosine: {float(mfcc_cosine_val):.4f}"
    )
    fig.suptitle(f'{speaker_name}\n{metrics_text}\nТранскрипция: {transcript[:100]}...', 
                 fontsize=10, y=0.995)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def process_comparisons(
    results_csv: Path,
    wav_base_dir: Path,
    generated_audio_dir: Path,
    output_dir: Path,
    create_plots: bool = True,
    max_samples: Optional[int] = None
):
    """
    Обрабатывает сравнения оригинальных и сгенерированных аудио.
    
    Args:
        results_csv: Путь к CSV файлу с результатами генерации
        wav_base_dir: Базовая директория с оригинальными WAV файлами
        generated_audio_dir: Директория со сгенерированными аудио
        output_dir: Директория для сохранения результатов
        create_plots: Создавать ли графики
        max_samples: Максимальное количество образцов для обработки (None = все)
    """
    # Создаем выходную директорию
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Инициализируем plots_dir заранее, чтобы он был доступен в цикле
    plots_dir: Optional[Path] = None
    if create_plots:
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
    
    # Читаем CSV с результатами
    print("📋 Загрузка результатов генерации...")
    tasks = []
    with open(results_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            original_audio_path = row.get("original_audio", "").strip()
            generated_audio_path = row.get("generated_audio", "").strip()
            speaker_name = row.get("speaker_name", "").strip()
            transcript = row.get("transcript", "").strip()
            speaker_id = row.get("speaker_id", "").strip()
            
            if not original_audio_path or not generated_audio_path:
                continue
            
            # Формируем полные пути
            original_full_path = wav_base_dir / original_audio_path
            generated_full_path = Path(generated_audio_path)
            
            if not original_full_path.exists():
                print(f"[WARNING] Оригинальный файл не найден: {original_full_path}")
                continue
            
            if not generated_full_path.exists():
                print(f"[WARNING] Сгенерированный файл не найден: {generated_full_path}")
                continue
            
            tasks.append({
                "speaker_id": speaker_id,
                "speaker_name": speaker_name,
                "transcript": transcript,
                "original_path": original_full_path,
                "generated_path": generated_full_path
            })
    
    if max_samples:
        tasks = tasks[:max_samples]
    
    total_tasks = len(tasks)
    print(f"[INFO] Найдено {total_tasks} пар для сравнения\n")
    
    if total_tasks == 0:
        print("[WARNING] Не найдено пар для сравнения!")
        return
    
    # Обрабатываем с прогресс-баром
    results = []
    start_time = time.time()
    
    print("[INFO] Начинаем сравнение аудио...\n")
    
    for i, task in enumerate(tqdm(tasks, desc="Сравнение аудио", unit="пара", ncols=100)):
        speaker_name = task["speaker_name"]
        transcript = task["transcript"]
        original_path = task["original_path"]
        generated_path = task["generated_path"]
        
        try:
            # Загружаем аудио
            original_audio, sr_orig = load_audio(original_path)
            generated_audio, sr_gen = load_audio(generated_path)
            
            # Вычисляем метрики (возвращает Dict[str, float])
            metrics_dict = compute_audio_metrics(original_audio, generated_audio, sr_orig)
            # Создаем словарь с правильным типом, копируя все значения
            metrics: Dict[str, Union[float, str]] = {k: v for k, v in metrics_dict.items()}
            
            # Добавляем информацию о файлах (строковые значения)
            metrics['speaker_id'] = task["speaker_id"]
            metrics['speaker_name'] = speaker_name
            metrics['transcript'] = transcript
            metrics['original_path'] = str(original_path)
            metrics['generated_path'] = str(generated_path)
            
            results.append(metrics)
            
            # Создаем график
            if create_plots and plots_dir is not None:
                plot_filename = f"{task['speaker_id']}_{speaker_name}_{i:04d}_comparison.png"
                plot_path = plots_dir / plot_filename
                create_comparison_plot(
                    original_audio, generated_audio, sr_orig,
                    plot_path, metrics, speaker_name, transcript
                )
        
        except Exception as e:
            print(f"\n[WARNING] Ошибка при обработке {speaker_name} (файл {i}): {e}")
            continue
    
    # Сохраняем результаты в CSV
    if results:
        output_csv = output_dir / "comparison_results.csv"
        save_comparison_csv(output_csv, results)
        
        # Создаем сводную статистику
        create_summary_statistics(output_dir / "summary_statistics.txt", results)
        
        # Создаем графики статистики
        if create_plots:
            create_statistics_plots(output_dir / "statistics_plots", results)
    
    elapsed = time.time() - start_time
    
    print(f"\n[INFO] Обработано {len(results)}/{total_tasks} пар")
    print(f"[INFO] Время обработки: {elapsed:.1f}с ({elapsed/60:.1f} мин)")
    print(f"[INFO] Результаты сохранены в {output_dir}")


def save_comparison_csv(output_csv: Path, results: List[Dict]):
    """Сохраняет результаты сравнения в CSV файл."""
    if not results:
        return
    
    # Определяем все возможные ключи
    fieldnames = [
        'speaker_id', 'speaker_name', 'transcript',
        'original_path', 'generated_path',
        'original_duration', 'generated_duration', 'duration_diff',
        'mse', 'mae', 'rmse', 'correlation',
        'mfcc_cosine_similarity', 'mfcc_mse',
        'mel_cosine_similarity', 'mel_mse',
        'chroma_cosine_similarity', 'rms_correlation'
    ]
    
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            # Оставляем только нужные поля
            row = {k: result.get(k, '') for k in fieldnames}
            writer.writerow(row)


def create_summary_statistics(output_file: Path, results: List[Dict]):
    """Создает файл со сводной статистикой."""
    if not results:
        return
    
    metrics_to_analyze = [
        'mse', 'mae', 'rmse', 'correlation',
        'mfcc_cosine_similarity', 'mfcc_mse',
        'mel_cosine_similarity', 'mel_mse',
        'chroma_cosine_similarity', 'rms_correlation',
        'duration_diff'
    ]
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("=" * 80 + "\n")
        f.write("СВОДНАЯ СТАТИСТИКА СРАВНЕНИЯ АУДИО\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Всего обработано пар: {len(results)}\n\n")
        
        for metric in metrics_to_analyze:
            values = [r.get(metric, 0) for r in results if metric in r]
            if values:
                f.write(f"{metric}:\n")
                f.write(f"  Среднее: {np.mean(values):.6f}\n")
                f.write(f"  Медиана: {np.median(values):.6f}\n")
                f.write(f"  Мин: {np.min(values):.6f}\n")
                f.write(f"  Макс: {np.max(values):.6f}\n")
                f.write(f"  Стд. откл.: {np.std(values):.6f}\n")
                f.write("\n")


def create_statistics_plots(output_dir: Path, results: List[Dict]):
    """Создает графики статистики метрик."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_to_plot = [
        ('mse', 'MSE'),
        ('mae', 'MAE'),
        ('correlation', 'Корреляция'),
        ('mfcc_cosine_similarity', 'MFCC Cosine Similarity'),
        ('mel_cosine_similarity', 'Mel Cosine Similarity'),
        ('chroma_cosine_similarity', 'Chroma Cosine Similarity'),
    ]
    
    for metric_key, metric_name in metrics_to_plot:
        values = [r.get(metric_key, 0) for r in results if metric_key in r]
        if not values:
            continue
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Гистограмма
        ax1.hist(values, bins=30, edgecolor='black', alpha=0.7)
        ax1.set_xlabel(metric_name)
        ax1.set_ylabel('Частота')
        ax1.set_title(f'Распределение {metric_name}')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(np.mean(values), color='r', linestyle='--', 
                   label=f'Среднее: {np.mean(values):.4f}')
        ax1.axvline(np.median(values), color='g', linestyle='--', 
                   label=f'Медиана: {np.median(values):.4f}')
        ax1.legend()
        
        # Box plot
        ax2.boxplot(values, vert=True)
        ax2.set_ylabel(metric_name)
        ax2.set_title(f'Box Plot {metric_name}')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(output_dir / f"{metric_key}_statistics.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    # Общий график всех метрик
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, (metric_key, metric_name) in enumerate(metrics_to_plot):
        if idx >= len(axes):
            break
        values = [r.get(metric_key, 0) for r in results if metric_key in r]
        if values:
            axes[idx].hist(values, bins=30, edgecolor='black', alpha=0.7)
            axes[idx].set_xlabel(metric_name)
            axes[idx].set_ylabel('Частота')
            axes[idx].set_title(metric_name)
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "all_metrics_overview.png", dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Сравнение оригинальных и сгенерированных аудио")
    parser.add_argument(
        "--results_csv",
        type=str,
        default="outputs/generated_audio/generation_results.csv",
        help="Путь к CSV файлу с результатами генерации"
    )
    parser.add_argument(
        "--wav_dir",
        type=str,
        default="data/wav",
        help="Базовая директория с оригинальными WAV файлами"
    )
    parser.add_argument(
        "--generated_audio_dir",
        type=str,
        default="outputs/generated_audio",
        help="Директория со сгенерированными аудио"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/comparison_results",
        help="Директория для сохранения результатов сравнения"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Не создавать графики (только метрики)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Максимальное количество образцов для обработки (для тестирования)"
    )
    
    args = parser.parse_args()
    
    # Определяем пути
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    results_csv = project_root / args.results_csv if not Path(args.results_csv).is_absolute() else Path(args.results_csv)
    wav_base_dir = project_root / args.wav_dir if not Path(args.wav_dir).is_absolute() else Path(args.wav_dir)
    generated_audio_dir = project_root / args.generated_audio_dir if not Path(args.generated_audio_dir).is_absolute() else Path(args.generated_audio_dir)
    output_dir = project_root / args.output_dir if not Path(args.output_dir).is_absolute() else Path(args.output_dir)
    
    # Проверяем существование файлов и директорий
    if not results_csv.exists():
        print(f"[ERROR] Файл результатов не найден: {results_csv}")
        return
    
    if not wav_base_dir.exists():
        print(f"[ERROR] Директория с оригинальными WAV не найдена: {wav_base_dir}")
        return
    
    if not generated_audio_dir.exists():
        print(f"[ERROR] Директория со сгенерированными аудио не найдена: {generated_audio_dir}")
        return
    
    # Запускаем обработку
    process_comparisons(
        results_csv=results_csv,
        wav_base_dir=wav_base_dir,
        generated_audio_dir=generated_audio_dir,
        output_dir=output_dir,
        create_plots=not args.no_plots,
        max_samples=args.max_samples
    )


if __name__ == "__main__":
    main()

