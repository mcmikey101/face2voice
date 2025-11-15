"""
Скрипт для тестирования TTS метрик на реальных данных.
Использует реальные аудио файлы из проекта.

Использование:
    python test_real_data.py
    python test_real_data.py --generated outputs/test.wav --reference resources/demo_speaker0.mp3
    python test_real_data.py --batch  # Тестировать все доступные пары
"""

import sys
import os
import argparse
import json
from pathlib import Path
import numpy as np

# Установка кодировки для Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Добавляем путь к проекту (корневая директория)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from face2voice.metrics import TTSMetrics, TTSMetricsBatch


def format_metric_value(value):
    """Форматировать значение метрики для вывода"""
    if np.isinf(value) or np.isnan(value):
        return "N/A"
    return f"{value:.4f}"


def print_metrics_table(metrics, title="Метрики"):
    """Красиво вывести метрики в виде таблицы"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)
    
    # Группировка метрик
    groups = {
        "Спектральные метрики": ['mcd'],
        "F0 (Высота тона)": ['f0_rmse', 'f0_correlation', 'f0_mean_error'],
        "Длительность": ['duration_ratio', 'duration_error', 'generated_duration', 'reference_duration'],
        "Энергия": ['energy_ratio', 'energy_error'],
        "Сходство голоса": ['speaker_similarity'],
        "SOTA метрики": ['stoi', 'pesq']
    }
    
    for group_name, metric_keys in groups.items():
        found_metrics = []
        for key in metric_keys:
            if key in metrics:
                found_metrics.append((key, metrics[key]))
        
        if found_metrics:
            print(f"\n{group_name}:")
            print("-" * 70)
            for key, value in found_metrics:
                formatted = format_metric_value(value)
                # Описание метрик
                descriptions = {
                    'mcd': 'MCD (Mel Cepstral Distortion, dB)',
                    'f0_rmse': 'F0 RMSE (Hz)',
                    'f0_correlation': 'F0 Correlation (0-1)',
                    'f0_mean_error': 'F0 Mean Error (Hz)',
                    'duration_ratio': 'Duration Ratio',
                    'duration_error': 'Duration Error (sec)',
                    'generated_duration': 'Generated Duration (sec)',
                    'reference_duration': 'Reference Duration (sec)',
                    'energy_ratio': 'Energy Ratio',
                    'energy_error': 'Energy Error',
                    'speaker_similarity': 'Speaker Similarity (0-1)',
                    'stoi': 'STOI (Intelligibility, 0-1)',
                    'pesq': 'PESQ (Quality, -0.5 to 4.5)'
                }
                desc = descriptions.get(key, key)
                print(f"  {desc:40s} : {formatted:>10s}")


def test_single_pair(generated_path, reference_path, sample_rate=24000):
    """Тестировать одну пару аудио файлов"""
    print("\n" + "=" * 70)
    print("ТЕСТИРОВАНИЕ НА РЕАЛЬНЫХ ДАННЫХ")
    print("=" * 70)
    print(f"\nСгенерированное аудио: {generated_path}")
    print(f"Референсное аудио:     {reference_path}")
    
    # Проверка существования файлов
    if not os.path.exists(generated_path):
        print(f"\n[ERROR] Файл не найден: {generated_path}")
        return None
    
    if not os.path.exists(reference_path):
        print(f"\n[ERROR] Файл не найден: {reference_path}")
        return None
    
    try:
        # Инициализация метрик
        metrics_calc = TTSMetrics(sample_rate=sample_rate)
        
        # Вычисление метрик
        print("\n[INFO] Вычисление метрик...")
        metrics = metrics_calc.compute_all_metrics(
            generated_audio=generated_path,
            reference_audio=reference_path
        )
        
        # Вывод результатов
        print_metrics_table(metrics, "Результаты метрик")
        
        # Интерпретация результатов
        print("\n" + "=" * 70)
        print("  Интерпретация результатов")
        print("=" * 70)
        
        if 'mcd' in metrics and not np.isinf(metrics['mcd']):
            mcd = metrics['mcd']
            if mcd < 5.0:
                print(f"  MCD: {mcd:.2f} dB - Отличное качество (MCD < 5.0)")
            elif mcd < 10.0:
                print(f"  MCD: {mcd:.2f} dB - Хорошее качество (5.0 <= MCD < 10.0)")
            else:
                print(f"  MCD: {mcd:.2f} dB - Требует улучшения (MCD >= 10.0)")
        
        if 'f0_correlation' in metrics:
            corr = metrics['f0_correlation']
            if corr > 0.8:
                print(f"  F0 Correlation: {corr:.4f} - Отличная корреляция (> 0.8)")
            elif corr > 0.6:
                print(f"  F0 Correlation: {corr:.4f} - Хорошая корреляция (0.6-0.8)")
            else:
                print(f"  F0 Correlation: {corr:.4f} - Низкая корреляция (< 0.6)")
        
        if 'speaker_similarity' in metrics:
            sim = metrics['speaker_similarity']
            if sim > 0.8:
                print(f"  Speaker Similarity: {sim:.4f} - Очень похожий голос (> 0.8)")
            elif sim > 0.6:
                print(f"  Speaker Similarity: {sim:.4f} - Похожий голос (0.6-0.8)")
            else:
                print(f"  Speaker Similarity: {sim:.4f} - Разные голоса (< 0.6)")
        
        if 'duration_ratio' in metrics:
            ratio = metrics['duration_ratio']
            if 0.9 <= ratio <= 1.1:
                print(f"  Duration Ratio: {ratio:.4f} - Длительность совпадает (0.9-1.1)")
            else:
                print(f"  Duration Ratio: {ratio:.4f} - Длительность отличается")
        
        if 'stoi' in metrics:
            stoi_val = metrics['stoi']
            if stoi_val > 0.75:
                print(f"  STOI: {stoi_val:.4f} - Отличная разборчивость (> 0.75)")
            elif stoi_val > 0.6:
                print(f"  STOI: {stoi_val:.4f} - Хорошая разборчивость (0.6-0.75)")
            else:
                print(f"  STOI: {stoi_val:.4f} - Низкая разборчивость (< 0.6)")
        
        if 'pesq' in metrics:
            pesq_val = metrics['pesq']
            if pesq_val > 3.0:
                print(f"  PESQ: {pesq_val:.4f} - Отличное качество (> 3.0)")
            elif pesq_val > 2.0:
                print(f"  PESQ: {pesq_val:.4f} - Хорошее качество (2.0-3.0)")
            else:
                print(f"  PESQ: {pesq_val:.4f} - Низкое качество (< 2.0)")
        
        # Сохранение результатов (относительно корня проекта)
        project_root = Path(__file__).parent.parent
        generated_path_obj = Path(generated_path)
        if not generated_path_obj.is_absolute():
            generated_path_obj = project_root / generated_path_obj
        
        output_file = generated_path_obj.parent / f"{generated_path_obj.stem}_metrics.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'generated_audio': str(generated_path),
                'reference_audio': str(reference_path),
                'metrics': {k: (float(v) if not (np.isinf(v) or np.isnan(v)) else None) 
                           for k, v in metrics.items()}
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n[OK] Результаты сохранены: {output_file}")
        
        return metrics
        
    except Exception as e:
        print(f"\n[ERROR] Ошибка при вычислении метрик: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_batch_pairs(pairs, sample_rate=24000):
    """Тестировать несколько пар файлов"""
    print("\n" + "=" * 70)
    print("BATCH ТЕСТИРОВАНИЕ")
    print("=" * 70)
    print(f"Количество пар: {len(pairs)}\n")
    
    all_metrics = []
    valid_pairs = []
    
    for i, (gen_path, ref_path) in enumerate(pairs, 1):
        print(f"\n[{i}/{len(pairs)}] Тестирование пары:")
        print(f"  Generated: {gen_path}")
        print(f"  Reference: {ref_path}")
        
        if not os.path.exists(gen_path) or not os.path.exists(ref_path):
            print(f"  [SKIP] Пропущено (файлы не найдены)")
            continue
        
        try:
            metrics_calc = TTSMetrics(sample_rate=sample_rate)
            metrics = metrics_calc.compute_all_metrics(
                generated_audio=gen_path,
                reference_audio=ref_path
            )
            all_metrics.append(metrics)
            valid_pairs.append((gen_path, ref_path))
            print(f"  [OK] Метрики вычислены")
        except Exception as e:
            print(f"  [ERROR] Ошибка: {e}")
    
    if not all_metrics:
        print("\n[ERROR] Не удалось вычислить метрики ни для одной пары")
        return
    
    # Вычислить средние метрики
    print("\n" + "=" * 70)
    print("  СРЕДНИЕ МЕТРИКИ ПО ВСЕМ ПАРАМ")
    print("=" * 70)
    
    averaged = {}
    for key in all_metrics[0].keys():
        values = [m[key] for m in all_metrics 
                 if key in m and not (np.isinf(m[key]) or np.isnan(m[key]))]
        if values:
            averaged[f'avg_{key}'] = float(np.mean(values))
            averaged[f'std_{key}'] = float(np.std(values))
            averaged[f'min_{key}'] = float(np.min(values))
            averaged[f'max_{key}'] = float(np.max(values))
    
    print_metrics_table(averaged, "Средние метрики")
    
    # Сохранение результатов (относительно корня проекта)
    project_root = Path(__file__).parent.parent
    output_file = project_root / "outputs" / "batch_metrics_results.json"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'num_pairs': len(valid_pairs),
            'pairs': [{'generated': str(g), 'reference': str(r)} 
                     for g, r in valid_pairs],
            'averaged_metrics': averaged,
            'individual_metrics': [
                {k: (float(v) if not (np.isinf(v) or np.isnan(v)) else None) 
                 for k, v in m.items()}
                for m in all_metrics
            ]
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n[OK] Batch результаты сохранены: {output_file}")


def find_available_pairs():
    """Найти доступные пары файлов для тестирования"""
    pairs = []
    project_root = Path(__file__).parent.parent
    
    # Поиск в outputs/
    outputs_dir = project_root / "outputs"
    if outputs_dir.exists():
        # Ищем wav файлы в outputs
        for wav_file in outputs_dir.rglob("*.wav"):
            # Пытаемся найти соответствующий референсный файл
            ref_candidates = [
                project_root / "resources" / "demo_speaker0.mp3",
                project_root / "resources" / "demo_speaker1.mp3",
                project_root / "resources" / "demo_speaker2.mp3",
                project_root / "resources" / "example_reference.mp3",
            ]
            
            for ref_candidate in ref_candidates:
                if ref_candidate.exists():
                    pairs.append((str(wav_file.relative_to(project_root)), 
                                 str(ref_candidate.relative_to(project_root))))
                    break  # Используем первый найденный референс
    
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Тестирование TTS метрик на реальных данных")
    parser.add_argument(
        '--generated', '-g',
        type=str,
        help='Путь к сгенерированному аудио файлу'
    )
    parser.add_argument(
        '--reference', '-r',
        type=str,
        help='Путь к референсному аудио файлу'
    )
    parser.add_argument(
        '--batch', '-b',
        action='store_true',
        help='Тестировать все доступные пары файлов'
    )
    parser.add_argument(
        '--sample-rate', '-sr',
        type=int,
        default=24000,
        help='Sample rate для метрик (по умолчанию: 24000)'
    )
    
    args = parser.parse_args()
    
    if args.batch:
        # Batch режим
        pairs = find_available_pairs()
        if not pairs:
            project_root = Path(__file__).parent.parent
            print("[WARNING] Не найдено доступных пар файлов для тестирования")
            print("\nДоступные файлы:")
            print("  Generated (outputs/):")
            for wav in (project_root / "outputs").rglob("*.wav"):
                print(f"    - {wav.relative_to(project_root)}")
            print("\n  Reference (resources/):")
            for mp3 in (project_root / "resources").glob("*.mp3"):
                print(f"    - {mp3.relative_to(project_root)}")
            return
        
        test_batch_pairs(pairs, sample_rate=args.sample_rate)
    
    elif args.generated and args.reference:
        # Одна пара
        test_single_pair(args.generated, args.reference, sample_rate=args.sample_rate)
    
    else:
        # Интерактивный режим - показать доступные файлы и предложить выбор
        print("=" * 70)
        print("ТЕСТИРОВАНИЕ TTS МЕТРИК НА РЕАЛЬНЫХ ДАННЫХ")
        print("=" * 70)
        
        project_root = Path(__file__).parent.parent
        
        print("\nДоступные сгенерированные файлы (outputs/):")
        generated_files = list((project_root / "outputs").rglob("*.wav"))
        if not generated_files:
            print("  Не найдено")
        else:
            for i, f in enumerate(generated_files, 1):
                print(f"  {i}. {f.relative_to(project_root)}")
        
        print("\nДоступные референсные файлы (resources/):")
        reference_files = list((project_root / "resources").glob("*.mp3"))
        if not reference_files:
            print("  Не найдено")
        else:
            for i, f in enumerate(reference_files, 1):
                print(f"  {i}. {f}")
        
        if generated_files and reference_files:
            # Использовать первую пару по умолчанию
            gen_file = generated_files[0]
            ref_file = reference_files[0]
            
            print(f"\n[INFO] Используется первая пара по умолчанию:")
            print(f"  Generated: {gen_file.relative_to(project_root)}")
            print(f"  Reference: {ref_file.relative_to(project_root)}")
            print("\nДля выбора других файлов используйте:")
            print("  python tests/test_real_data.py --generated <path> --reference <path>")
            print("\nДля batch тестирования:")
            print("  python tests/test_real_data.py --batch")
            
            test_single_pair(str(gen_file.relative_to(project_root)), 
                           str(ref_file.relative_to(project_root)), 
                           sample_rate=args.sample_rate)
        else:
            print("\n[ERROR] Недостаточно файлов для тестирования")
            print("\nИспользование:")
            print("  python tests/test_real_data.py --generated <path> --reference <path>")
            print("  python tests/test_real_data.py --batch")


if __name__ == "__main__":
    main()

