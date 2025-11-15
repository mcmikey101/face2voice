"""
Тестовый скрипт для проверки работоспособности TTS метрик.
Запустите: python test_tts_metrics.py
"""

import sys
import os
import numpy as np
import soundfile as sf
import torch
from pathlib import Path
import tempfile

# Установка кодировки для Windows
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# Добавляем путь к проекту (корневая директория)
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_imports():
    """Тест 1: Проверка импортов"""
    print("=" * 60)
    print("Тест 1: Проверка импортов")
    print("=" * 60)
    
    try:
        from face2voice.metrics import TTSMetrics, TTSMetricsBatch
        print("[OK] Импорт TTSMetrics успешен")
        print("[OK] Импорт TTSMetricsBatch успешен")
        return True
    except Exception as e:
        print(f"[ERROR] Ошибка импорта: {e}")
        return False


def create_test_audio(duration=2.0, sample_rate=24000, frequency=440.0):
    """Создать тестовое аудио (синусоида)"""
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t) * 0.5
    return audio.astype(np.float32)


def test_tts_metrics_basic():
    """Тест 2: Базовое использование TTSMetrics"""
    print("\n" + "=" * 60)
    print("Тест 2: Базовое использование TTSMetrics")
    print("=" * 60)
    
    try:
        from face2voice.metrics import TTSMetrics
        
        # Создать тестовые аудио файлы
        with tempfile.TemporaryDirectory() as tmpdir:
            gen_audio = create_test_audio(frequency=440.0)
            ref_audio = create_test_audio(frequency=440.0)  # Та же частота
            
            gen_path = os.path.join(tmpdir, "generated.wav")
            ref_path = os.path.join(tmpdir, "reference.wav")
            
            sf.write(gen_path, gen_audio, 24000)
            sf.write(ref_path, ref_audio, 24000)
            
            # Инициализация метрик
            metrics_calc = TTSMetrics(sample_rate=24000)
            print("[OK] TTSMetrics инициализирован")
            
            # Вычисление метрик
            metrics = metrics_calc.compute_all_metrics(
                generated_audio=gen_path,
                reference_audio=ref_path
            )
            
            print("[OK] Метрики вычислены успешно")
            print(f"  MCD: {metrics.get('mcd', 'N/A')}")
            print(f"  F0 RMSE: {metrics.get('f0_rmse', 'N/A')}")
            print(f"  Duration Ratio: {metrics.get('duration_ratio', 'N/A')}")
            print(f"  Energy Ratio: {metrics.get('energy_ratio', 'N/A')}")
            
            return True
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tts_metrics_with_arrays():
    """Тест 3: Использование с numpy массивами"""
    print("\n" + "=" * 60)
    print("Тест 3: Использование с numpy массивами")
    print("=" * 60)
    
    try:
        from face2voice.metrics import TTSMetrics
        
        # Создать тестовые аудио как массивы
        gen_audio = create_test_audio(frequency=440.0)
        ref_audio = create_test_audio(frequency=440.0)
        
        metrics_calc = TTSMetrics(sample_rate=24000)
        
        # Вычисление метрик напрямую с массивами
        metrics = metrics_calc.compute_all_metrics(
            generated_audio=gen_audio,
            reference_audio=ref_audio
        )
        
        print("[OK] Метрики вычислены с numpy массивами")
        print(f"  Получено метрик: {len(metrics)}")
        
        return True
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_speaker_similarity():
    """Тест 4: Speaker Similarity"""
    print("\n" + "=" * 60)
    print("Тест 4: Speaker Similarity")
    print("=" * 60)
    
    try:
        from face2voice.metrics import TTSMetrics
        
        # Создать тестовые embeddings
        emb1 = torch.randn(256)
        emb2 = torch.randn(256)
        
        metrics_calc = TTSMetrics(sample_rate=24000)
        
        similarity = metrics_calc.compute_speaker_similarity(emb1, emb2)
        
        print(f"[OK] Speaker similarity вычислена: {similarity:.4f}")
        print(f"  Диапазон: [-1, 1], значение: {similarity:.4f}")
        
        # Проверить, что значение в разумных пределах
        assert -1.0 <= similarity <= 1.0, "Similarity должна быть в диапазоне [-1, 1]"
        
        return True
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trainer_integration():
    """Тест 5: Интеграция с Trainer (проверка, что не ломается)"""
    print("\n" + "=" * 60)
    print("Тест 5: Интеграция с Trainer")
    print("=" * 60)
    
    try:
        from face2voice.trainer.trainer import Trainer
        
        # Проверить, что Trainer можно создать без метрик (по умолчанию)
        # Мы не создаем реальный trainer, просто проверяем сигнатуру
        import inspect
        sig = inspect.signature(Trainer.__init__)
        params = list(sig.parameters.keys())
        
        print("[OK] Trainer.__init__ параметры:")
        for param in params:
            print(f"  - {param}")
        
        # Проверить, что compute_tts_metrics есть в параметрах
        if 'compute_tts_metrics' in params:
            print("[OK] Параметр compute_tts_metrics найден")
        else:
            print("[ERROR] Параметр compute_tts_metrics не найден")
            return False
        
        # Проверить значение по умолчанию
        default_value = sig.parameters['compute_tts_metrics'].default
        if default_value == False:
            print("[OK] Значение по умолчанию: False (метрики отключены)")
        else:
            print(f"⚠ Значение по умолчанию: {default_value}")
        
        return True
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_batch_metrics():
    """Тест 6: Batch обработка метрик"""
    print("\n" + "=" * 60)
    print("Тест 6: Batch обработка метрик")
    print("=" * 60)
    
    try:
        from face2voice.metrics import TTSMetrics, TTSMetricsBatch
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Создать несколько тестовых аудио
            gen_audios = []
            ref_audios = []
            
            for i in range(3):
                gen_audio = create_test_audio(frequency=440.0 + i * 10)
                ref_audio = create_test_audio(frequency=440.0 + i * 10)
                
                gen_path = os.path.join(tmpdir, f"gen_{i}.wav")
                ref_path = os.path.join(tmpdir, f"ref_{i}.wav")
                
                sf.write(gen_path, gen_audio, 24000)
                sf.write(ref_path, ref_audio, 24000)
                
                gen_audios.append(gen_path)
                ref_audios.append(ref_path)
            
            # Вычислить batch метрики
            metrics_calc = TTSMetrics(sample_rate=24000)
            batch_metrics = TTSMetricsBatch(metrics_calc)
            
            averaged_metrics = batch_metrics.compute_batch_metrics(
                generated_audios=gen_audios,
                reference_audios=ref_audios
            )
            
            print("[OK] Batch метрики вычислены")
            print(f"  Получено усредненных метрик: {len(averaged_metrics)}")
            for key in list(averaged_metrics.keys())[:5]:  # Показать первые 5
                print(f"  {key}: {averaged_metrics[key]:.4f}")
            
            return True
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_individual_metrics():
    """Тест 7: Индивидуальные метрики"""
    print("\n" + "=" * 60)
    print("Тест 7: Индивидуальные метрики")
    print("=" * 60)
    
    try:
        from face2voice.metrics import TTSMetrics
        
        gen_audio = create_test_audio(frequency=440.0)
        ref_audio = create_test_audio(frequency=440.0)
        
        metrics_calc = TTSMetrics(sample_rate=24000)
        
        # Тест MCD
        mcd = metrics_calc.compute_mcd(gen_audio, ref_audio)
        print(f"[OK] MCD: {mcd:.4f} dB")
        
        # Тест F0 метрик
        f0_metrics = metrics_calc.compute_f0_metrics(gen_audio, ref_audio)
        print(f"[OK] F0 RMSE: {f0_metrics['f0_rmse']:.4f} Hz")
        print(f"[OK] F0 Correlation: {f0_metrics['f0_correlation']:.4f}")
        
        # Тест Duration метрик
        duration_metrics = metrics_calc.compute_duration_metrics(gen_audio, ref_audio)
        print(f"[OK] Duration Ratio: {duration_metrics['duration_ratio']:.4f}")
        
        # Тест Energy метрик
        energy_metrics = metrics_calc.compute_energy_metrics(gen_audio, ref_audio)
        print(f"[OK] Energy Ratio: {energy_metrics['energy_ratio']:.4f}")
        
        # Тест STOI (если доступен)
        try:
            from pystoi import stoi
            stoi_score = metrics_calc.compute_stoi(gen_audio, ref_audio)
            print(f"[OK] STOI: {stoi_score:.4f}")
        except ImportError:
            print("[SKIP] STOI: pystoi не установлен")
        except Exception as e:
            print(f"[WARN] STOI: {e}")
        
        # Тест PESQ (если доступен)
        try:
            from pesq import pesq
            pesq_score = metrics_calc.compute_pesq(gen_audio, ref_audio)
            print(f"[OK] PESQ: {pesq_score:.4f}")
        except ImportError:
            print("[SKIP] PESQ: pesq не установлен")
        except Exception as e:
            print(f"[WARN] PESQ: {e}")
        
        return True
    except Exception as e:
        print(f"✗ Ошибка: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Запустить все тесты"""
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ TTS МЕТРИК")
    print("=" * 60)
    
    tests = [
        ("Импорты", test_imports),
        ("Базовое использование", test_tts_metrics_basic),
        ("Использование с массивами", test_tts_metrics_with_arrays),
        ("Speaker Similarity", test_speaker_similarity),
        ("Интеграция с Trainer", test_trainer_integration),
        ("Batch обработка", test_batch_metrics),
        ("Индивидуальные метрики", test_individual_metrics),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n[ERROR] Критическая ошибка в тесте '{test_name}': {e}")
            results.append((test_name, False))
    
    # Итоги
    print("\n" + "=" * 60)
    print("ИТОГИ ТЕСТИРОВАНИЯ")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "[PASS] ПРОЙДЕН" if result else "[FAIL] ПРОВАЛЕН"
        print(f"{status}: {test_name}")
    
    print("\n" + "=" * 60)
    print(f"Пройдено: {passed}/{total} тестов")
    print("=" * 60)
    
    if passed == total:
        print("\n[SUCCESS] Все тесты пройдены успешно!")
        return 0
    else:
        print(f"\n[WARNING] Провалено тестов: {total - passed}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

