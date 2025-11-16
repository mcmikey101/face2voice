# TTS Metrics Module

Модуль для вычисления метрик качества Text-to-Speech (TTS) для проекта Face2Voice.

## Описание

Этот модуль предоставляет комплексные метрики для оценки качества синтезированной речи:

- **MCD (Mel Cepstral Distortion)** - оценка спектрального качества
- **F0 метрики** - оценка точности высоты тона (pitch)
- **Duration метрики** - оценка длительности речи
- **Speaker Similarity** - оценка сходства голоса
- **Energy метрики** - оценка энергетических характеристик
- **STOI** - разборчивость речи (SOTA метрика, требует `pystoi`)
- **PESQ** - перцептуальное качество (SOTA метрика, требует `pesq`)

## Использование

### Базовое использование

```python
from face2voice.metrics import TTSMetrics

# Инициализация
metrics_calc = TTSMetrics(sample_rate=24000)

# Вычисление всех метрик
metrics = metrics_calc.compute_all_metrics(
    generated_audio="path/to/generated.wav",
    reference_audio="path/to/reference.wav"
)

print(f"MCD: {metrics['mcd']:.2f} dB")
print(f"F0 RMSE: {metrics['f0_rmse']:.2f} Hz")
print(f"Speaker Similarity: {metrics['speaker_similarity']:.4f}")
```

### Использование в Trainer

```python
from face2voice.trainer import Trainer

# Создание trainer с метриками
trainer = Trainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    num_epochs=100,
    model_save_path="checkpoints/",
    compute_tts_metrics=True,  # Включить вычисление метрик
    tts_metrics_sample_rate=24000
)

trainer.train_face_to_voice()
```

### Использование в Inference

Метрики автоматически вычисляются, если в конфигурации указан `reference_audio`:

```yaml
inference_mode:
  single:
    text: "Hello, world"
    face_image: "path/to/face.jpg"
    output_path: "output.wav"
    reference_audio: "path/to/reference.wav"  # Опционально
```

## Метрики

### MCD (Mel Cepstral Distortion)
- **Единица измерения**: dB
- **Диапазон**: 0-∞ (меньше = лучше)
- **Описание**: Оценка спектрального искажения в mel-кепстральной области

### F0 метрики
- **f0_rmse**: RMSE между F0 последовательностями (Hz)
- **f0_correlation**: Корреляция между F0 последовательностями (0-1, выше = лучше)
- **f0_mean_error**: Средняя абсолютная ошибка F0 (Hz)

### Duration метрики
- **duration_ratio**: Отношение длительностей (1.0 = идеально)
- **duration_error**: Абсолютная ошибка длительности (секунды)

### Speaker Similarity
- **Единица измерения**: Косинусное сходство (0-1, выше = лучше)
- **Описание**: Сходство speaker embeddings между сгенерированным и референсным аудио

### Energy метрики
- **energy_ratio**: Отношение энергий
- **energy_error**: Абсолютная ошибка энергии

### STOI (Short-Time Objective Intelligibility) - SOTA метрика
- **Единица измерения**: 0-1 (выше = лучше)
- **Описание**: Оценка разборчивости речи, коррелирует с человеческим восприятием
- **Типичные значения**: > 0.75 - отличная, 0.6-0.75 - хорошая, < 0.6 - низкая
- **Требует**: `pip install pystoi`

### PESQ (Perceptual Evaluation of Speech Quality) - SOTA метрика
- **Единица измерения**: -0.5 до 4.5 (выше = лучше)
- **Описание**: Оценка качества речи, коррелирует с MOS (Mean Opinion Score)
- **Типичные значения**: > 3.0 - отличное, 2.0-3.0 - хорошее, < 2.0 - низкое
- **Требует**: `pip install pesq`

## Batch обработка

Для вычисления метрик на батче:

```python
from face2voice.metrics import TTSMetrics, TTSMetricsBatch

metrics_calc = TTSMetrics(sample_rate=24000)
batch_metrics = TTSMetricsBatch(metrics_calc)

averaged_metrics = batch_metrics.compute_batch_metrics(
    generated_audios=["gen1.wav", "gen2.wav", ...],
    reference_audios=["ref1.wav", "ref2.wav", ...]
)
```

## Примечания

- Метрики опциональны и не влияют на существующий код
- По умолчанию метрики отключены в Trainer (`compute_tts_metrics=False`)
- Метрики требуют наличия референсного аудио для сравнения
- Некоторые метрики могут быть недоступны для очень коротких аудио
- **STOI и PESQ** требуют установки дополнительных библиотек:
  ```bash
  pip install pystoi pesq
  ```
  Если библиотеки не установлены, эти метрики будут пропущены (graceful degradation)

