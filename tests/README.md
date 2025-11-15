# Тесты для TTS метрик

Эта папка содержит тестовые скрипты для проверки работоспособности TTS метрик.

## Файлы

- `test_tts_metrics.py` - Автоматические тесты метрик (unit tests)
- `test_real_data.py` - Тестирование на реальных аудио данных

## Использование

### Запуск из корневой директории проекта

```bash
# Автоматические тесты
python tests/test_tts_metrics.py

# Тестирование на реальных данных
python tests/test_real_data.py

# Batch тестирование
python tests/test_real_data.py --batch
```

### Важно

Все скрипты должны запускаться из **корневой директории проекта**, чтобы пути к файлам работали корректно:

```bash
cd C:\Users\d3\Desktop\face2voice-main
python tests/test_real_data.py
```

## Результаты

- Результаты тестирования сохраняются в `outputs/`
- JSON файлы с метриками: `outputs/*_metrics.json`
- Batch результаты: `outputs/batch_metrics_results.json`

