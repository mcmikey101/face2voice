import os
import tempfile
import shutil
import uuid
from pathlib import Path
from typing import List, Dict
from datetime import datetime

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from face2voice.inference.inference import Inference


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

inference: Inference | None = None

# Хранилище задач (в продакшене использовать Redis/DB)
jobs: Dict[str, dict] = {}


def init_inference() -> Inference:
    """
    Инициализация Inference с путями, аналогичными примеру из inference.py.
    """
    base_dir = Path(__file__).resolve().parent.parent
    ckpt_root = base_dir / "face2voice" / "checkpoints"

    face2voice_ckpt = ckpt_root / "f2v" / "face2voice_ckpt_aug_b64_1hid.pth"
    face_encoder_ckpt = ckpt_root / "face_encoder" / "facenet_checkpoint.pth"
    shape_pred_path = ckpt_root / "dlib" / "shape_predictor_68_face_landmarks.dat"
    tone_conv_ckpt = ckpt_root / "tone_conv" / "checkpoint.pth"
    tone_conv_conf = ckpt_root / "tone_conv" / "config.json"
    tts_name = "tts_models/multilingual/multi-dataset/xtts_v2"
    tts_ckpt = ckpt_root / "xtts"
    tts_conf = ckpt_root / "xtts" / "config.json"
    speakers_path = ckpt_root / "xtts" / "speakers_xtts.pth"
    speaker = "Filip Traverse"

    for p in [
        face2voice_ckpt,
        face_encoder_ckpt,
        shape_pred_path,
        tone_conv_ckpt,
        tone_conv_conf,
        tts_ckpt,
        tts_conf,
        speakers_path,
    ]:
        if not p.exists():
            raise RuntimeError(f"Не найден файл/директория модели: {p}")

    infer = Inference(
        face2voice_ckpt=str(face2voice_ckpt),
        face_encoder_ckpt=str(face_encoder_ckpt),
        shape_pred_path=str(shape_pred_path),
        tone_conv_ckpt=str(tone_conv_ckpt),
        tone_conv_conf=str(tone_conv_conf),
        tts_name=str(tts_name),
        tts_ckpt=str(tts_ckpt),
        tts_conf=str(tts_conf),
        speakers_path=str(speakers_path),
        speaker=speaker,
    )
    return infer


@app.on_event("startup")
def on_startup() -> None:
    global inference
    inference = init_inference()


def cleanup_files(files: List[str], temp_dir: str | None = None) -> None:
    for path in files:
        try:
            if path and os.path.exists(path):
                os.remove(path)
        except Exception:
            pass

    if temp_dir and os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


def process_job(job_id: str, text: str, image_paths: List[str], temp_dir: str):
    """
    Фоновая задача для генерации аудио.
    """
    try:
        jobs[job_id]["status"] = "processing"
        
        base_audio_path = os.path.join(temp_dir, "base_tts.wav")
        output_audio_path = os.path.join(temp_dir, "result.wav")

        inference.synthesize_voice(
            image_path=image_paths,
            base_audio_path=base_audio_path,
            output_path=output_audio_path,
            text=text,
            language="ru",
        )

        if not os.path.exists(output_audio_path):
            jobs[job_id]["status"] = "error"
            jobs[job_id]["error"] = "Аудиофайл не был создан"
            return

        jobs[job_id]["status"] = "done"
        jobs[job_id]["url"] = f"/api/download/{job_id}"
        jobs[job_id]["output_path"] = output_audio_path
        jobs[job_id]["temp_files"] = image_paths + [base_audio_path]
        jobs[job_id]["temp_dir"] = temp_dir

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"] = str(e)
        cleanup_files(image_paths + [base_audio_path, output_audio_path], temp_dir)


@app.post("/api/generate")
async def generate_audio(
    background_tasks: BackgroundTasks,
    text: str = Form(...),
    images: List[UploadFile] = File(...),
):
    """
    Создает задачу на генерацию аудио и возвращает job_id для отслеживания.
    """
    if inference is None:
        raise HTTPException(status_code=500, detail="Модель не инициализирована")

    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Текст обязателен")

    if len(images) == 0:
        raise HTTPException(status_code=400, detail="Нужно загрузить хотя бы одно фото")

    if len(images) > 16:
        raise HTTPException(status_code=400, detail="Максимум 16 фотографий")

    # Создаем временную директорию
    temp_dir = tempfile.mkdtemp(prefix="face2voice_")
    image_paths: List[str] = []

    # Сохраняем изображения
    for idx, img in enumerate(images):
        filename = img.filename or f"image_{idx}.png"
        ext = os.path.splitext(filename)[1].lower()
        if ext not in [".jpg", ".jpeg", ".png"]:
            cleanup_files(image_paths, temp_dir)
            raise HTTPException(
                status_code=400,
                detail=f"Недопустимый формат файла: {filename}. Разрешены JPG/PNG.",
            )

        out_path = os.path.join(temp_dir, f"image_{idx}{ext}")
        content = await img.read()
        with open(out_path, "wb") as f:
            f.write(content)
        image_paths.append(out_path)

    # Создаем задачу
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "pending",
        "created_at": datetime.now().isoformat(),
    }

    # Запускаем обработку в фоне
    background_tasks.add_task(process_job, job_id, text, image_paths, temp_dir)

    return {"id": job_id, "status": "pending"}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """
    Проверяет статус задачи.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Задача не найдена")

    job = jobs[job_id]
    
    return {
        "status": job["status"],
        "url": job.get("url"),
        "error": job.get("error"),
    }


@app.get("/api/download/{job_id}")
async def download_audio(job_id: str, background_tasks: BackgroundTasks):
    """
    Скачивает готовое аудио и планирует очистку файлов ПОСЛЕ отправки.
    """
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Задача не найдена")

    job = jobs[job_id]
    
    if job["status"] != "done":
        raise HTTPException(status_code=400, detail="Аудио еще не готово")

    output_path = job.get("output_path")
    if not output_path or not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Аудиофайл не найден")

    # КРИТИЧНО: очистка происходит ПОСЛЕ отправки файла
    temp_files = job.get("temp_files", [])
    temp_dir = job.get("temp_dir")
    
    # Добавляем output_path в список для удаления
    files_to_clean = temp_files + [output_path]
    
    # Планируем очистку ПОСЛЕ отправки
    background_tasks.add_task(cleanup_files, files_to_clean, temp_dir)
    background_tasks.add_task(lambda: jobs.pop(job_id, None))

    return FileResponse(
        path=output_path,
        media_type="audio/wav",
        filename="face2voice.wav",
    )