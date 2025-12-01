import os
import tempfile
import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from face2voice.inference.inference import Inference


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # в проде лучше ограничить доменами фронтенда
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

inference: Inference | None = None


def init_inference() -> Inference:
    """
    Инициализация Inference с путями, аналогичными примеру из inference.py.
    Путь считается относительно корня репозитория face2voice.
    """
    base_dir = Path(__file__).resolve().parent.parent  # корень репо face2voice
    ckpt_root = base_dir / "face2voice" / "checkpoints"

    face2voice_ckpt = ckpt_root / "f2v" / "face2voice_ckpt_aug_b64_1hid.pth"
    face_encoder_ckpt = ckpt_root / "face_encoder" / "facenet_checkpoint.pth"
    shape_pred_path = ckpt_root / "dlib" / "shape_predictor_68_face_landmarks.dat"
    tone_conv_ckpt = ckpt_root / "tone_conv" / "checkpoint.pth"
    tone_conv_conf = ckpt_root / "tone_conv" / "config.json"
    tts_ckpt = ckpt_root / "xtts"
    tts_conf = ckpt_root / "xtts" / "config.json"
    tts_name = 'tts_models/multilingual/multi-dataset/xtts_v2'
    speakers_path = ckpt_root / "xtts" / "speakers_xtts.pth"
    speaker = "Nova Hogarth"

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


@app.post("/api/generate")
async def generate_audio(
    background_tasks: BackgroundTasks,
    text: str = Form(...),
    images: List[UploadFile] = File(...),
):
    """
    Вход:
      - text: текст для озвучивания
      - images: 1–16 фотографий (JPG/PNG)

    Выход:
      - аудиофайл (audio/wav)
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

    temp_dir = tempfile.mkdtemp(prefix="face2voice_")
    image_paths: List[str] = []

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

    image_path = image_paths[0]

    base_audio_path = os.path.join(temp_dir, "base_tts.wav")
    output_audio_path = os.path.join(temp_dir, "result.wav")

    try:
        print("synth")
        inference.synthesize_voice(
            image_path=image_paths,
            base_audio_path=base_audio_path,
            output_path=output_audio_path,
            text=text,
            language="ru",
        )
    except Exception as e:
        cleanup_files(image_paths + [base_audio_path, output_audio_path], temp_dir)
        raise HTTPException(status_code=500, detail=f"Ошибка генерации аудио: {e}")

    if not os.path.exists(output_audio_path):
        cleanup_files(image_paths + [base_audio_path, output_audio_path], temp_dir)
        raise HTTPException(status_code=500, detail="Аудиофайл не был создан")

    files_to_remove = image_paths + [base_audio_path, output_audio_path]
    background_tasks.add_task(cleanup_files, files_to_remove, temp_dir)

    return FileResponse(
        path=output_audio_path,
        media_type="audio/wav",
        filename="face2voice.wav",
    )
