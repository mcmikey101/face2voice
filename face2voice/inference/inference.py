import dlib
import torch
import numpy as np
from PIL import Image
from typing import Optional
from torchvision import transforms
from face2voice.models.SpeakerEncoder import SpeakerEncoder
from face2voice.models.Face2Voice import Face2VoiceModel
from face2voice.models.FaceEncoder import FaceEncoder
from face2voice.models.TTSModel import TTSModel
import os
class Inference():
    def __init__(self, face2voice_ckpt, face_encoder_ckpt, shape_pred_path, tone_conv_ckpt, tone_conv_conf, tts_name, tts_ckpt, tts_conf, speakers_path, speaker):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.shape_pred_path = shape_pred_path

        self.face_transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.base_tts = TTSModel(model_name=tts_name, model_path=tts_ckpt, config_path=tts_conf, speakers_path=speakers_path, speaker=speaker)

        self.speaker_encoder = SpeakerEncoder(ckpt_path=tone_conv_ckpt, config_path=tone_conv_conf)

        self.face_encoder = FaceEncoder()
        face_enc_state_dict = torch.load(face_encoder_ckpt)
        self.face_encoder.load_state_dict(state_dict=face_enc_state_dict)

        self.face2voice = Face2VoiceModel(face_encoder=self.face_encoder, speaker_encoder=self.speaker_encoder)
        f2v_state_dict = torch.load(face2voice_ckpt, weights_only=False)
        self.face2voice.load_state_dict(f2v_state_dict["model_state_dict"])
        self.face2voice.eval()

    def synthesize_base(self,
        text: str,
        output_path,
        language="ru"):
        
        self.base_tts.synthesize(text, output_path=output_path, language=language)

    def process_image(self, image_path, output_path=None, face_size=112, padding=0.3):
        try:
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor(
                self.shape_pred_path
            )

            img = Image.open(image_path).convert("RGB")
            img_np = np.array(img)

            faces = detector(img_np, 1)
            if len(faces) == 0:
                print("No faces detected")
                return None

            face = faces[0]
            landmarks = predictor(img_np, face)

            left_eye = np.mean(
                [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)],
                axis=0
            )
            right_eye = np.mean(
                [(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)],
                axis=0
            )

            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            angle = np.degrees(np.arctan2(dy, dx))

            center_x = (face.left() + face.right()) // 2
            center_y = (face.top() + face.bottom()) // 2

            img_rot = img.rotate(angle, center=(center_x, center_y), expand=True)
            rot_np = np.array(img_rot)

            def rotate_point(x, y, cx, cy, ang):
                rad = np.radians(ang)
                xr = (x - cx) * np.cos(rad) - (y - cy) * np.sin(rad) + cx
                yr = (x - cx) * np.sin(rad) + (y - cy) * np.cos(rad) + cy
                return xr, yr

            x1, y1 = rotate_point(face.left(), face.top(), center_x, center_y, angle)
            x2, y2 = rotate_point(face.right(), face.bottom(), center_x, center_y, angle)

            face_w = x2 - x1
            face_h = y2 - y1

            pad_w = int(face_w * padding)
            pad_h = int(face_h * padding)

            x1 = max(0, int(x1 - pad_w))
            y1 = max(0, int(y1 - pad_h))
            x2 = min(rot_np.shape[1], int(x2 + pad_w))
            y2 = min(rot_np.shape[0], int(y2 + pad_h))

            crop_np = rot_np[y1:y2, x1:x2]
            out = Image.fromarray(crop_np).resize((face_size, face_size), Image.LANCZOS)

            if output_path:
                out.save(output_path)

            return out
        except Exception as e:
            print(e)
            return None

    def get_audio_emb(self, audio_path):
        try:
            emb = self.speaker_encoder.encode_single(audio=audio_path, input="audio", return_numpy=False)
            emb = emb.detach().clone().requires_grad_(True).transpose(1, 2).squeeze(0).reshape(1, -1, 1)
            return emb
        except Exception as e:
            print(e)
            return None

    def get_image_emb(self, image_path, output_path=None):
        try:
            img = self.process_image(image_path=image_path, output_path=output_path)
            img = self.face_transform(img)
            emb = self.face2voice(img.unsqueeze(0))
            emb = emb.detach().clone().requires_grad_(True).reshape(1, -1, 1)
            return emb
        except Exception as e:
            print(e)
            return None
    
    def compare_embeddings(self, emb1, emb2):
        return torch.nn.functional.cosine_similarity(emb1, emb2, dim=1)
    
    def average_emb(self, images):
        try:
            avg_emb = torch.zeros((1, 256, 1))
            for img in images:
                emb = self.get_image_emb(image_path=img)
                avg_emb += emb
            return avg_emb / len(images)
        except Exception as e:
            print(e)
            return None
        
    def clone_voice(self, image_path, base_audio_path, output_path):
        if type(image_path) is list:
            tgt_emb = self.average_emb(image_path)
        else:
            tgt_emb = self.get_image_emb(image_path=image_path)
        src_emb = self.get_audio_emb(audio_path=base_audio_path)
        self.speaker_encoder.tone_color_converter.convert(audio_src_path=base_audio_path, src_se=src_emb, tgt_se=tgt_emb, output_path=output_path)

    def synthesize_voice(self, image_path, base_audio_path, output_path, text: str, language="ru"):
        try:

            self.synthesize_base(text=text,
            language=language,
            output_path=base_audio_path)

            self.clone_voice(image_path, base_audio_path, output_path)

        except Exception as e:
            print(e)
            return None
        
import argparse
import sys

def cli_main():
    parser = argparse.ArgumentParser(
        description="Face2Voice: Генерация голоса по тексту и изображению лица"
    )

    parser.add_argument(
        "--text",
        type=str,
        required=True,
        help="Текст для генерации речи"
    )

    parser.add_argument(
        "--images",
        type=str,
        nargs="+",
        required=True,
        help="Пути к изображениям лица (одно или несколько)"
    )

    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="Путь сохранения итогового аудио (wav)"
    )

    parser.add_argument(
        "--lang",
        type=str,
        default="ru",
        help="Язык синтеза речи (например: ru, en)"
    )

    # Конфигурация чекпоинтов — при необходимости заменить под вашу структуру
    parser.add_argument(
        "--face2voice_ckpt",
        type=str,
        default="face2voice/checkpoints/f2v/face2voice_ckpt_aug_b64_1hid.pth"
    )
    parser.add_argument(
        "--face_encoder_ckpt",
        type=str,
        default="face2voice/checkpoints/face_encoder/facenet_checkpoint.pth"
    )
    parser.add_argument(
        "--shape_pred_path",
        type=str,
        default="face2voice/checkpoints/dlib/shape_predictor_68_face_landmarks.dat"
    )
    parser.add_argument(
        "--tone_conv_ckpt",
        type=str,
        default="face2voice/checkpoints/tone_conv/checkpoint.pth"
    )
    parser.add_argument(
        "--tone_conv_conf",
        type=str,
        default="face2voice/checkpoints/tone_conv/config.json"
    )
    parser.add_argument(
        "--tts_name",
        type=str,
        default="tts_models/multilingual/multi-dataset/xtts_v2"
    )
    parser.add_argument(
        "--tts_ckpt",
        type=str,
        default="face2voice/checkpoints/xtts"
    )
    parser.add_argument(
        "--tts_conf",
        type=str,
        default="face2voice/checkpoints/xtts/config.json"
    )
    parser.add_argument(
        "--speakers_path",
        type=str,
        default="face2voice/checkpoints/xtts/speakers_xtts.pth"
    )
    parser.add_argument(
        "--speaker",
        type=str,
        default="Filip Traverse",
        help="Имя спикера XTTS"
    )

    args = parser.parse_args()

    print("[INFO] Инициализация моделей…")
    try:
        infer = Inference(
            face2voice_ckpt=args.face2voice_ckpt,
            face_encoder_ckpt=args.face_encoder_ckpt,
            shape_pred_path=args.shape_pred_path,
            tone_conv_ckpt=args.tone_conv_ckpt,
            tone_conv_conf=args.tone_conv_conf,
            tts_name=args.tts_name,
            tts_ckpt=args.tts_ckpt,
            tts_conf=args.tts_conf,
            speakers_path=args.speakers_path,
            speaker=args.speaker,
        )
    except Exception as e:
        print(f"[ERROR] Не удалось инициализировать модели: {e}")
        sys.exit(1)

    print("[INFO] Генерация базового TTS-аудио…")
    tmp_out = args.out + ".base_tmp.wav"
    infer.synthesize_base(text=args.text, output_path=tmp_out, language=args.lang)

    print("[INFO] Клонирование голоса…")
    infer.clone_voice(
        image_path=args.images,
        base_audio_path=tmp_out,
        output_path=args.out
    )

    # Удаляем временный файл
    try:
        import os
        if os.path.exists(tmp_out):
            os.remove(tmp_out)
    except:
        pass

    print(f"[OK] Готово! Аудио сохранено в: {args.out}")


if __name__ == "__main__":
    cli_main()