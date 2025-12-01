import cv2
import pyaudio
import threading
import collections
import time
from face2voice.models.Face2Voice import Face2VoiceModel
from face2voice.models.SpeakerEncoder import SpeakerEncoder
from face2voice.models.FaceEncoder import FaceEncoder
import torch
import torchaudio
from torchvision import transforms
from openvoice.mel_processing import spectrogram_torch
from PIL import Image

# Video capture and face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
video = cv2.VideoCapture(0)

# Audio circular buffer (last 6 seconds)
RATE = 22050
CHUNK = 1024
BUFFER_DURATION = 1
buffer_size = int(RATE * BUFFER_DURATION / CHUNK)
audio_buffer = collections.deque(maxlen=buffer_size)
audio_lock = threading.Lock()

speaker_encoder = SpeakerEncoder(ckpt_path=r"face2voice\checkpoints\tone_conv\checkpoint.pth", config_path=r"face2voice\checkpoints\tone_conv\config.json")
face_encoder = FaceEncoder()
face_enc_state_dict = torch.load(r"face2voice\checkpoints\face_encoder\facenet_checkpoint.pth")
face_encoder.load_state_dict(state_dict=face_enc_state_dict)

face2voice = Face2VoiceModel(face_encoder=face_encoder, speaker_encoder=speaker_encoder)
f2v_state_dict = torch.load(r"face2voice\checkpoints\f2v\face2voice_ckpt_aug_b64_1hid.pth", weights_only=False)
face2voice.load_state_dict(f2v_state_dict["model_state_dict"])
face2voice.eval()

face_transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

device = "gpu" if torch.cuda.is_available() else "cpu"

def audio_thread():
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16,
                     channels=1,
                     rate=RATE,
                     input=True,
                     frames_per_buffer=CHUNK)

    while True:
        data = stream.read(CHUNK, exception_on_overflow=False)
        with audio_lock:
            audio_buffer.append(data)

threading.Thread(target=audio_thread, daemon=True).start()

if __name__ == "__main__":

    while True:
        ret, frame = video.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)

        with audio_lock:
            raw_audio = b"".join(audio_buffer)

        if raw_audio:
            audio_tensor = torch.frombuffer(raw_audio, dtype=torch.int16).float() / 32768.0
            waveform = audio_tensor.unsqueeze(0)  # [1, samples]
            waveform = torchaudio.functional.resample(waveform, orig_freq=RATE, new_freq=RATE)

        face_crops = []
        for (x, y, w, h) in faces:
            crop = frame[y:y+h, x:x+w]
            
            img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            img = face_transform(img)
            img_emb = face2voice(img.unsqueeze(0))
            img_emb = img_emb.detach().clone().requires_grad_(True).reshape(256)


            waveform = waveform.to(device)
            spec = spectrogram_torch(y=waveform, sampling_rate=22050,
                n_fft=1024,
                hop_size=256,
                win_size=1024
                )
            audio_emb = face2voice.speaker_encoder.encode_single(spec, input="spec_tensor")
            audio_emb = audio_emb.detach().clone().requires_grad_(True).transpose(1, 2).squeeze(0).reshape(256)

            a_norm = img_emb / img_emb.norm(dim=-1, keepdim=True)
            b_norm = audio_emb / audio_emb.norm(dim=-1, keepdim=True)
            sim = (a_norm * b_norm).sum(dim=-1)
            print(len(faces), sim)
            if sim > 0.5:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        cv2.imshow("Video", frame)

        if cv2.waitKey(1) == 27:
            break

    video.release()
    cv2.destroyAllWindows()






