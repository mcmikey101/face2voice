from face2voice.models.FaceEncoder import ArcFaceEncoder
from face2voice.models.SpeakerEncoder import OpenVoiceSpeakerEncoder
import torch
from datasets import FaceVoiceDataset
from models.Face2Voice import Face2VoiceModel
from face2voice.models.FaceEncoder import FaceEncoder
from face2voice.models.SpeakerEncoder import SpeakerEncoder
from torch.utils.data import DataLoader
from torchvision import transforms
from trainer.trainer import train_face_to_voice

def main():
    """Example training script."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Initialize face encoder (ArcFace)
    face_encoder = FaceEncoder()
    state_dict = torch.load(r"face2voice\checkpoints\face_encoder\facenet_checkpoint.pth")
    face_encoder.load_state_dict(state_dict=state_dict)
    face_encoder = face_encoder.to(device)
    face_encoder.eval()
    
    # 2. Initialize speaker encoder
    speaker_encoder = SpeakerEncoder()
    
    # 3. Create Face2Voice model
    model = Face2VoiceModel(
        face_encoder=face_encoder,
        speaker_encoder=speaker_encoder
    )
    
    # 4. Create datasets and dataloaders
    
    face_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train_dataset = FaceVoiceDataset(audio_dir="", face_dir="", transform=face_transform)
    
    val_dataset = FaceVoiceDataset(audio_dir="", face_dir="", transform=face_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # 5. Train the model
    print("\nStarting training...")
    train_face_to_voice(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=50,
        device=device
    )
    
    print("\nTraining complete!")