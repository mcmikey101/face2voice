from face2voice.models.FaceEncoder import ArcFaceEncoder
from face2voice.models.SpeakerEncoder import OpenVoiceSpeakerEncoder
import torch
from datasets import FaceVoiceDataset
from models.Face2Voice import Face2VoiceModel
from torch.utils.data import DataLoader
from torchvision import transforms
from trainer.trainer import train_face_to_voice

def main():
    """Example training script."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Initialize face encoder (ArcFace)
    face_encoder = ArcFaceEncoder(config_path="config/arcface_config.yaml")
    face_encoder = face_encoder.to(device)
    face_encoder.eval()
    
    # 2. Initialize OpenVoice speaker encoder
    openvoice_encoder = OpenVoiceSpeakerEncoder(
        openvoice_path="checkpoints/base_speakers/EN",
        device=device
    )
    
    # 3. Create Face2Voice model
    model = Face2VoiceModel(
        face_encoder=face_encoder,
        openvoice_encoder=openvoice_encoder,
        face_dim=512,
        voice_dim=256
    )
    
    # 4. Create datasets and dataloaders
    
    face_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    train_dataset = FaceVoiceDataset()
    
    val_dataset = FaceVoiceDataset()
    
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