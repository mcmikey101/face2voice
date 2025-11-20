import torch
from face2voice.datasets.FaceVoiceDataset import FaceVoiceDataset
from face2voice.models.Face2Voice import Face2VoiceModel
from face2voice.models.FaceEncoder import FaceEncoder
from face2voice.models.SpeakerEncoder import SpeakerEncoder
from torch.utils.data import DataLoader
from torchvision import transforms
from face2voice.trainer.trainer import Trainer
import argparse
import matplotlib.pyplot as plt
import numpy as np

def build_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-fenc", "--face_encoder_ckpt",
        type=str,
        required=True,
        help="Path to the face encoder checkpoint"
    )

    parser.add_argument(
        "-tckpt", "--tone_conv_ckpt",
        type=str,
        required=True,
        help="Path to the tone conversion checkpoint"
    )

    parser.add_argument(
        "-tconf", "--tone_conv_config",
        type=str,
        required=True,
        help="Path to the tone conversion configuration file"
    )

    parser.add_argument(
        "-csv", "--dataset_csv",
        type=str,
        required=True,
        help="Path to the dataset CSV file"
    )
    parser.add_argument(
        "-aud", "--audio_base_path",
        type=str,
        required=True,
        help="Path to the audio files"
    )
    parser.add_argument(
        "-img", "--images_base_path",
        type=str,
        required=True,
        help="Path to the image files"
    )

    parser.add_argument(
        "-load", "--model_load_path",
        type=str,
        help="Path to the model checkpoint load"
    )

    parser.add_argument(
        "-save", "--model_save_path",
        type=str,
        required=True,
        help="Path to the model checkpoint save"
    )

    return parser

def main(face_encoder_ckpt, tone_conv_ckpt, tone_conv_config, dataset_csv, audio_base_path, images_base_path, model_save_path, model_load_path=None, ):
    """Example training script."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Initialize face encoder (ArcFace)
    face_encoder = FaceEncoder()
    state_dict = torch.load(face_encoder_ckpt)
    face_encoder.load_state_dict(state_dict=state_dict)
    face_encoder = face_encoder.to(device)
    face_encoder.eval()
    # 2. Initialize speaker encoder
    speaker_encoder = SpeakerEncoder(ckpt_path=tone_conv_ckpt, config_path=tone_conv_config)
    # 3. Create Face2Voice model
    model = Face2VoiceModel(
        face_encoder=face_encoder,
        speaker_encoder=speaker_encoder
    )
    if model_load_path is not None:
        f2v_state_dict = torch.load(model_load_path, weights_only=False)
        model.load_state_dict(f2v_state_dict["model_state_dict"])

    # 4. Create datasets and dataloaders
    
    face_transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = FaceVoiceDataset(csv_path=dataset_csv, audio_base_path=audio_base_path, 
                                    face_image_base_path=images_base_path, split="train", transform_face=face_transform)
    
    val_dataset = FaceVoiceDataset(csv_path=dataset_csv, audio_base_path=audio_base_path, 
                                    face_image_base_path=images_base_path, split="test", transform_face=face_transform)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=64,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    trainer = Trainer(model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=15,
        model_save_path=model_save_path,
        device=device)
    
    # 5. Train the model
    print("\nStarting training...")

    trainer.train_face_to_voice()

    print("\nTraining complete!")

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    params = {
        "face_encoder_ckpt": args.face_encoder_ckpt,
        "tone_conv_ckpt": args.tone_conv_ckpt,
        "tone_conv_config": args.tone_conv_config,
        "dataset_csv": args.dataset_csv,
        "audio_base_path": args.audio_base_path,
        "images_base_path": args.images_base_path,
        "model_save_path": args.model_save_path,
        "model_load_path": args.model_load_path
    }
    main(**params)