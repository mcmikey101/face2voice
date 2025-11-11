"""
OpenVoice V2 TTS Wrapper with Face2Voice Integration.

This module provides a unified interface to generate speech from face images
by integrating the trained Face2Voice model as a speaker encoder replacement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from typing import Union, Optional, Tuple
import numpy as np
from PIL import Image
import tempfile
import soundfile as sf


class TTSModel(nn.Module):
    """
    Complete TTS system that generates speech from face images.
    
    Architecture:
        Face Image → Face2Voice Model → Speaker Embedding → OpenVoice TTS → Audio
    
    This replaces OpenVoice's audio-based speaker encoder with our face-based encoder.
    
    Usage:
        tts = TTSModel(
            face2voice_checkpoint='checkpoints/face2voice/best_model.pth',
            openvoice_base='checkpoints/base_speakers/EN_V2',
            openvoice_converter='checkpoints/converter_v2'
        )
        
        audio = tts.synthesize(
            text="Hello, this is generated speech.",
            face_image="person.jpg"
        )
    """
    
    def __init__(
        self,
        face2voice_checkpoint: str,
        openvoice_base: str,
        openvoice_converter: str,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        language: str = 'English',
        speed: float = 1.0
    ):
        """
        Initialize Face2Voice TTS system.
        
        Args:
            face2voice_checkpoint: Path to trained Face2Voice model checkpoint
            openvoice_base: Path to OpenVoice V2 base speaker checkpoint
            openvoice_converter: Path to OpenVoice V2 converter checkpoint
            device: Device to run inference on
            language: Language for TTS ('English' or 'Chinese')
            speed: Speaking speed multiplier
        """
        super().__init__()
        
        self.device = torch.device(device)
        self.language = language
        self.speed = speed
        
        # Load Face2Voice model (our trained model)
        print("Loading Face2Voice model...")
        self.face2voice_model = self._load_face2voice_model(face2voice_checkpoint)
        self.face2voice_model.eval()
        
        # Load OpenVoice V2 components
        print("Loading OpenVoice V2 TTS...")
        self._load_openvoice_components(openvoice_base, openvoice_converter)
        
        # Image preprocessing
        from torchvision import transforms
        self.face_transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        print(f"✓ Face2Voice TTS initialized on {self.device}")
    
    def _load_face2voice_model(self, checkpoint_path: str):
        """Load the trained Face2Voice model."""
        from models.Face2Voice import Face2VoiceModel
        from face2voice.models.FaceEncoder import ArcFaceEncoder
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Reconstruct model from checkpoint config
        if 'config' in checkpoint:
            cfg = checkpoint['config']
            
            # Create sub-components (these won't be used, just for loading state dict)
            face_encoder = ArcFaceEncoder(
                backbone=cfg['model']['face_encoder']['backbone'],
                embedding_dim=cfg['model']['face_encoder']['embedding_dim'],
                pretrained=False,  # We'll load from checkpoint
                freeze=True
            )
            
            model = Face2VoiceModel(face_encoder, openvoice_encoder=None, face_dim=512, voice_dim=256, hidden_dims=[512, 384])
        else:
            raise ValueError("Checkpoint must contain 'config' field")
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model = model.to(self.device)
        
        return model
    
    def _load_openvoice_components(self, base_path: str, converter_path: str):
        """Load OpenVoice V2 TTS components."""
        try:
            from openvoice.api import BaseSpeakerTTS, ToneColorConverter
            
            # Load base speaker TTS
            self.base_speaker_tts = BaseSpeakerTTS(
                f'{base_path}/config.json',
                device=self.device
            )
            self.base_speaker_tts.load_ckpt(f'{base_path}/checkpoint.pth')
            
            # Load tone color converter
            self.tone_color_converter = ToneColorConverter(
                f'{converter_path}/config.json',
                device=self.device
            )
            self.tone_color_converter.load_ckpt(f'{converter_path}/checkpoint.pth')
            
            print("✓ OpenVoice V2 components loaded")
            
        except Exception as e:
            print(f"Error loading OpenVoice V2: {e}")
            raise
    
    def preprocess_face_image(
        self,
        face_image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Preprocess face image for the model.
        
        Args:
            face_image: Face image (path, PIL Image, numpy array, or tensor)
        
        Returns:
            Preprocessed face tensor (1, 3, 112, 112)
        """
        # Load image if path
        if isinstance(face_image, (str, Path)):
            face_image = Image.open(face_image).convert('RGB')
        
        # Convert numpy to PIL
        elif isinstance(face_image, np.ndarray):
            if face_image.dtype != np.uint8:
                face_image = (face_image * 255).astype(np.uint8)
            face_image = Image.fromarray(face_image)
        
        # Convert tensor to PIL
        elif isinstance(face_image, torch.Tensor):
            if face_image.dim() == 4:
                face_image = face_image.squeeze(0)
            face_image = face_image.cpu()
            if face_image.shape[0] == 3:  # CHW format
                face_image = face_image.permute(1, 2, 0)
            face_image = (face_image.numpy() * 255).astype(np.uint8)
            face_image = Image.fromarray(face_image)
        
        # Apply transforms
        face_tensor = self.face_transform(face_image)
        face_tensor = face_tensor.unsqueeze(0)  # Add batch dimension
        
        return face_tensor
    
    @torch.no_grad()
    def extract_speaker_embedding(
        self,
        face_image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor]
    ) -> torch.Tensor:
        """
        Extract speaker embedding from face image.
        
        Args:
            face_image: Face image in various formats
        
        Returns:
            Speaker embedding (1, 256) for OpenVoice
        """
        # Preprocess image
        face_tensor = self.preprocess_face_image(face_image)
        face_tensor = face_tensor.to(self.device)
        
        # Extract speaker embedding via Face2Voice model
        speaker_embedding = self.face2voice_model(face_tensor)
        
        return speaker_embedding
    
    def synthesize(
        self,
        text: str,
        face_image: Union[str, Path, Image.Image, np.ndarray, torch.Tensor],
        output_path: Optional[str] = None,
        return_tensor: bool = False,
        speaker: str = 'default',
        normalize: bool = True
    ) -> Union[str, torch.Tensor]:
        """
        Generate speech from text and face image.
        
        Args:
            text: Text to synthesize
            face_image: Face image (determines voice characteristics)
            output_path: Path to save audio (optional)
            return_tensor: If True, return audio tensor instead of path
            speaker: Base speaker style for OpenVoice
            normalize: Whether to normalize the speaker embedding
        
        Returns:
            Path to generated audio file or audio tensor
        """
        print(f"Synthesizing: '{text}'")
        
        # Step 1: Extract speaker embedding from face
        print("  1. Extracting speaker embedding from face...")
        speaker_embedding = self.extract_speaker_embedding(face_image)
        
        # Normalize embedding if requested
        if normalize:
            speaker_embedding = F.normalize(speaker_embedding, p=2, dim=1)
        
        print(f"     Speaker embedding shape: {speaker_embedding.shape}")
        print(f"     Embedding norm: {speaker_embedding.norm().item():.4f}")
        
        # Step 2: Generate base audio with text
        print(f"  2. Generating base audio...")
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_base:
            base_audio_path = tmp_base.name
        
        self.base_speaker_tts.tts(
            text,
            base_audio_path,
            speaker=speaker,
            language=self.language,
            speed=self.speed
        )
        
        print(f"     Base audio generated: {base_audio_path}")
        
        # Step 3: Apply voice characteristics from face
        print("  3. Applying voice characteristics...")
        
        # Convert embedding to numpy for OpenVoice
        speaker_emb_np = speaker_embedding.cpu().numpy().squeeze()
        
        # Generate output path if not provided
        if output_path is None:
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_out:
                output_path = tmp_out.name
        
        # Apply tone conversion
        self.tone_color_converter.convert(
            audio_src_path=base_audio_path,
            src_se=speaker_emb_np,
            tgt_se=speaker_emb_np,  # Use same embedding for source and target
            output_path=output_path,
            message="Face2Voice TTS"
        )
        
        print(f"✓ Speech generated: {output_path}")
        
        # Cleanup base audio
        Path(base_audio_path).unlink(missing_ok=True)
        
        # Return tensor if requested
        if return_tensor:
            audio, sr = sf.read(output_path)
            audio_tensor = torch.from_numpy(audio).float()
            return audio_tensor
        
        return output_path
    
    def synthesize_batch(
        self,
        texts: list,
        face_images: list,
        output_dir: str = 'outputs',
        speaker: str = 'default'
    ) -> list:
        """
        Generate speech for multiple text-face pairs.
        
        Args:
            texts: List of texts to synthesize
            face_images: List of face images (same length as texts)
            output_dir: Directory to save audio files
            speaker: Base speaker style
        
        Returns:
            List of paths to generated audio files
        """
        assert len(texts) == len(face_images), "Number of texts and faces must match"
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_paths = []
        
        for i, (text, face) in enumerate(zip(texts, face_images)):
            output_path = str(output_dir / f"audio_{i:04d}.wav")
            
            audio_path = self.synthesize(
                text=text,
                face_image=face,
                output_path=output_path,
                speaker=speaker
            )
            
            output_paths.append(audio_path)
        
        print(f"\n✓ Generated {len(output_paths)} audio files in {output_dir}")
        
        return output_paths
    
    def clone_voice_from_face(
        self,
        face_image: Union[str, Path, Image.Image],
        texts: list,
        output_dir: str = 'outputs/cloned_voice'
    ) -> list:
        """
        Generate multiple audio samples with the same voice (from one face).
        
        Args:
            face_image: Single face image to use for all samples
            texts: List of texts to synthesize with this voice
            output_dir: Directory to save audio files
        
        Returns:
            List of paths to generated audio files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract speaker embedding once
        print(f"Cloning voice from face image...")
        speaker_embedding = self.extract_speaker_embedding(face_image)
        
        output_paths = []
        
        for i, text in enumerate(texts):
            print(f"\nGenerating sample {i+1}/{len(texts)}")
            print(f"  Text: '{text}'")
            
            output_path = str(output_dir / f"sample_{i:04d}.wav")
            
            # Generate base audio
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp:
                base_audio_path = tmp.name
            
            self.base_speaker_tts.tts(
                text,
                base_audio_path,
                speaker='default',
                language=self.language,
                speed=self.speed
            )
            
            # Apply voice characteristics
            speaker_emb_np = speaker_embedding.cpu().numpy().squeeze()
            
            self.tone_color_converter.convert(
                audio_src_path=base_audio_path,
                src_se=speaker_emb_np,
                tgt_se=speaker_emb_np,
                output_path=output_path
            )
            
            # Cleanup
            Path(base_audio_path).unlink(missing_ok=True)
            
            output_paths.append(output_path)
            print(f"  ✓ Saved: {output_path}")
        
        print(f"\n✓ Generated {len(output_paths)} samples in {output_dir}")
        
        return output_paths

# ==================== Usage Examples ====================

def example_single_synthesis():
    """Example: Generate speech from a single face image."""
    print("="*60)
    print("Example: Single Speech Synthesis")
    print("="*60)
    
    # Initialize TTS system
    tts = TTSModel(
        face2voice_checkpoint='checkpoints/face2voice/best_model.pth',
        openvoice_base='checkpoints/base_speakers/EN_V2',
        openvoice_converter='checkpoints/converter_v2',
        device='cuda'
    )
    
    # Generate speech
    audio_path = tts.synthesize(
        text="Hello, this is a test of face-to-voice text-to-speech synthesis.",
        face_image="test_images/person1.jpg",
        output_path="outputs/sample.wav"
    )
    
    print(f"\n✓ Generated audio: {audio_path}")


def example_voice_cloning():
    """Example: Clone a voice from one face for multiple texts."""
    print("="*60)
    print("Example: Voice Cloning from Face")
    print("="*60)
    
    tts = TTSModel(
        face2voice_checkpoint='checkpoints/face2voice/best_model.pth',
        openvoice_base='checkpoints/base_speakers/EN_V2',
        openvoice_converter='checkpoints/converter_v2'
    )
    
    texts = [
        "Hello, welcome to face-to-voice synthesis.",
        "This system can generate speech from facial images.",
        "The voice characteristics are predicted from the face.",
        "Thank you for listening to this demonstration."
    ]
    
    audio_paths = tts.clone_voice_from_face(
        face_image="test_images/person1.jpg",
        texts=texts,
        output_dir="outputs/cloned_voice"
    )
    
    print(f"\n✓ Generated {len(audio_paths)} audio samples")


def example_batch_synthesis():
    """Example: Generate speech for multiple face-text pairs."""
    print("="*60)
    print("Example: Batch Synthesis")
    print("="*60)
    
    tts = TTSModel(
        face2voice_checkpoint='checkpoints/face2voice/best_model.pth',
        openvoice_base='checkpoints/base_speakers/EN_V2',
        openvoice_converter='checkpoints/converter_v2'
    )
    
    texts = [
        "This is person one speaking.",
        "This is person two speaking.",
        "This is person three speaking."
    ]
    
    face_images = [
        "test_images/person1.jpg",
        "test_images/person2.jpg",
        "test_images/person3.jpg"
    ]
    
    audio_paths = tts.synthesize_batch(
        texts=texts,
        face_images=face_images,
        output_dir="outputs/batch"
    )
    
    print(f"\n✓ Generated {len(audio_paths)} audio files")


def example_compare_voices():
    """Example: Generate same text with different faces."""
    print("="*60)
    print("Example: Compare Different Voices")
    print("="*60)
    
    tts = TTSModel(
        face2voice_checkpoint='checkpoints/face2voice/best_model.pth',
        openvoice_base='checkpoints/base_speakers/EN_V2',
        openvoice_converter='checkpoints/converter_v2'
    )
    
    text = "The quick brown fox jumps over the lazy dog."
    
    face_images = [
        "test_images/person1.jpg",
        "test_images/person2.jpg",
        "test_images/person3.jpg"
    ]
    
    for i, face_image in enumerate(face_images):
        output_path = f"outputs/comparison/voice_{i+1}.wav"
        
        tts.synthesize(
            text=text,
            face_image=face_image,
            output_path=output_path
        )
        
        print(f"✓ Generated voice {i+1}: {output_path}")


def example_with_pil_image():
    """Example: Use PIL Image directly."""
    print("="*60)
    print("Example: Direct PIL Image Input")
    print("="*60)
    
    from PIL import Image
    
    tts = TTSModel(
        face2voice_checkpoint='checkpoints/face2voice/best_model.pth',
        openvoice_base='checkpoints/base_speakers/EN_V2',
        openvoice_converter='checkpoints/converter_v2'
    )
    
    # Load image with PIL
    face_image = Image.open("test_images/person1.jpg")
    
    # Can apply preprocessing here if needed
    # face_image = face_image.resize((256, 256))
    # face_image = face_image.convert('RGB')
    
    audio_path = tts.synthesize(
        text="This example uses a PIL Image directly.",
        face_image=face_image,
        output_path="outputs/pil_example.wav"
    )
    
    print(f"\n✓ Generated audio: {audio_path}")


if __name__ == "__main__":
    # Run examples
    import sys
    
    if len(sys.argv) > 1:
        example_name = sys.argv[1]
        
        if example_name == "single":
            example_single_synthesis()
        elif example_name == "clone":
            example_voice_cloning()
        elif example_name == "batch":
            example_batch_synthesis()
        elif example_name == "compare":
            example_compare_voices()
        elif example_name == "pil":
            example_with_pil_image()
        else:
            print(f"Unknown example: {example_name}")
            print("Available examples: single, clone, batch, compare, pil")
    else:
        print("Usage: python OpenVoiceTTS.py [example_name]")
        print("Examples:")
        print("  single   - Single speech synthesis")
        print("  clone    - Voice cloning from one face")
        print("  batch    - Batch synthesis with multiple faces")
        print("  compare  - Compare different voices")
        print("  pil      - Use PIL Image directly")