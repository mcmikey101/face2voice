import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union, Optional, Tuple
import numpy as np
import librosa
from pathlib import Path


class SpeakerEncoder(nn.Module):
    
    def __init__(
        self,
        ckpt_base: str = "checkpoints/base_speakers/EN_V2",
        ckpt_converter: str = "checkpoints/converter_v2",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        version: str = "v2"
    ):
        """
        Initialize OpenVoice V2 speaker encoder.
        
        Args:
            ckpt_base: Path to base speaker checkpoint directory (EN_V2 or ZH_V2)
            ckpt_converter: Path to converter checkpoint directory
            device: Device to run the model on
            version: OpenVoice version ('v2' or 'v1')
        """
        super().__init__()
        
        self.device = torch.device(device)
        self.ckpt_base = Path(ckpt_base)
        self.ckpt_converter = Path(ckpt_converter)
        self.version = version
        
        # Initialize OpenVoice V2 components
        self._load_openvoice_v2()
        
        # Speaker embedding dimension (256 for OpenVoice V2)
        self.embedding_dim = 256
        
        # Audio preprocessing parameters
        self.sample_rate = 24000  # OpenVoice V2 uses 24kHz (upgraded from V1's 16kHz)
        
        print(f"OpenVoice V2 Speaker Encoder initialized")
        print(f"Embedding dimension: {self.embedding_dim}")
        print(f"Sample rate: {self.sample_rate}")
        print(f"Device: {self.device}")
    
    def preprocess_audio(
        self,
        audio: Union[str, np.ndarray, torch.Tensor],
        sample_rate: Optional[int] = None
    ) -> np.ndarray:
        """
        Preprocess audio for speaker encoding.
        
        Args:
            audio: Audio file path, numpy array, or torch tensor
            sample_rate: Sample rate of input audio (if array/tensor)
        
        Returns:
            Preprocessed audio as numpy array
        """
        # Load audio if path is provided
        if isinstance(audio, (str, Path)):
            audio_data, sr = librosa.load(str(audio), sr=self.sample_rate)
            return audio_data
        
        # Convert torch tensor to numpy
        if isinstance(audio, torch.Tensor):
            audio = audio.cpu().numpy()
        
        # Resample if necessary
        if sample_rate is not None and sample_rate != self.sample_rate:
            audio = librosa.resample(
                audio,
                orig_sr=sample_rate,
                target_sr=self.sample_rate
            )
        
        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=0)
        
        # Normalize
        audio = audio.astype(np.float32)
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()
        
        return audio
    
    @torch.no_grad()
    def extract_embedding(
        self,
        audio: Union[str, np.ndarray, torch.Tensor],
        sample_rate: Optional[int] = None,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Extract speaker embedding from audio.
        
        Args:
            audio: Audio file path, numpy array, or torch tensor
            sample_rate: Sample rate of input audio
            normalize: Whether to L2-normalize the embedding
        
        Returns:
            Speaker embedding tensor of shape (embedding_dim,)
        """
        # Preprocess audio
        audio_data = self.preprocess_audio(audio, sample_rate)
        
        # Save temporarily for OpenVoice (it expects file path)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            import soundfile as sf
            sf.write(tmp_path, audio_data, self.sample_rate)
        
        try:
            # Extract speaker embedding using OpenVoice
            se = self.se_extractor.get_se(
                tmp_path,
                self.tone_converter,
                target_dir='processed',
                vad=True  # Use voice activity detection
            )
            
            # Convert to torch tensor
            if isinstance(se, np.ndarray):
                embedding = torch.from_numpy(se).float()
            else:
                embedding = se
            
            # Normalize if requested
            if normalize:
                embedding = F.normalize(embedding, p=2, dim=-1)
            
            return embedding.to(self.device)
        
        finally:
            # Clean up temporary file
            Path(tmp_path).unlink(missing_ok=True)
    
    @torch.no_grad()
    def extract_embeddings_batch(
        self,
        audio_list: list,
        sample_rate: Optional[int] = None,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Extract speaker embeddings from a batch of audio files.
        
        Args:
            audio_list: List of audio paths, arrays, or tensors
            sample_rate: Sample rate of input audio
            normalize: Whether to L2-normalize embeddings
        
        Returns:
            Batch of speaker embeddings, shape (batch_size, embedding_dim)
        """
        embeddings = []
        
        for audio in audio_list:
            emb = self.extract_embedding(audio, sample_rate, normalize)
            embeddings.append(emb)
        
        return torch.stack(embeddings)
    
    def forward(
        self,
        audio: Union[str, np.ndarray, torch.Tensor],
        sample_rate: Optional[int] = None,
        normalize: bool = True
    ) -> torch.Tensor:
        """
        Forward pass - extract speaker embedding.
        
        Args:
            audio: Audio input
            sample_rate: Sample rate of input
            normalize: Whether to normalize embedding
        
        Returns:
            Speaker embedding
        """
        return self.extract_embeddings_batch(audio, sample_rate, normalize)
    
    def save_embedding(self, embedding: torch.Tensor, path: str):
        """Save speaker embedding to file."""
        torch.save(embedding.cpu(), path)
        print(f"Saved embedding to {path}")
    
    def load_embedding(self, path: str) -> torch.Tensor:
        """Load speaker embedding from file."""
        embedding = torch.load(path, map_location=self.device)
        print(f"Loaded embedding from {path}")
        return embedding