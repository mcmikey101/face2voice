import torch
import torch.nn as nn
import torchaudio
from typing import Union, List, Optional
import numpy as np
import os
from openvoice import se_extractor
from openvoice.mel_processing import mel_spectrogram_torch
from PIL import Image
import torchvision


class SpeakerEncoder(nn.Module):
    """
    Wrapper class for OpenVoice V2 Speaker Encoder that extracts speaker embeddings
    from audio files or tensors, supporting both single and batch processing.
    """
    
    def __init__(
        self,
        ckpt_path: str,
        config_path: Optional[str] = None,
        device: Optional[str] = None
    ):
        """
        Initialize the OpenVoice V2 Speaker Encoder Wrapper.
        
        Args:
            ckpt_path: Path to speaker encoder checkpoint
            config_path: Path to config file (optional)
            device: Device to run the model on ('cuda', 'cpu', or None for auto)
        """
        super().__init__()
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Import OpenVoice components
        try:
            from openvoice.api import ToneColorConverter
        except ImportError:
            raise ImportError(
                "OpenVoice not installed. Install with: pip install openvoice"
            )
        
        # Load the tone color converter which contains the speaker encoder
        self.tone_color_converter = ToneColorConverter(
            config_path,
            device=self.device
        )
        self.tone_color_converter.load_ckpt(ckpt_path)
        # Set to evaluation mode
        self.tone_color_converter.model.eval()
    
    def preprocess_audio(
        self,
        audio: Union[str, torch.Tensor, np.ndarray],
        target_sr: int = 16000
    ) -> torch.Tensor:
        """
        Preprocess audio input to the format expected by the encoder.
        
        Args:
            audio: Audio file path, torch tensor, or numpy array
            target_sr: Target sample rate (default: 16000)
        
        Returns:
            Preprocessed audio tensor
        """
        # Load audio if it's a file path
        if isinstance(audio, str):
            waveform, sr = torchaudio.load(audio)
        elif isinstance(audio, np.ndarray):
            waveform = torch.from_numpy(audio).float()
            sr = target_sr
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
        else:
            waveform = audio
            sr = target_sr
        
        # Resample if necessary
        if sr != target_sr:
            resampler = torchaudio.transforms.Resample(sr, target_sr)
            waveform = resampler(waveform)
        
        # Convert to mono if stereo
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        
        return waveform
    
    @torch.no_grad()
    def encode_single(
        self,
        audio: Union[str, torch.Tensor, np.ndarray],
        return_numpy: bool = False,
        input: str = "spec_tensor"
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Extract speaker embedding from a single audio sample.
        
        Args:
            audio: Audio file path, tensor, or numpy array
            return_numpy: If True, return numpy array instead of torch tensor
        
        Returns:
            Speaker embedding vector
        """
        embedding: Optional[torch.Tensor] = None
        
        # If audio is a file path, use OpenVoice's built-in method
        if input == "audio":
            if isinstance(audio, str):
                # OpenVoice provides se_extractor for extracting embeddings
                from openvoice import se_extractor
                
                # Extract speaker embedding using OpenVoice's method
                # This returns the tone color embedding
                embedding, _ = se_extractor.get_se(
                    audio,
                    self.tone_color_converter,
                    target_dir='temp_se',
                    vad=True  # Voice activity detection
                )
                
                # Clean up temp files
                if os.path.exists('temp_se'):
                    import shutil
                    shutil.rmtree('temp_se')
            else:
                raise ValueError(f"For input='audio', audio must be a file path (str), got {type(audio)}")

        elif input == "spec_tensor":
            if isinstance(audio, str):
                mel_spec = torch.load(audio)
            elif isinstance(audio, torch.Tensor):
                mel_spec = audio
            elif isinstance(audio, np.ndarray):
                mel_spec = torch.from_numpy(audio)
            else:
                raise ValueError(f"Unsupported audio type for spec_tensor input: {type(audio)}")

            if mel_spec.dim() == 2:
                mel_spec = mel_spec.unsqueeze(0)
            embedding = self.tone_color_converter.model.ref_enc(
                mel_spec.transpose(1, 2)
            )
        else:
            raise ValueError(f"Unsupported input type: {input}")
        
        if embedding is None:
            raise ValueError("Embedding was not generated. Check input parameters.")
        
        embedding = embedding.to(self.device)
        
        if return_numpy:
            return embedding.cpu().numpy()
        
        embedding = embedding.unsqueeze(0)
        
        return embedding
    
    @torch.no_grad()
    def encode_batch(
        self,
        audio: List[Union[str, torch.Tensor, np.ndarray]],
        return_numpy: bool = False,
        input: str = "spec_tensor"
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Extract speaker embeddings from a batch of audio samples.
        
        Args:
            audio_list: List of audio file paths, tensors, or numpy arrays
            return_numpy: If True, return numpy array instead of torch tensor
            use_vad: Whether to use voice activity detection (for file paths)
        
        Returns:
            Batch of speaker embeddings (batch_size, embedding_dim)
        """
        embeddings = []
        
        if input == "audio":
            # Use OpenVoice's batch processing for files
            
            for audio_path in audio:
                embedding, _ = se_extractor.get_se(
                    audio_path,
                    self.tone_color_converter,
                    target_dir='temp_se_batch',
                    vad=True
                )
                embeddings.append(embedding)
            
            # Clean up temp files
            if os.path.exists('temp_se_batch'):
                import shutil
                shutil.rmtree('temp_se_batch')
        elif input == "spec_img":
            # Process tensors/arrays
            mel_specs = []
            hps = self.tone_color_converter.hps
            
            for audio_item in audio:
                waveform = self.preprocess_audio(audio_item)
                waveform = waveform.to(self.device)
                # Get mel parameters with defaults
                n_fft = getattr(hps, 'filter_length', 1024)
                num_mels = getattr(hps, 'n_mel_channels', 80)
                sampling_rate = getattr(hps, 'sampling_rate', 24000)
                hop_size = getattr(hps, 'hop_length', 256)
                win_size = getattr(hps, 'win_length', 1024)
                fmin = getattr(hps, 'mel_fmin', 0.0)
                fmax = getattr(hps, 'mel_fmax', 8000.0)
                mel_spec = mel_spectrogram_torch(
                    waveform, 
                    n_fft=n_fft, 
                    num_mels=num_mels,
                    sampling_rate=sampling_rate, 
                    hop_size=hop_size, 
                    win_size=win_size,
                    fmin=fmin,
                    fmax=fmax,
                    center=False
                )
                mel_specs.append(mel_spec)
            
            # Pad to same length for batching
            max_len = max(spec.shape[-1] for spec in mel_specs)
            
            padded_specs = []
            for spec in mel_specs:
                if spec.dim() == 2:
                    spec = spec.unsqueeze(0)
                
                pad_len = max_len - spec.shape[-1]
                if pad_len > 0:
                    spec = torch.nn.functional.pad(spec, (0, pad_len))
                
                padded_specs.append(spec)
            
            # Stack into batch
            batch = torch.cat(padded_specs, dim=0)
            batch = batch.transpose(1, 2)  # (B, T, C)
            
            # Get embeddings
            batch_emb = self.tone_color_converter.model.ref_enc(batch)
            
            for i in range(batch_emb.shape[0]):
                embeddings.append(batch_emb[i])

        elif input == "spec_tensor":
            # Convert all inputs to tensors
            audio_tensors: List[torch.Tensor] = []
            for item in audio:
                if isinstance(item, str):
                    tensor = torch.load(item)
                elif isinstance(item, np.ndarray):
                    tensor = torch.from_numpy(item)
                elif isinstance(item, torch.Tensor):
                    tensor = item
                else:
                    raise ValueError(f"Unsupported type in audio list: {type(item)}")
                audio_tensors.append(tensor)
            
            max_len = max(spec.shape[-1] for spec in audio_tensors)
            
            padded_specs = []
            for spec in audio_tensors:
                if spec.dim() == 2:
                    spec = spec.unsqueeze(0)
                
                pad_len = max_len - spec.shape[-1]
                if pad_len > 0:
                    spec = torch.nn.functional.pad(spec, (0, pad_len))
                
                padded_specs.append(spec)
            
            # Stack into batch
            batch = torch.cat(padded_specs, dim=0)
            batch = batch.transpose(1, 2)  # (B, T, C)
            
            # Get embeddings
            batch_emb = self.tone_color_converter.model.ref_enc(batch)
            
            for i in range(batch_emb.shape[0]):
                embeddings.append(batch_emb[i])
        
        # Stack all embeddings
        batch_embeddings = torch.stack(embeddings, dim=0)
        
        if return_numpy:
            return batch_embeddings.cpu().numpy()
        
        batch_embeddings = torch.tensor(batch_embeddings)
        return batch_embeddings
    
    def forward(
        self,
        audio: Union[str, torch.Tensor, np.ndarray, List],
        return_numpy: bool = True,
        input: str = "spec_tensor"
    ) -> Union[torch.Tensor, np.ndarray]:
        """
        Forward pass - automatically handles single or batch input.
        
        Args:
            audio: Single audio or list of audio inputs
            return_numpy: If True, return numpy array instead of torch tensor
        
        Returns:
            Speaker embedding(s)
        """
        if isinstance(audio, list):
            return self.encode_batch(audio, return_numpy=return_numpy, input=input)
        else:
            return self.encode_single(audio, return_numpy=return_numpy, input=input)
