"""
TTS Metrics Module for Face2Voice Project.

This module provides comprehensive metrics for evaluating Text-to-Speech quality:
- Mel Cepstral Distortion (MCD)
- Fundamental Frequency (F0) metrics
- Duration metrics
- Speaker similarity metrics
- Spectral quality metrics
- STOI (Short-Time Objective Intelligibility) - SOTA metric
- PESQ (Perceptual Evaluation of Speech Quality) - SOTA metric
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, Union, List
import librosa
import soundfile as sf
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

# Optional imports for STOI and PESQ
try:
    from pystoi import stoi
    STOI_AVAILABLE = True
except ImportError:
    STOI_AVAILABLE = False
    warnings.warn("pystoi not available. STOI metric will be disabled.")

try:
    from pesq import pesq
    PESQ_AVAILABLE = True
except ImportError:
    PESQ_AVAILABLE = False
    warnings.warn("pesq not available. PESQ metric will be disabled.")


class TTSMetrics:
    """
    Comprehensive TTS evaluation metrics.
    
    Computes various metrics to assess the quality of generated speech:
    - Spectral quality (MCD)
    - Prosody (F0, duration)
    - Speaker similarity
    - Intelligibility (STOI) - SOTA metric
    - Perceptual quality (PESQ) - SOTA metric
    - Overall quality
    """
    
    def __init__(
        self,
        sample_rate: int = 24000,
        hop_length: int = 256,
        n_fft: int = 1024,
        n_mels: int = 80,
        f0_min: float = 50.0,
        f0_max: float = 800.0,
        device: str = 'cpu'
    ):
        """
        Initialize TTS metrics calculator.
        
        Args:
            sample_rate: Audio sample rate
            hop_length: Hop length for STFT
            n_fft: FFT window size
            n_mels: Number of mel filterbanks
            f0_min: Minimum F0 frequency (Hz)
            f0_max: Maximum F0 frequency (Hz)
            device: Device for computation
        """
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.device = device
        
        # Pre-compute mel filterbank
        self.mel_basis = librosa.filters.mel(
            sr=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels
        )
    
    def compute_all_metrics(
        self,
        generated_audio: Union[str, np.ndarray, torch.Tensor],
        reference_audio: Union[str, np.ndarray, torch.Tensor],
        reference_speaker_embedding: Optional[torch.Tensor] = None,
        generated_speaker_embedding: Optional[torch.Tensor] = None,
        speaker_encoder: Optional[object] = None
    ) -> Dict[str, float]:
        """
        Compute all available TTS metrics.
        
        Args:
            generated_audio: Generated audio (path, array, or tensor)
            reference_audio: Reference audio (path, array, or tensor)
            reference_speaker_embedding: Optional pre-computed reference embedding
            generated_speaker_embedding: Optional pre-computed generated embedding
            speaker_encoder: Optional speaker encoder for similarity computation
        
        Returns:
            Dictionary of metric names and values
        """
        # Load audio
        gen_audio, gen_sr = self._load_audio(generated_audio)
        ref_audio, ref_sr = self._load_audio(reference_audio)
        
        # Ensure same sample rate
        if gen_sr != self.sample_rate:
            gen_audio = librosa.resample(gen_audio, orig_sr=gen_sr, target_sr=self.sample_rate)
        if ref_sr != self.sample_rate:
            ref_audio = librosa.resample(ref_audio, orig_sr=ref_sr, target_sr=self.sample_rate)
        
        metrics = {}
        
        # Spectral metrics
        try:
            mcd = self.compute_mcd(gen_audio, ref_audio)
            metrics['mcd'] = mcd
        except Exception as e:
            warnings.warn(f"Failed to compute MCD: {e}")
            metrics['mcd'] = float('inf')
        
        # F0 metrics
        try:
            f0_metrics = self.compute_f0_metrics(gen_audio, ref_audio)
            metrics.update(f0_metrics)
        except Exception as e:
            warnings.warn(f"Failed to compute F0 metrics: {e}")
            metrics['f0_rmse'] = float('inf')
            metrics['f0_correlation'] = 0.0
        
        # Duration metrics
        try:
            duration_metrics = self.compute_duration_metrics(gen_audio, ref_audio)
            metrics.update(duration_metrics)
        except Exception as e:
            warnings.warn(f"Failed to compute duration metrics: {e}")
            metrics['duration_ratio'] = 1.0
        
        # Speaker similarity (if embeddings provided)
        if reference_speaker_embedding is not None and generated_speaker_embedding is not None:
            try:
                similarity = self.compute_speaker_similarity(
                    generated_speaker_embedding,
                    reference_speaker_embedding
                )
                metrics['speaker_similarity'] = similarity
            except Exception as e:
                warnings.warn(f"Failed to compute speaker similarity: {e}")
                metrics['speaker_similarity'] = 0.0
        
        elif speaker_encoder is not None:
            try:
                # Extract embeddings from audio
                gen_emb = self._extract_speaker_embedding(gen_audio, speaker_encoder)
                ref_emb = self._extract_speaker_embedding(ref_audio, speaker_encoder)
                similarity = self.compute_speaker_similarity(gen_emb, ref_emb)
                metrics['speaker_similarity'] = similarity
            except Exception as e:
                warnings.warn(f"Failed to compute speaker similarity: {e}")
                metrics['speaker_similarity'] = 0.0
        
        # Energy metrics
        try:
            energy_metrics = self.compute_energy_metrics(gen_audio, ref_audio)
            metrics.update(energy_metrics)
        except Exception as e:
            warnings.warn(f"Failed to compute energy metrics: {e}")
        
        # STOI (Short-Time Objective Intelligibility)
        if STOI_AVAILABLE:
            try:
                stoi_score = self.compute_stoi(gen_audio, ref_audio)
                metrics['stoi'] = stoi_score
            except Exception as e:
                warnings.warn(f"Failed to compute STOI: {e}")
                metrics['stoi'] = 0.0
        
        # PESQ (Perceptual Evaluation of Speech Quality)
        if PESQ_AVAILABLE:
            try:
                pesq_score = self.compute_pesq(gen_audio, ref_audio)
                metrics['pesq'] = pesq_score
            except Exception as e:
                warnings.warn(f"Failed to compute PESQ: {e}")
                metrics['pesq'] = 0.0
        
        return metrics
    
    def compute_mcd(
        self,
        generated_audio: np.ndarray,
        reference_audio: np.ndarray
    ) -> float:
        """
        Compute Mel Cepstral Distortion (MCD) between generated and reference audio.
        
        MCD measures spectral distortion in the mel-cepstral domain.
        Lower values indicate better quality.
        
        Args:
            generated_audio: Generated audio waveform
            reference_audio: Reference audio waveform
        
        Returns:
            MCD value (dB)
        """
        # Compute mel spectrograms
        gen_mel = self._compute_mel_spectrogram(generated_audio)
        ref_mel = self._compute_mel_spectrogram(reference_audio)
        
        # Align lengths
        min_len = min(gen_mel.shape[1], ref_mel.shape[1])
        gen_mel = gen_mel[:, :min_len]
        ref_mel = ref_mel[:, :min_len]
        
        # Convert to mel cepstrum (MFCC)
        gen_mfcc = librosa.feature.mfcc(
            S=gen_mel,
            n_mfcc=13,
            sr=self.sample_rate
        )
        ref_mfcc = librosa.feature.mfcc(
            S=ref_mel,
            n_mfcc=13,
            sr=self.sample_rate
        )
        
        # Compute MCD
        diff = gen_mfcc - ref_mfcc
        mcd = np.mean(np.sqrt(np.sum(diff ** 2, axis=0)))
        
        # Convert to dB (standard MCD formula)
        mcd_db = (10.0 / np.log(10.0)) * np.sqrt(2.0) * mcd
        
        return float(mcd_db)
    
    def compute_f0_metrics(
        self,
        generated_audio: np.ndarray,
        reference_audio: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute F0 (fundamental frequency) related metrics.
        
        Args:
            generated_audio: Generated audio waveform
            reference_audio: Reference audio waveform
        
        Returns:
            Dictionary with F0 metrics
        """
        # Extract F0 using librosa
        gen_f0, gen_voiced = self._extract_f0(generated_audio)
        ref_f0, ref_voiced = self._extract_f0(reference_audio)
        
        # Align F0 sequences
        min_len = min(len(gen_f0), len(ref_f0))
        gen_f0 = gen_f0[:min_len]
        ref_f0 = ref_f0[:min_len]
        gen_voiced = gen_voiced[:min_len]
        ref_voiced = ref_voiced[:min_len]
        
        # Only compare voiced frames
        voiced_mask = gen_voiced & ref_voiced
        if voiced_mask.sum() == 0:
            return {
                'f0_rmse': float('inf'),
                'f0_correlation': 0.0,
                'f0_mean_error': float('inf')
            }
        
        gen_f0_voiced = gen_f0[voiced_mask]
        ref_f0_voiced = ref_f0[voiced_mask]
        
        # Compute RMSE
        f0_rmse = np.sqrt(np.mean((gen_f0_voiced - ref_f0_voiced) ** 2))
        
        # Compute correlation
        if len(gen_f0_voiced) > 1:
            f0_correlation = np.corrcoef(gen_f0_voiced, ref_f0_voiced)[0, 1]
            if np.isnan(f0_correlation):
                f0_correlation = 0.0
        else:
            f0_correlation = 0.0
        
        # Mean error
        f0_mean_error = np.mean(np.abs(gen_f0_voiced - ref_f0_voiced))
        
        return {
            'f0_rmse': float(f0_rmse),
            'f0_correlation': float(f0_correlation),
            'f0_mean_error': float(f0_mean_error)
        }
    
    def compute_duration_metrics(
        self,
        generated_audio: np.ndarray,
        reference_audio: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute duration-related metrics.
        
        Args:
            generated_audio: Generated audio waveform
            reference_audio: Reference audio waveform
        
        Returns:
            Dictionary with duration metrics
        """
        gen_duration = len(generated_audio) / self.sample_rate
        ref_duration = len(reference_audio) / self.sample_rate
        
        duration_ratio = gen_duration / ref_duration if ref_duration > 0 else 1.0
        duration_error = abs(gen_duration - ref_duration)
        
        return {
            'duration_ratio': float(duration_ratio),
            'duration_error': float(duration_error),
            'generated_duration': float(gen_duration),
            'reference_duration': float(ref_duration)
        }
    
    def compute_energy_metrics(
        self,
        generated_audio: np.ndarray,
        reference_audio: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute energy-related metrics.
        
        Args:
            generated_audio: Generated audio waveform
            reference_audio: Reference audio waveform
        
        Returns:
            Dictionary with energy metrics
        """
        gen_energy = np.mean(generated_audio ** 2)
        ref_energy = np.mean(reference_audio ** 2)
        
        energy_ratio = gen_energy / ref_energy if ref_energy > 0 else 1.0
        energy_error = abs(gen_energy - ref_energy)
        
        return {
            'energy_ratio': float(energy_ratio),
            'energy_error': float(energy_error)
        }
    
    def compute_stoi(
        self,
        generated_audio: np.ndarray,
        reference_audio: np.ndarray
    ) -> float:
        """
        Compute STOI (Short-Time Objective Intelligibility) score.
        
        STOI measures speech intelligibility and correlates well with human perception.
        Range: 0-1 (higher is better, typically > 0.75 is good)
        
        Args:
            generated_audio: Generated audio waveform
            reference_audio: Reference audio waveform
        
        Returns:
            STOI score (0-1)
        """
        if not STOI_AVAILABLE:
            raise ImportError("pystoi is not installed. Install with: pip install pystoi")
        
        # STOI requires specific sample rates (usually 10kHz or 16kHz)
        # Resample to 10kHz if needed (STOI standard)
        stoi_sr = 10000
        if self.sample_rate != stoi_sr:
            gen_audio_resampled = librosa.resample(gen_audio, orig_sr=self.sample_rate, target_sr=stoi_sr)
            ref_audio_resampled = librosa.resample(ref_audio, orig_sr=self.sample_rate, target_sr=stoi_sr)
        else:
            gen_audio_resampled = gen_audio
            ref_audio_resampled = reference_audio
        
        # Align lengths
        min_len = min(len(gen_audio_resampled), len(ref_audio_resampled))
        gen_audio_resampled = gen_audio_resampled[:min_len]
        ref_audio_resampled = ref_audio_resampled[:min_len]
        
        # Compute STOI
        stoi_score = stoi(ref_audio_resampled, gen_audio_resampled, stoi_sr, extended=False)
        
        return float(stoi_score)
    
    def compute_pesq(
        self,
        generated_audio: np.ndarray,
        reference_audio: np.ndarray
    ) -> float:
        """
        Compute PESQ (Perceptual Evaluation of Speech Quality) score.
        
        PESQ measures speech quality and correlates with MOS (Mean Opinion Score).
        Range: -0.5 to 4.5 (higher is better, typically > 3.0 is good)
        
        Args:
            generated_audio: Generated audio waveform
            reference_audio: Reference audio waveform
        
        Returns:
            PESQ score (-0.5 to 4.5)
        """
        if not PESQ_AVAILABLE:
            raise ImportError("pesq is not installed. Install with: pip install pesq")
        
        # PESQ requires 8kHz or 16kHz sample rate
        # Use 16kHz (more common for speech)
        pesq_sr = 16000
        
        if self.sample_rate != pesq_sr:
            gen_audio_resampled = librosa.resample(gen_audio, orig_sr=self.sample_rate, target_sr=pesq_sr)
            ref_audio_resampled = librosa.resample(reference_audio, orig_sr=self.sample_rate, target_sr=pesq_sr)
        else:
            gen_audio_resampled = generated_audio
            ref_audio_resampled = reference_audio
        
        # Align lengths
        min_len = min(len(gen_audio_resampled), len(ref_audio_resampled))
        gen_audio_resampled = gen_audio_resampled[:min_len]
        ref_audio_resampled = ref_audio_resampled[:min_len]
        
        # Ensure audio is in correct format (float32, mono)
        gen_audio_resampled = gen_audio_resampled.astype(np.float32)
        ref_audio_resampled = ref_audio_resampled.astype(np.float32)
        
        # Compute PESQ
        # pesq function signature: pesq(ref, deg, fs)
        pesq_score = pesq(pesq_sr, ref_audio_resampled, gen_audio_resampled, 'wb')
        
        return float(pesq_score)
    
    def compute_speaker_similarity(
        self,
        embedding1: torch.Tensor,
        embedding2: torch.Tensor
    ) -> float:
        """
        Compute cosine similarity between speaker embeddings.
        
        Args:
            embedding1: First speaker embedding
            embedding2: Second speaker embedding
        
        Returns:
            Cosine similarity (0-1, higher is better)
        """
        if isinstance(embedding1, np.ndarray):
            embedding1 = torch.from_numpy(embedding1).float()
        if isinstance(embedding2, np.ndarray):
            embedding2 = torch.from_numpy(embedding2).float()
        
        # Flatten if needed
        if embedding1.dim() > 1:
            embedding1 = embedding1.flatten()
        if embedding2.dim() > 1:
            embedding2 = embedding2.flatten()
        
        # Normalize
        embedding1 = F.normalize(embedding1, p=2, dim=0)
        embedding2 = F.normalize(embedding2, p=2, dim=0)
        
        # Compute cosine similarity
        similarity = torch.dot(embedding1, embedding2).item()
        
        return float(similarity)
    
    def _load_audio(
        self,
        audio: Union[str, np.ndarray, torch.Tensor]
    ) -> Tuple[np.ndarray, int]:
        """Load audio from various formats."""
        if isinstance(audio, str):
            audio_data, sr = sf.read(audio)
            if len(audio_data.shape) > 1:
                audio_data = np.mean(audio_data, axis=1)  # Convert to mono
            return audio_data, sr
        elif isinstance(audio, torch.Tensor):
            audio_np = audio.cpu().numpy()
            if len(audio_np.shape) > 1:
                audio_np = np.mean(audio_np, axis=0)
            return audio_np, self.sample_rate
        else:
            # numpy array
            if len(audio.shape) > 1:
                audio = np.mean(audio, axis=0)
            return audio, self.sample_rate
    
    def _compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:
        """Compute mel spectrogram."""
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=0,
            fmax=self.sample_rate // 2
        )
        # Convert to log scale
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec
    
    def _extract_f0(
        self,
        audio: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract F0 and voiced/unvoiced flags.
        
        Returns:
            Tuple of (f0_array, voiced_flags)
        """
        # Use librosa's pyin for F0 extraction
        f0, voiced_flag, voiced_prob = librosa.pyin(
            audio,
            fmin=self.f0_min,
            fmax=self.f0_max,
            sr=self.sample_rate,
            frame_length=self.n_fft,
            hop_length=self.hop_length
        )
        
        # Replace NaN with 0
        f0 = np.nan_to_num(f0, nan=0.0)
        voiced_flag = np.nan_to_num(voiced_flag, nan=False).astype(bool)
        
        return f0, voiced_flag
    
    def _extract_speaker_embedding(
        self,
        audio: np.ndarray,
        speaker_encoder: object
    ) -> torch.Tensor:
        """Extract speaker embedding from audio using speaker encoder."""
        # Convert to tensor if needed
        if isinstance(audio, np.ndarray):
            audio_tensor = torch.from_numpy(audio).float()
        else:
            audio_tensor = audio
        
        # Ensure correct shape
        if audio_tensor.dim() == 1:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # Extract embedding
        with torch.no_grad():
            embedding = speaker_encoder.encode_single(
                audio_tensor,
                return_numpy=False,
                input="audio"
            )
        
        return embedding


class TTSMetricsBatch:
    """
    Batch computation of TTS metrics for multiple samples.
    """
    
    def __init__(self, metrics_calculator: TTSMetrics):
        """
        Initialize batch metrics calculator.
        
        Args:
            metrics_calculator: TTSMetrics instance
        """
        self.metrics_calc = metrics_calculator
    
    def compute_batch_metrics(
        self,
        generated_audios: List[Union[str, np.ndarray]],
        reference_audios: List[Union[str, np.ndarray]],
        reference_embeddings: Optional[List[torch.Tensor]] = None,
        generated_embeddings: Optional[List[torch.Tensor]] = None,
        speaker_encoder: Optional[object] = None
    ) -> Dict[str, float]:
        """
        Compute metrics for a batch of samples.
        
        Args:
            generated_audios: List of generated audio samples
            reference_audios: List of reference audio samples
            reference_embeddings: Optional list of reference embeddings
            generated_embeddings: Optional list of generated embeddings
            speaker_encoder: Optional speaker encoder
        
        Returns:
            Dictionary with averaged metrics
        """
        assert len(generated_audios) == len(reference_audios), \
            "Number of generated and reference audios must match"
        
        all_metrics = []
        
        for i, (gen_audio, ref_audio) in enumerate(zip(generated_audios, reference_audios)):
            ref_emb = reference_embeddings[i] if reference_embeddings else None
            gen_emb = generated_embeddings[i] if generated_embeddings else None
            
            metrics = self.metrics_calc.compute_all_metrics(
                generated_audio=gen_audio,
                reference_audio=ref_audio,
                reference_speaker_embedding=ref_emb,
                generated_speaker_embedding=gen_emb,
                speaker_encoder=speaker_encoder
            )
            
            all_metrics.append(metrics)
        
        # Average metrics
        averaged_metrics = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics if key in m and not np.isinf(m[key])]
            if values:
                averaged_metrics[f'avg_{key}'] = float(np.mean(values))
                averaged_metrics[f'std_{key}'] = float(np.std(values))
        
        return averaged_metrics

