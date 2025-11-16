import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from typing import Optional, Tuple, Dict
from openvoice.mel_processing import spectrogram_torch
import torchaudio
import os

class FaceVoiceDataset(Dataset):
    """
    PyTorch Dataset for VoxCeleb audio spectrograms and face images
    """
    def __init__(
        self,
        audio_base_path,
        face_image_base_path,
        csv_path: str,
        split: str = 'train',
        root_dir: Optional[str] = None,
        load_faces: bool = True,
        transform_audio=None,
        transform_face=None,
        device = "cpu"
    ):
        assert split in ['train', 'test'], "split must be 'train' or 'test'"
        
        self.csv_path = csv_path
        self.audio_base_path = audio_base_path
        self.face_image_base_path = face_image_base_path
        self.split = split
        self.load_faces = load_faces
        self.transform_audio = transform_audio
        self.transform_face = transform_face
        self.max_length = 22050 * 10
        self.device = device

        
        # Load metadata
        self.df = pd.read_csv(csv_path)
        
        # Filter by split if csv contains both
        if 'identification' in self.df.columns:
            split_code = 'trn' if split == 'train' else 'tst'
            self.df = self.df[self.df['identification'] == split_code].reset_index(drop=True)
        
        # Determine root directory
        if root_dir is None:
            root_dir = Path(csv_path).parent.parent
        self.root_dir = Path(root_dir)
        
        # Create speaker label mapping
        self.speakers = sorted(self.df['speaker'].unique())
        self.speaker_to_idx = {spk: idx for idx, spk in enumerate(self.speakers)}
        self.num_speakers = len(self.speakers)
        
        # Count samples per speaker
        samples_per_speaker = self.df.groupby('speaker').size()
        avg_samples = samples_per_speaker.mean()
        
        print(f"Loaded {split} split:")
        print(f"  Total samples: {len(self.df)}")
        print(f"  Unique speakers: {self.num_speakers}")
        print(f"  Avg samples per speaker: {avg_samples:.1f}")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def _load_face(self, speaker_name: str) -> Optional[torch.Tensor]:        
        face_path = os.path.join(self.face_image_base_path, speaker_name + ".jpg").replace("/", "\\")
        
        face = Image.open(face_path).convert("RGB")
        
        if self.transform_face is not None:
            face = self.transform_face(face).to(self.device)
        
        return face
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        speaker_name = row['speaker']
        
        # Load spectrogram
        audio_path = os.path.join(self.audio_base_path, row['segment']).replace("/", "\\")

        try: 
            
            # Load based on file extension
            if str(audio_path).endswith('.npz'):
                spec = np.load(audio_path)['spec']
                spec = torch.from_numpy(spec)
            else:  # .wav
                waveform, sr = torchaudio.load(audio_path)
                
                cur_len = waveform.shape[-1]
                if cur_len >= self.max_length:
                    return waveform[:, :self.max_length]
                pad_len = self.max_length - cur_len
                waveform = torch.nn.functional.pad(waveform, (0, pad_len))
            
                # Resample if needed
                if sr != 22050:
                    resampler = torchaudio.transforms.Resample(sr, 22050)
                    waveform = resampler(waveform)
            
                # Convert to mono
                if waveform.shape[0] > 1:
                    waveform = waveform.mean(dim=0, keepdim=True)
                
                # Generate spectrogram
                waveform = waveform.to(self.device)
                spec = spectrogram_torch(y=waveform, sampling_rate=22050,
                    n_fft=1024,
                    hop_size=256,
                    win_size=1024
                    )
            
            # Ensure float32
            spec = spec.float()
            
            # Apply audio transform
            if self.transform_audio is not None:
                spec = self.transform_audio(spec)
            
            # Prepare output
            output = {
                'spectrogram': spec,
                'speaker_id': self.speaker_to_idx[speaker_name],
                'segment_id': row['segment'],
                'speaker_name': speaker_name
            }

            face = self._load_face(speaker_name)
            if face is not None:
                output['face'] = face

            return output

        except Exception as e:
            print(e)
            return None
    
    def get_speaker_samples(self, speaker_name: str):
        """Get all samples for a specific speaker"""
        speaker_df = self.df[self.df['speaker'] == speaker_name]
        return [self[idx] for idx in speaker_df.index]
    
    def get_num_speakers(self) -> int:
        """Get total number of unique speakers"""
        return self.num_speakers
    
    def get_samples_per_speaker(self) -> Dict[str, int]:
        """Get number of samples for each speaker"""
        return self.df.groupby('speaker').size().to_dict()