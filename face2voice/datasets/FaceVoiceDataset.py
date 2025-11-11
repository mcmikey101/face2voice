import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms


class FaceVoiceDataset(Dataset):
    """
    Dataset for audio spectrograms (npz) paired with face images.
    """
    
    def __init__(self, audio_dir, face_dir, transform=None):
        """
        Args:
            audio_dir: Directory containing .npz spectrogram files
            face_dir: Directory containing face images
            transform: Optional transform for face images
        """
        self.audio_dir = Path(audio_dir)
        self.face_dir = Path(face_dir)
        
        # Get all audio files
        self.audio_files = sorted(list(self.audio_dir.glob('*.npz')))
        
        # Get all face images
        self.face_files = sorted(list(self.face_dir.glob('*.[jp][pn]g')))  # jpg, jpeg, png
        
        # Default transform for faces if none provided
        self.transform = transform if transform else transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Create mapping from audio to face
        self._create_audio_face_mapping()
        
        print(f"Loaded {len(self.audio_files)} audio files and {len(self.face_files)} face images")
    
    def _create_audio_face_mapping(self):
        """
        Create mapping between audio files and face images.
        Assumes filenames contain speaker/identity information.
        Modify this based on your naming convention.
        """
        # Example: if files are named like 'speaker001_audio_001.npz' and 'speaker001.jpg'
        # Extract speaker ID from audio filename and map to corresponding face
        
        self.audio_to_face = {}
        
        # Create face lookup dict
        face_lookup = {}
        for face_path in self.face_files:
            # Extract speaker ID from face filename (modify as needed)
            speaker_id = face_path.stem.split('_')[0]  # e.g., 'speaker001' from 'speaker001.jpg'
            face_lookup[speaker_id] = face_path
        
        # Map each audio file to its face
        for idx, audio_path in enumerate(self.audio_files):
            # Extract speaker ID from audio filename (modify as needed)
            speaker_id = audio_path.stem.split('_')[0]  # e.g., 'speaker001' from 'speaker001_audio_001.npz'
            
            if speaker_id in face_lookup:
                self.audio_to_face[idx] = face_lookup[speaker_id]
            else:
                # Handle missing face (use default or skip)
                self.audio_to_face[idx] = None
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict with keys:
                - 'audio': torch tensor of spectrogram
                - 'face': torch tensor of face image
                - 'audio_path': path to audio file
                - 'face_path': path to face file
        """
        # Load audio spectrogram from npz
        audio_path = self.audio_files[idx]
        audio_data = np.load(audio_path)
        
        # Assume spectrogram is stored under 'spectrogram' key
        # Modify key name if different in your npz files
        if 'spectrogram' in audio_data:
            spectrogram = audio_data['spectrogram']
        else:
            # Use first array in npz if no 'spectrogram' key
            spectrogram = audio_data[list(audio_data.keys())[0]]
        
        # Convert to torch tensor
        audio_tensor = torch.from_numpy(spectrogram).float()
        
        # Add channel dimension if needed (C, H, W)
        if audio_tensor.ndim == 2:
            audio_tensor = audio_tensor.unsqueeze(0)
        
        # Load corresponding face image
        face_path = self.audio_to_face.get(idx)
        
        if face_path is not None and face_path.exists():
            face_img = Image.open(face_path).convert('RGB')
            face_tensor = self.transform(face_img)
        else:
            # Return blank/zero tensor if face not found
            face_tensor = torch.zeros(3, 112, 112)
        
        return {
            'audio': audio_tensor,
            'face': face_tensor,
            'audio_path': str(audio_path),
            'face_path': str(face_path) if face_path else None
        }


# Example usage
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    # Initialize dataset
    dataset = FaceVoiceDataset(
        audio_dir='path/to/spectrograms',
        face_dir='path/to/faces'
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4
    )
    
    # Test iteration
    for batch in dataloader:
        audio = batch['audio']  # (B, C, H, W)
        faces = batch['face']   # (B, 3, 160, 160)
        
        print(f"Audio batch shape: {audio.shape}")
        print(f"Face batch shape: {faces.shape}")
        break