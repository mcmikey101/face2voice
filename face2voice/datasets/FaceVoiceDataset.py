import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from pathlib import Path
import cv2
from typing import Optional, Tuple, Dict
import os

try:
    from clearml import Dataset as ClearMLDataset, Task
    CLEARML_AVAILABLE = True
except ImportError:
    CLEARML_AVAILABLE = False
    print("ClearML not available. Install with: pip install clearml")


class VoxCelebDataset(Dataset):
    """
    PyTorch Dataset for VoxCeleb audio spectrograms and face images
    
    Args:
        csv_path: Path to processed_segments.csv
        split: 'train' or 'test'
        root_dir: Root directory containing train/test folders
        load_faces: Whether to load face images (default: True)
        transform_audio: Optional transform for spectrograms
        transform_face: Optional transform for face images
    """
    
    def __init__(
        self,
        csv_path: str,
        split: str = 'train',
        root_dir: Optional[str] = None,
        load_faces: bool = True,
        transform_audio=None,
        transform_face=None
    ):
        assert split in ['train', 'test'], "split must be 'train' or 'test'"
        
        self.csv_path = csv_path
        self.split = split
        self.load_faces = load_faces
        self.transform_audio = transform_audio
        self.transform_face = transform_face
        
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
        
        print(f"Loaded {split} split: {len(self.df)} samples, {self.num_speakers} speakers")
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        
        # Load spectrogram
        spec_path = row['spectrogram_path']
        if not Path(spec_path).is_absolute():
            spec_path = self.root_dir / spec_path
        
        # Load based on file extension
        if str(spec_path).endswith('.npz'):
            spec = np.load(spec_path)['spec']
            spec = torch.from_numpy(spec)
        else:  # .pth
            spec = torch.load(spec_path, map_location='cpu')
        
        # Ensure float32
        spec = spec.float()
        
        # Apply audio transform
        if self.transform_audio is not None:
            spec = self.transform_audio(spec)
        
        # Prepare output
        output = {
            'spectrogram': spec,
            'speaker_id': self.speaker_to_idx[row['speaker']],
            'segment_id': row['segment'],
            'speaker_name': row['speaker']
        }
        
        # Load face if available and requested
        if self.load_faces and pd.notna(row.get('face_path')):
            face_path = row['face_path']
            if not Path(face_path).is_absolute():
                face_path = self.root_dir / face_path
            
            if Path(face_path).exists():
                face = cv2.imread(str(face_path))
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = torch.from_numpy(face).permute(2, 0, 1).float() / 255.0
                
                if self.transform_face is not None:
                    face = self.transform_face(face)
                
                output['face'] = face
            else:
                output['face'] = torch.zeros(3, 112, 112)
        
        return output
    
    def get_speaker_samples(self, speaker_name: str):
        """Get all samples for a specific speaker"""
        speaker_df = self.df[self.df['speaker'] == speaker_name]
        return [self[idx] for idx in speaker_df.index]
    
    def get_num_speakers(self) -> int:
        """Get total number of unique speakers"""
        return self.num_speakers


class VoxCelebClearMLDataset:
    """
    Wrapper for VoxCeleb dataset with ClearML integration
    
    Usage:
        # Upload dataset to ClearML
        uploader = VoxCelebClearMLDataset()
        dataset_id = uploader.upload_dataset(
            local_path='/path/to/voxceleb_processed',
            dataset_name='VoxCeleb-Processed',
            dataset_project='datasets/voxceleb'
        )
        
        # Download and use dataset
        local_path = uploader.download_dataset(dataset_id)
        train_dataset = VoxCelebDataset(
            csv_path=f'{local_path}/train/audio_processed.csv',
            split='train'
        )
    """
    
    def __init__(self):
        if not CLEARML_AVAILABLE:
            raise ImportError("ClearML is required. Install with: pip install clearml")
    
    def upload_dataset(
        self,
        local_path: str,
        dataset_name: str = 'VoxCeleb-Processed',
        dataset_project: str = 'datasets/voxceleb',
        tags: Optional[list] = None
    ) -> str:
        """
        Upload VoxCeleb dataset to ClearML
        
        Args:
            local_path: Path to voxceleb_processed directory
            dataset_name: Name for the dataset in ClearML
            dataset_project: Project name in ClearML
            tags: Optional tags for the dataset
            
        Returns:
            dataset_id: ClearML dataset ID
        """
        local_path = Path(local_path)
        
        # Create ClearML dataset
        dataset = ClearMLDataset.create(
            dataset_name=dataset_name,
            dataset_project=dataset_project,
            dataset_tags=tags or ['voxceleb', 'audio', 'faces', 'spectrograms']
        )
        
        print(f"Uploading dataset from {local_path}...")
        
        # Add train spectrograms
        train_spec_dir = local_path / 'train' / 'spectrograms'
        if train_spec_dir.exists():
            dataset.add_files(
                path=str(train_spec_dir),
                dataset_path='train/spectrograms',
                recursive=True
            )
            print(f"  ✓ Added train spectrograms")
        
        # Add test spectrograms
        test_spec_dir = local_path / 'test' / 'spectrograms'
        if test_spec_dir.exists():
            dataset.add_files(
                path=str(test_spec_dir),
                dataset_path='test/spectrograms',
                recursive=True
            )
            print(f"  ✓ Added test spectrograms")
        
        # Add train faces
        train_face_dir = local_path / 'train' / 'faces'
        if train_face_dir.exists():
            dataset.add_files(
                path=str(train_face_dir),
                dataset_path='train/faces',
                recursive=True
            )
            print(f"  ✓ Added train faces")
        
        # Add test faces
        test_face_dir = local_path / 'test' / 'faces'
        if test_face_dir.exists():
            dataset.add_files(
                path=str(test_face_dir),
                dataset_path='test/faces',
                recursive=True
            )
            print(f"  ✓ Added test faces")
        
        # Add CSV files
        for csv_file in ['train/audio_processed.csv', 'test/audio_processed.csv', 
                         'audio_processed_all.csv', 'frames_processed.csv']:
            csv_path = local_path / csv_file
            if csv_path.exists():
                dataset.add_files(
                    path=str(csv_path),
                    dataset_path=csv_file
                )
                print(f"  ✓ Added {csv_file}")
        
        # Upload and finalize
        dataset.upload()
        dataset.finalize()
        
        dataset_id = dataset.id
        print(f"\n✓ Dataset uploaded successfully!")
        print(f"  Dataset ID: {dataset_id}")
        print(f"  Dataset URL: {dataset.get_default_storage()}")
        
        return dataset_id
    
    def download_dataset(
        self,
        dataset_id: Optional[str] = None,
        dataset_name: Optional[str] = None,
        dataset_project: Optional[str] = None,
        local_path: Optional[str] = None
    ) -> str:
        """
        Download VoxCeleb dataset from ClearML
        
        Args:
            dataset_id: ClearML dataset ID (provide this OR dataset_name+project)
            dataset_name: Name of dataset in ClearML
            dataset_project: Project name in ClearML
            local_path: Where to download (default: ~/.clearml/cache)
            
        Returns:
            local_path: Path to downloaded dataset
        """
        if dataset_id:
            dataset = ClearMLDataset.get(dataset_id=dataset_id)
        elif dataset_name and dataset_project:
            dataset = ClearMLDataset.get(
                dataset_name=dataset_name,
                dataset_project=dataset_project
            )
        else:
            raise ValueError("Provide either dataset_id or (dataset_name + dataset_project)")
        
        print(f"Downloading dataset: {dataset.name}")
        
        # Get local copy
        dataset_path = dataset.get_local_copy()
        
        if local_path:
            # Move to custom location if specified
            import shutil
            local_path = Path(local_path)
            local_path.mkdir(parents=True, exist_ok=True)
            
            for item in Path(dataset_path).iterdir():
                dest = local_path / item.name
                if item.is_dir():
                    shutil.copytree(item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(item, dest)
            
            dataset_path = str(local_path)
        
        print(f"✓ Dataset downloaded to: {dataset_path}")
        return dataset_path
    
    def get_dataset_info(self, dataset_id: str) -> Dict:
        """Get information about a ClearML dataset"""
        dataset = ClearMLDataset.get(dataset_id=dataset_id)
        
        info = {
            'id': dataset.id,
            'name': dataset.name,
            'project': dataset.project,
            'tags': dataset.tags,
            'created': dataset.created,
            'description': dataset.description
        }
        
        return info


# Example usage and utility functions
def create_dataloaders(
    dataset_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    load_faces: bool = True
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test dataloaders
    
    Args:
        dataset_path: Path to voxceleb_processed directory
        batch_size: Batch size for dataloaders
        num_workers: Number of worker processes
        load_faces: Whether to load face images
        
    Returns:
        train_loader, test_loader
    """
    dataset_path = Path(dataset_path)
    
    # Create datasets
    train_dataset = VoxCelebDataset(
        csv_path=str(dataset_path / 'train' / 'audio_processed.csv'),
        split='train',
        root_dir=str(dataset_path),
        load_faces=load_faces
    )
    
    test_dataset = VoxCelebDataset(
        csv_path=str(dataset_path / 'test' / 'audio_processed.csv'),
        split='test',
        root_dir=str(dataset_path),
        load_faces=load_faces
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


def collate_fn_with_faces(batch):
    """Custom collate function that handles optional faces"""
    spectrograms = torch.stack([item['spectrogram'] for item in batch])
    speaker_ids = torch.tensor([item['speaker_id'] for item in batch])
    
    output = {
        'spectrogram': spectrograms,
        'speaker_id': speaker_ids,
        'segment_id': [item['segment_id'] for item in batch],
        'speaker_name': [item['speaker_name'] for item in batch]
    }
    
    # Stack faces if available
    if 'face' in batch[0]:
        faces = torch.stack([item['face'] for item in batch])
        output['face'] = faces
    
    return output


if __name__ == "__main__":
    # Example 1: Upload dataset to ClearML
    """
    clearml_dataset = VoxCelebClearMLDataset()
    
    dataset_id = clearml_dataset.upload_dataset(
        local_path='/kaggle/working/voxceleb_processed',
        dataset_name='VoxCeleb-Processed-v1',
        dataset_project='datasets/voxceleb',
        tags=['voxceleb', 'audio', 'spectrograms', 'faces', 'v1']
    )
    
    print(f"Dataset ID: {dataset_id}")
    """
    
    # Example 2: Download and use dataset
    """
    clearml_dataset = VoxCelebClearMLDataset()
    
    # Download from ClearML
    local_path = clearml_dataset.download_dataset(
        dataset_id='your-dataset-id-here'
        # OR
        # dataset_name='VoxCeleb-Processed-v1',
        # dataset_project='datasets/voxceleb'
    )
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        dataset_path=local_path,
        batch_size=32,
        num_workers=4,
        load_faces=True
    )
    
    # Test loading
    for batch in train_loader:
        print(f"Spectrogram shape: {batch['spectrogram'].shape}")
        print(f"Face shape: {batch['face'].shape}")
        print(f"Speaker IDs: {batch['speaker_id']}")
        break
    """
    
    # Example 3: Use local dataset (no ClearML)
    """
    train_dataset = VoxCelebDataset(
        csv_path='voxceleb_processed/train/audio_processed.csv',
        split='train',
        root_dir='voxceleb_processed',
        load_faces=True
    )
    
    # Get a sample
    sample = train_dataset[0]
    print(f"Spectrogram shape: {sample['spectrogram'].shape}")
    print(f"Speaker ID: {sample['speaker_id']}")
    print(f"Speaker name: {sample['speaker_name']}")
    if 'face' in sample:
        print(f"Face shape: {sample['face'].shape}")
    
    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        num_workers=4,
        collate_fn=collate_fn_with_faces
    )
    """
    
    print("VoxCeleb Dataset classes ready!")
    print("\nQuick start:")
    print("1. Upload: clearml_dataset.upload_dataset(local_path, dataset_name)")
    print("2. Download: local_path = clearml_dataset.download_dataset(dataset_id)")
    print("3. Use: train_loader, test_loader = create_dataloaders(local_path)")