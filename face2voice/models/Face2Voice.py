import torch
import torch.nn as nn

class Face2VoiceModel(nn.Module):
    """
    Complete face-to-voice model combining:
    1. Face encoder (ArcFace)
    2. Mapping network (face space -> voice space)
    3. Speaker encoder (speaker) - for extracting target embeddings
    """
    
    def __init__(
        self,
        face_encoder,
        speaker_encoder,
        face_dim: int = 512,
        voice_dim: int = 256,
        hidden_dims: list = [512, 384]
    ):
        """
        Initialize model.
        
        Args:
            face_encoder: Pretrained face encoder (e.g., ArcFace)
            speaker_encoder: speaker speaker encoder for targets
            face_dim: Dimension of face embeddings
            voice_dim: Dimension of voice embeddings
            hidden_dims: Hidden layer dimensions for mapping network
        """
        super().__init__()
        
        # Face encoder (frozen)
        self.face_encoder = face_encoder
        for param in self.face_encoder.parameters():
            param.requires_grad = False
        self.face_encoder.eval()
        
        # speaker encoder (frozen, for extracting targets)
        self.speaker_encoder = speaker_encoder
        for param in self.speaker_encoder.parameters():
            param.requires_grad = False
        self.speaker_encoder.eval()
        
        # Mapping network (trainable)
        layers = []
        in_dim = face_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(in_dim, voice_dim))
        
        self.mapping_network = nn.Sequential(*layers)
    
    def forward(self, face_images: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: face images -> predicted voice embeddings.
        
        Args:
            face_images: Batch of face images, shape (B, 3, H, W)
        
        Returns:
            Predicted voice embeddings, shape (B, voice_dim)
        """
        # Extract face embeddings (no gradient)
        with torch.no_grad():
            face_embeddings = self.face_encoder(face_images)
        
        # Map to voice space (with gradient)
        predicted_voice_embeddings = self.mapping_network(face_embeddings)
        
        return predicted_voice_embeddings

    def extract_batch_target_embeddings(self, audios):
        return self.speaker_encoder.encode_batch(audios)