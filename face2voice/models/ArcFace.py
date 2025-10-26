import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import yaml
from pathlib import Path


class ArcFaceEncoder(nn.Module):
    """
    Pretrained ArcFace face encoder wrapper for extracting face embeddings.
    
    Supports multiple backbone architectures and provides flexible configuration
    for fine-tuning and feature extraction.
    """
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[dict] = None):
        """
        Initialize ArcFace encoder.
        
        Args:
            config_path: Path to YAML configuration file
            config_dict: Dictionary with configuration (alternative to config_path)
        """
        super().__init__()
        
        # Load configuration
        if config_path is not None:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif config_dict is not None:
            self.config = config_dict
        else:
            # Default configuration
            self.config = self._get_default_config()
        
        # Build model
        self._build_model()
        
        # Load pretrained weights if specified
        if self.config['pretrained']['enabled']:
            self._load_pretrained_weights()
        
        # Set training mode
        self._configure_training_mode()
    
    def _get_default_config(self) -> dict:
        """Get default configuration."""
        return {
            'model': {
                'backbone': 'resnet50',
                'embedding_dim': 512,
                'input_size': 112,
                'dropout': 0.0,
                'use_batchnorm': True
            },
            'pretrained': {
                'enabled': True,
                'weights_path': 'pretrained/arcface_resnet50.pth',
                'strict_load': True
            },
            'training': {
                'freeze_backbone': True,
                'freeze_bn': True,
                'trainable_layers': [],  # e.g., ['layer4', 'fc']
                'learning_rate': 1e-4,
                'weight_decay': 5e-4
            },
            'augmentation': {
                'enabled': True,
                'horizontal_flip': True,
                'color_jitter': False
            }
        }
    
    def _build_model(self):
        """Build the backbone architecture."""
        backbone_type = self.config['model']['backbone']
        embedding_dim = self.config['model']['embedding_dim']
        dropout = self.config['model']['dropout']
        
        if backbone_type == 'resnet18':
            from torchvision.models import resnet18
            self.backbone = resnet18(pretrained=False)
            in_features = 512
        elif backbone_type == 'resnet34':
            from torchvision.models import resnet34
            self.backbone = resnet34(pretrained=False)
            in_features = 512
        elif backbone_type == 'resnet50':
            from torchvision.models import resnet50
            self.backbone = resnet50(pretrained=False)
            in_features = 2048
        elif backbone_type == 'resnet100':
            # For ResNet-100, you'd need a custom implementation or use
            # a library like insightface's backbones
            raise NotImplementedError("ResNet-100 requires custom implementation")
        else:
            raise ValueError(f"Unsupported backbone: {backbone_type}")
        
        # Remove the final classification layer
        self.backbone.fc = nn.Identity()
        
        # Add feature extraction head
        layers = []
        
        if self.config['model']['use_batchnorm']:
            layers.append(nn.BatchNorm1d(in_features))
        
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
        
        layers.append(nn.Linear(in_features, embedding_dim))
        
        if self.config['model']['use_batchnorm']:
            layers.append(nn.BatchNorm1d(embedding_dim))
        
        self.embedding_head = nn.Sequential(*layers)
        
        # Store feature dimension
        self.embedding_dim = embedding_dim
    
    def _load_pretrained_weights(self):
        """Load pretrained ArcFace weights."""
        weights_path = self.config['pretrained']['weights_path']
        strict = self.config['pretrained']['strict_load']
        
        if not Path(weights_path).exists():
            print(f"Warning: Pretrained weights not found at {weights_path}")
            print("Model will use random initialization.")
            return
        
        try:
            state_dict = torch.load(weights_path, map_location='cpu')
            
            # Handle different state dict formats
            if 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']
            
            # Remove 'module.' prefix if present (from DataParallel)
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            
            # Load weights
            missing_keys, unexpected_keys = self.load_state_dict(state_dict, strict=strict)
            
            if missing_keys:
                print(f"Missing keys: {missing_keys}")
            if unexpected_keys:
                print(f"Unexpected keys: {unexpected_keys}")
            
            print(f"Loaded pretrained weights from {weights_path}")
        
        except Exception as e:
            print(f"Error loading pretrained weights: {e}")
            print("Model will use random initialization.")
    
    def _configure_training_mode(self):
        """Configure which parts of the model are trainable."""
        freeze_backbone = self.config['training']['freeze_backbone']
        freeze_bn = self.config['training']['freeze_bn']
        trainable_layers = self.config['training']['trainable_layers']
        
        if freeze_backbone:
            # Freeze all backbone parameters
            for param in self.backbone.parameters():
                param.requires_grad = False
            
            # Unfreeze specific layers if specified
            if trainable_layers:
                for layer_name in trainable_layers:
                    if hasattr(self.backbone, layer_name):
                        layer = getattr(self.backbone, layer_name)
                        for param in layer.parameters():
                            param.requires_grad = True
                        print(f"Unfroze layer: {layer_name}")
        
        if freeze_bn:
            # Set all BatchNorm layers to eval mode
            for module in self.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    module.eval()
                    for param in module.parameters():
                        param.requires_grad = False
    
    def forward(self, x: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """
        Forward pass to extract face embeddings.
        
        Args:
            x: Input images, shape (B, 3, H, W)
            normalize: Whether to L2-normalize the embeddings
        
        Returns:
            Face embeddings, shape (B, embedding_dim)
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Get embeddings
        embeddings = self.embedding_head(features)
        
        # Normalize if requested
        if normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        
        return embeddings
    
    def extract_features(self, x: torch.Tensor, layer: str = 'embedding') -> torch.Tensor:
        """
        Extract features from intermediate layers.
        
        Args:
            x: Input images
            layer: Which layer to extract from ('backbone', 'embedding')
        
        Returns:
            Features from the specified layer
        """
        if layer == 'backbone':
            return self.backbone(x)
        elif layer == 'embedding':
            return self.forward(x, normalize=False)
        else:
            raise ValueError(f"Unknown layer: {layer}")
    
    def get_trainable_parameters(self):
        """Get list of trainable parameters for optimizer."""
        return [p for p in self.parameters() if p.requires_grad]
    
    def train(self, mode: bool = True):
        """Override train method to handle frozen BatchNorm."""
        super().train(mode)
        
        # Keep BatchNorm in eval mode if configured
        if self.config['training']['freeze_bn']:
            for module in self.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
                    module.eval()
        
        return self