import torch
from torch.utils.data import DataLoader
import numpy as np
from face2voice.models.Face2Voice import Face2VoiceModel
from face2voice.losses.compound_loss import CompoundLoss
from face2voice.metrics import TTSMetrics
import warnings
import os
warnings.filterwarnings("ignore")

class Trainer:
    def __init__(self,
            model: Face2VoiceModel,
            train_loader: DataLoader,
            val_loader: DataLoader,
            num_epochs: int,
            model_save_path: str,
            opt_ckpt_path = None,
            device: str = "cuda",
            compute_tts_metrics: bool = False,
            tts_metrics_sample_rate: int = 24000):
        """
        Initialize Trainer.
        
        Args:
            model: Face2Voice model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            num_epochs: Number of training epochs
            model_save_path: Path to save model checkpoints
            opt_ckpt_path: Optional path to optimizer checkpoint
            device: Device to train on
            compute_tts_metrics: Whether to compute TTS metrics during validation
            tts_metrics_sample_rate: Sample rate for TTS metrics computation
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.device = device
        self.model_save_path = model_save_path
        self.opt_ckpt_path = opt_ckpt_path
        self.compute_tts_metrics = compute_tts_metrics
        
        # Initialize TTS metrics if enabled
        if self.compute_tts_metrics:
            self.tts_metrics = TTSMetrics(
                sample_rate=tts_metrics_sample_rate,
                device=device
            )

    def train_face_to_voice(self):
        """
        Training loop for face-to-voice model.
        
        Args:
            model: Face2Voice model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            num_epochs: Number of training epochs
            device: Device to train on
        """
        model = self.model.to(self.device)
        
        # Optimizer (only for mapping network)
        optimizer = torch.optim.AdamW(
            model.mapping_network.parameters(),
            lr=1e-4,
            weight_decay=5e-4
        )
        if self.opt_ckpt_path is not None:
            optimizer.load_state_dict(torch.load(self.opt_ckpt_path))

        # Scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.num_epochs,
            eta_min=1e-6
        )
        
        # Loss function
        criterion = CompoundLoss()
        
        best_val_loss = float('inf')
        
        for epoch in range(1, self.num_epochs + 1):
            print(f"\n{'='*60}")
            print(f"Epoch {epoch}/{self.num_epochs}")
            print(f"{'='*60}")
            
            # Training
            model.train()
            model.face_encoder.eval()  # Keep frozen encoders in eval
            model.speaker_encoder.eval()
            
            train_losses = []
            
            for batch_idx, batch in enumerate(self.train_loader):
                face_images = batch["face"].to(self.device)
                
                # Extract target embeddings from audio
                target_embeddings = model.extract_batch_target_embeddings(batch["spectrogram"].to(self.device))
                target_embeddings = target_embeddings.to(self.device)
                
                # Forward pass
                predicted_embeddings = model(face_images)
                
                # Compute loss
                loss, loss_dict = criterion(predicted_embeddings, target_embeddings)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.mapping_network.parameters(), 1.0)
                optimizer.step()
                
                train_losses.append(loss.item())
                
                # Log
                if batch_idx % 10 == 0:
                    print(f"Batch {batch_idx}/{len(self.train_loader)}")
                    print(f"  Loss: {loss_dict['total']:.4f}")
                    print(f"  Cosine: {loss_dict['cosine']:.4f}")
                    print(f"  MSE: {loss_dict['mse']:.4f}")
                    print(f"  Contrastive: {loss_dict['contrastive']:.4f}")
            
            avg_train_loss = np.mean(train_losses)
            
            # Validation
            model.eval()
            val_losses = []
            val_metrics_summary = {}
            
            with torch.no_grad():
                for batch in self.val_loader:
                    face_images = batch["face"].to(self.device)
                    
                    target_embeddings = model.extract_batch_target_embeddings(batch["spectrogram"].to(self.device))
                    target_embeddings = target_embeddings.to(self.device)
                    
                    predicted_embeddings = model(face_images)
                    
                    loss, loss_dict = criterion(predicted_embeddings, target_embeddings)
                    val_losses.append(loss.item())
                    
                    # Compute TTS metrics if enabled (only on first batch to save time)
                    if self.compute_tts_metrics and len(val_losses) == 1:
                        try:
                            # Compute speaker similarity metric
                            pred_norm = torch.nn.functional.normalize(predicted_embeddings, p=2, dim=1)
                            target_norm = torch.nn.functional.normalize(target_embeddings, p=2, dim=1)
                            cosine_sim = torch.nn.functional.cosine_similarity(pred_norm, target_norm, dim=1)
                            val_metrics_summary['val_speaker_similarity'] = cosine_sim.mean().item()
                        except Exception as e:
                            warnings.warn(f"Failed to compute TTS metrics: {e}")
            
            avg_val_loss = np.mean(val_losses)
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}")
            if val_metrics_summary:
                for metric_name, metric_value in val_metrics_summary.items():
                    print(f"  {metric_name}: {metric_value:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss
                }
                # Add metrics to checkpoint if available
                if val_metrics_summary:
                    checkpoint['metrics'] = val_metrics_summary
                
                torch.save(checkpoint, os.path.join(self.model_save_path, "face2voice_ckpt.pth"))
                torch.save(optimizer.state_dict(), os.path.join(self.model_save_path, "optimizer_ckpt.pth"))
                print(f"  Saved best model! Val loss: {best_val_loss:.4f}")
            
            # Update scheduler
            scheduler.step()