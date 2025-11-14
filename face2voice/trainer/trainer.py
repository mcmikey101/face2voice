import torch
from torch.utils.data import DataLoader
import numpy as np
from face2voice.models.Face2Voice import Face2VoiceModel
from face2voice.losses.compound_loss import CompoundLoss
import warnings
warnings.filterwarnings("ignore")

class Trainer:
    def __init__(self,
            model: Face2VoiceModel,
            train_loader: DataLoader,
            val_loader: DataLoader,
            num_epochs: int,
            model_save_path: str,
            device: str = "cuda"):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.device = device
        self.model_save_path = model_save_path

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
            
            with torch.no_grad():
                for face_images, audio_paths in self.val_loader:
                    face_images = face_images.to(self.device)
                    
                    target_embeddings = model.extract_batch_target_embeddings(audio_paths)
                    target_embeddings = target_embeddings.to(self.device)
                    
                    predicted_embeddings = model(face_images)
                    
                    loss, loss_dict = criterion(predicted_embeddings, target_embeddings)
                    val_losses.append(loss.item())
            
            avg_val_loss = np.mean(val_losses)
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {avg_train_loss:.4f}")
            print(f"  Val Loss: {avg_val_loss:.4f}")
            print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss
                }, self.model_save_path)
                print(f"  Saved best model! Val loss: {best_val_loss:.4f}")
            
            # Update scheduler
            scheduler.step()