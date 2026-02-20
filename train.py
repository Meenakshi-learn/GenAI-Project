import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import utils
import os
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

# Assuming these exist in your project structure
from dataset import get_dataloaders
from model import Generator, Discriminator, initialize_weights

class Pix2PixTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Store critical config values as instance variables
        self.epochs = config.get('epochs', 100)
        self.lambda_l1 = config.get('lambda_l1', 100)
        self.save_every = config.get('save_every', 5)
        self.val_every = config.get('val_every', 1)
        self.clip_grad = config.get('clip_grad', 1.0)
        self.label_smooth = config.get('label_smooth', 0.9) # Label smoothing factor
        
        # Create save directories
        self.save_dir = Path(config.get('save_dir', 'checkpoints'))
        self.sample_dir = Path(config.get('sample_dir', 'samples'))
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)
        
        # TensorBoard writer
        log_dir = f"runs/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.writer = SummaryWriter(log_dir=log_dir)
        print(f"TensorBoard logs saved to: {log_dir}")
        
        # Initialize models
        self.gen = Generator().to(self.device)
        self.disc = Discriminator().to(self.device)
        
        initialize_weights(self.gen)
        initialize_weights(self.disc)
        
        # Loss functions
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l1_loss = nn.L1Loss()
        
        # Optimizers
        lr = config.get('lr', 2e-4)
        betas = (0.5, 0.999)
        
        self.opt_gen = optim.Adam(self.gen.parameters(), lr=lr, betas=betas)
        self.opt_disc = optim.Adam(self.disc.parameters(), lr=lr, betas=betas)
        
        # Learning rate schedulers (Linear decay)
        # Note: LambdaLR passes the current step count to the lambda
        self.scheduler_gen = optim.lr_scheduler.LambdaLR(
            self.opt_gen, lr_lambda=lambda epoch: max(0, 1 - epoch / self.epochs)
        )
        self.scheduler_disc = optim.lr_scheduler.LambdaLR(
            self.opt_disc, lr_lambda=lambda epoch: max(0, 1 - epoch / self.epochs)
        )
        
        # Mixed Precision Training
        self.use_amp = self.device.type == 'cuda' and config.get('use_amp', True)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        # Training state
        self.epoch = 0
        self.best_loss = float('inf')
        self.global_step = 0

    def train_epoch(self, train_loader):
        self.gen.train()
        self.disc.train()
        
        epoch_gen_loss = 0
        epoch_disc_loss = 0
        epoch_l1_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, (distorted, clean, _) in enumerate(pbar):
            distorted = distorted.to(self.device, non_blocking=True)
            clean = clean.to(self.device, non_blocking=True)
            
            batch_size = distorted.size(0)
            
            # --- Train Discriminator ---
            self.opt_disc.zero_grad()
            
            # Real images
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                disc_real = self.disc(distorted, clean)
                # Label smoothing: real labels = 0.9 instead of 1.0
                real_labels = torch.ones_like(disc_real) * self.label_smooth
                disc_real_loss = self.bce_loss(disc_real, real_labels)
            
            # Fake images
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                fake_clean = self.gen(distorted)
                disc_fake = self.disc(distorted, fake_clean.detach())
                fake_labels = torch.zeros_like(disc_fake)
                disc_fake_loss = self.bce_loss(disc_fake, fake_labels)
            
            disc_loss = (disc_real_loss + disc_fake_loss) / 2
            
            if self.scaler:
                self.scaler.scale(disc_loss).backward()
                self.scaler.unscale_(self.opt_disc)
                torch.nn.utils.clip_grad_norm_(self.disc.parameters(), self.clip_grad)
                self.scaler.step(self.opt_disc)
                self.scaler.update()
            else:
                disc_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.disc.parameters(), self.clip_grad)
                self.opt_disc.step()
            
            # --- Train Generator ---
            self.opt_gen.zero_grad()
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                # Re-use fake_clean if possible, but ensure gradients flow
                # In this flow, fake_clean was created in autocast context above.
                # To be safe with AMP context boundaries, we often regenerate or ensure 
                # the tensor requires grad (it does). 
                # However, to ensure consistent autocast context for gen loss:
                disc_fake_gen = self.disc(distorted, fake_clean)
                adv_loss = self.bce_loss(disc_fake_gen, real_labels) # Target real for gen
                l1_loss = self.l1_loss(fake_clean, clean)
                gen_loss = adv_loss + self.lambda_l1 * l1_loss
            
            if self.scaler:
                self.scaler.scale(gen_loss).backward()
                self.scaler.unscale_(self.opt_gen)
                torch.nn.utils.clip_grad_norm_(self.gen.parameters(), self.clip_grad)
                self.scaler.step(self.opt_gen)
                self.scaler.update()
            else:
                gen_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.gen.parameters(), self.clip_grad)
                self.opt_gen.step()
            
            # Update statistics
            epoch_disc_loss += disc_loss.item()
            epoch_gen_loss += gen_loss.item()
            epoch_l1_loss += l1_loss.item()
            self.global_step += 1
            
            # Update progress bar
            pbar.set_postfix({
                'D_loss': f"{disc_loss.item():.4f}",
                'G_loss': f"{gen_loss.item():.4f}",
                'L1': f"{l1_loss.item():.4f}"
            })
            
            # Log to TensorBoard (frequent)
            if batch_idx % 10 == 0:
                self.writer.add_scalar('Loss/Discriminator', disc_loss.item(), self.global_step)
                self.writer.add_scalar('Loss/Generator', gen_loss.item(), self.global_step)
                self.writer.add_scalar('Loss/L1', l1_loss.item(), self.global_step)
        
        # Calculate averages
        num_batches = len(train_loader)
        avg_disc_loss = epoch_disc_loss / num_batches
        avg_gen_loss = epoch_gen_loss / num_batches
        avg_l1_loss = epoch_l1_loss / num_batches
        
        return avg_disc_loss, avg_gen_loss, avg_l1_loss

    @torch.no_grad()
    def validate(self, val_loader):
        self.gen.eval()
        total_l1_loss = 0
        count = 0
        
        for distorted, clean, _ in val_loader:
            distorted = distorted.to(self.device)
            clean = clean.to(self.device)
            
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                fake_clean = self.gen(distorted)
                l1_loss = self.l1_loss(fake_clean, clean)
            
            total_l1_loss += l1_loss.item()
            count += 1
        
        avg_l1_loss = total_l1_loss / count if count > 0 else 0
        return avg_l1_loss

    @torch.no_grad()
    def save_samples(self, val_loader, epoch):
        """Save sample images to visualize progress"""
        self.gen.eval()
        
        distorted, clean, _ = next(iter(val_loader))
        distorted = distorted.to(self.device)
        clean = clean.to(self.device)
        
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            fake_clean = self.gen(distorted)
        
        # Denormalize images (Assuming input is normalized to [-1, 1])
        # Clamp to ensure valid range [0, 1] for saving
        distorted = (distorted * 0.5 + 0.5).clamp(0, 1)
        clean = (clean * 0.5 + 0.5).clamp(0, 1)
        fake_clean = (fake_clean * 0.5 + 0.5).clamp(0, 1)
        
        # Create comparison grid: [Distorted, Fake, Clean]
        # Take first 8 samples
        n_samples = min(8, distorted.size(0))
        comparison = torch.cat([distorted[:n_samples], fake_clean[:n_samples], clean[:n_samples]], dim=0)
        
        # Move to CPU for saving
        comparison = comparison.cpu()
        
        save_path = self.sample_dir / f"epoch_{epoch:04d}.png"
        utils.save_image(comparison, save_path, nrow=n_samples, padding=2)
        
        # Log to TensorBoard
        self.writer.add_image('Samples/Epoch_' + str(epoch), comparison, epoch)

    def save_checkpoint(self, epoch, loss, is_best=False):
        checkpoint = {
            'epoch': epoch + 1, # Save next epoch to resume correctly
            'gen_state_dict': self.gen.state_dict(),
            'disc_state_dict': self.disc.state_dict(),
            'opt_gen_state_dict': self.opt_gen.state_dict(),
            'opt_disc_state_dict': self.opt_disc.state_dict(),
            'scheduler_gen_state_dict': self.scheduler_gen.state_dict(),
            'scheduler_disc_state_dict': self.scheduler_disc.state_dict(),
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'loss': loss,
            'best_loss': self.best_loss,
            'global_step': self.global_step
        }
        
        save_path = self.save_dir / f"checkpoint_epoch_{epoch:04d}.pth"
        torch.save(checkpoint, save_path)
        
        if is_best:
            best_path = self.save_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
        
        # Also save generator only for inference
        gen_path = self.save_dir / f"generator_epoch_{epoch:04d}.pth"
        torch.save(self.gen.state_dict(), gen_path)

    def load_checkpoint(self, checkpoint_path):
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint not found: {checkpoint_path}")
            return
            
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.epoch = checkpoint['epoch']
        self.gen.load_state_dict(checkpoint['gen_state_dict'])
        self.disc.load_state_dict(checkpoint['disc_state_dict'])
        self.opt_gen.load_state_dict(checkpoint['opt_gen_state_dict'])
        self.opt_disc.load_state_dict(checkpoint['opt_disc_state_dict'])
        
        if 'scheduler_gen_state_dict' in checkpoint:
            self.scheduler_gen.load_state_dict(checkpoint['scheduler_gen_state_dict'])
            self.scheduler_disc.load_state_dict(checkpoint['scheduler_disc_state_dict'])
            
        if self.scaler and 'scaler_state_dict' in checkpoint and checkpoint['scaler_state_dict']:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
            
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.global_step = checkpoint.get('global_step', 0)
        print(f"Loaded checkpoint from epoch {self.epoch}")

    def train(self, train_loader, val_loader):
        print(f"Starting training for {self.epochs} epochs...")
        print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")
        
        for epoch in range(self.epoch, self.epochs):
            self.epoch = epoch
            
            # Train
            disc_loss, gen_loss, l1_loss = self.train_epoch(train_loader)
            
            # Validate
            val_loss = float('inf')
            if (epoch + 1) % self.val_every == 0:
                val_loss = self.validate(val_loader)
            
            # Update learning rates
            self.scheduler_gen.step()
            self.scheduler_disc.step()
            
            # Log epoch metrics
            self.writer.add_scalar('Epoch/Train_Disc_Loss', disc_loss, epoch)
            self.writer.add_scalar('Epoch/Train_Gen_Loss', gen_loss, epoch)
            self.writer.add_scalar('Epoch/Train_L1_Loss', l1_loss, epoch)
            self.writer.add_scalar('Epoch/Val_L1_Loss', val_loss, epoch)
            self.writer.add_scalar('Epoch/Learning_Rate', self.opt_gen.param_groups[0]['lr'], epoch)
            
            print(f"\nEpoch {epoch}/{self.epochs}")
            print(f"  Train - D Loss: {disc_loss:.4f}, G Loss: {gen_loss:.4f}, L1: {l1_loss:.4f}")
            print(f"  Val - L1 Loss: {val_loss:.4f}")
            print(f"  LR: {self.opt_gen.param_groups[0]['lr']:.6f}")
            
            # Save samples
            if (epoch + 1) % self.save_every == 0:
                self.save_samples(val_loader, epoch)
            
            # Save checkpoint
            is_best = val_loss < self.best_loss
            if is_best:
                self.best_loss = val_loss
            
            if (epoch + 1) % self.save_every == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)
                if is_best:
                    print(f"  ✓ New best model saved! Val Loss: {val_loss:.4f}")
        
        print("\nTraining completed!")
        self.writer.close()

def main():
    config = {
        'processed_dir': 'data/processed',
        'batch_size': 16,
        'epochs': 50,
        'lr': 2e-4,
        'lambda_l1': 100,
        'num_workers': 4,
        'val_split': 0.1,
        'save_every': 5,
        'val_every': 1,
        'save_dir': 'checkpoints',
        'sample_dir': 'samples',
        'use_amp': True,          # Enable Mixed Precision
        'clip_grad': 1.0,         # Gradient clipping norm
        'label_smooth': 0.9,      # Label smoothing for Discriminator
        'resume_checkpoint': None # Path to checkpoint to resume training
    }
    
    # Create dataloaders
    train_loader, val_loader = get_dataloaders(
        config['processed_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        val_split=config['val_split']
    )
    
    # Create trainer
    trainer = Pix2PixTrainer(config)
    
    # Resume training if checkpoint provided
    if config['resume_checkpoint']:
        trainer.load_checkpoint(config['resume_checkpoint'])
    
    # Train
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()