"""
Training script for Language-Conditioned Diffusion Policy
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import wandb
from datetime import datetime

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from lcdp.models.diffusion_policy import DiffusionPolicy
from lcdp.data.dataset import RobotDataset, collate_fn


class Trainer:
    """Training manager for Diffusion Policy"""
    
    def __init__(self, config: dict, config_path: str):
        self.config = config
        self.device = config['device']
        
        # Set random seed
        torch.manual_seed(config['seed'])
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config['seed'])
        
        # Create model
        print("Creating model...")
        self.model = DiffusionPolicy(
            action_dim=config['model']['action_dim'],
            action_horizon=config['model']['action_horizon'],
            vision_encoder=config['model']['vision_encoder'],
            vision_feature_dim=config['model']['vision_feature_dim'],
            freeze_vision_backbone=config['model']['freeze_vision_backbone'],
            language_model=config['model']['language_model'],
            language_feature_dim=config['model']['language_feature_dim'],
            freeze_language=config['model']['freeze_language'],
            conditioning_type=config['model']['conditioning_type'],
            unet_base_channels=config['model']['unet_base_channels'],
            unet_channel_mult=tuple(config['model']['unet_channel_mult']),
            unet_num_res_blocks=config['model']['unet_num_res_blocks'],
            num_diffusion_steps=config['model']['num_diffusion_steps'],
            beta_schedule=config['model']['beta_schedule'],
            prediction_type=config['model']['prediction_type'],
            dropout=config['model']['dropout'],
            device=self.device
        )
        
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M")
        
        # Create datasets
        print("Loading data...")
        full_dataset = RobotDataset(
            data_path=config['dataset']['data_path'],
            horizon=config['dataset']['horizon'],
            obs_horizon=config['dataset']['obs_horizon'],
            action_horizon=config['dataset']['action_horizon'],
            image_size=tuple(config['dataset']['image_size']),
            normalize_actions=config['dataset']['normalize_actions'],
            augment=config['dataset']['augment'],
            file_format=config['dataset']['file_format']
        )
        
        # Split into train and validation
        val_size = int(len(full_dataset) * config['training']['val_split'])
        train_size = len(full_dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(
            full_dataset, [train_size, val_size]
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=config['training']['num_workers'],
            collate_fn=collate_fn,
            pin_memory=config['training']['pin_memory']
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training']['num_workers'],
            collate_fn=collate_fn,
            pin_memory=config['training']['pin_memory']
        )
        
        print(f"Train samples: {len(self.train_dataset)}, Val samples: {len(self.val_dataset)}")
        
        # Create optimizer
        if config['training']['optimizer'] == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay'],
                betas=tuple(config['training']['betas'])
            )
        elif config['training']['optimizer'] == 'adamw':
            self.optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=config['training']['learning_rate'],
                weight_decay=config['training']['weight_decay'],
                betas=tuple(config['training']['betas'])
            )
        
        # Create learning rate scheduler
        if config['training']['lr_scheduler'] == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config['training']['num_epochs']
            )
        elif config['training']['lr_scheduler'] == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=30,
                gamma=0.1
            )
        else:
            self.scheduler = None
        
        # Setup checkpointing
        self.checkpoint_dir = Path(config['training']['checkpoint_dir'])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.use_wandb = config['training']['use_wandb']
        if self.use_wandb:
            wandb.init(
                project=config['training']['wandb_project'],
                entity=config['training']['wandb_entity'],
                config=config,
                name=f"lcdp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        self.global_step = 0
        self.start_epoch = 0
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0.0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            images = batch['image'].to(self.device)
            instructions = batch['instruction']
            actions = batch['actions'].to(self.device)
            
            # Forward pass
            loss_dict = self.model.compute_loss(actions, images, instructions)
            loss = loss_dict['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config['training']['grad_clip'] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['grad_clip']
                )
            
            self.optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            self.global_step += 1
            
            # Logging
            if self.global_step % self.config['training']['log_every'] == 0:
                if self.use_wandb:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/lr': self.optimizer.param_groups[0]['lr'],
                        'train/epoch': epoch
                    }, step=self.global_step)
            
            pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = epoch_loss / len(self.train_loader)
        return avg_loss
    
    def validate(self, epoch: int):
        """Validate the model"""
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                images = batch['image'].to(self.device)
                instructions = batch['instruction']
                actions = batch['actions'].to(self.device)
                
                loss_dict = self.model.compute_loss(actions, images, instructions)
                val_loss += loss_dict['loss'].item()
        
        avg_val_loss = val_loss / len(self.val_loader)
        
        if self.use_wandb:
            wandb.log({
                'val/loss': avg_val_loss,
                'val/epoch': epoch
            }, step=self.global_step)
        
        return avg_val_loss
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'global_step': self.global_step,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"
        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / "best.pth"
            torch.save(checkpoint, best_path)
            print(f"Saved best model: {best_path}")
    
    def train(self):
        """Main training loop"""
        best_val_loss = float('inf')
        
        print("Starting training...")
        for epoch in range(self.start_epoch, self.config['training']['num_epochs']):
            # Train
            train_loss = self.train_epoch(epoch)
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")
            
            # Validate
            if (epoch + 1) % self.config['training']['val_every'] == 0:
                val_loss = self.validate(epoch)
                print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")
                
                # Save best model
                is_best = val_loss < best_val_loss
                if is_best:
                    best_val_loss = val_loss
                    print(f"New best validation loss: {best_val_loss:.4f}")
                
                # Save checkpoint
                if (epoch + 1) % self.config['training']['save_every'] == 0:
                    self.save_checkpoint(epoch, is_best=is_best)
            
            # Update learning rate
            if self.scheduler is not None:
                self.scheduler.step()
        
        print("Training completed!")
        if self.use_wandb:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Train Diffusion Policy")
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_config.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--data',
        type=str,
        default=None,
        help='Override data path'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Override checkpoint directory'
    )
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line args
    if args.data is not None:
        config['dataset']['data_path'] = args.data
    if args.output is not None:
        config['training']['checkpoint_dir'] = args.output
    
    # Create trainer and train
    trainer = Trainer(config, args.config)
    trainer.train()


if __name__ == "__main__":
    main()
