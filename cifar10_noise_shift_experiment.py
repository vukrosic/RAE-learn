#!/usr/bin/env python3
"""
CIFAR-10 Noise Schedule Shift Experiment

Compares DiT training with and without dimension-dependent noise schedule shift
on a real dataset (CIFAR-10) instead of single image overfitting.
"""

import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
import math
from stage1 import RAE
from stage2.models.lightningDiT import LightningDiT
from stage2.transport import create_transport, Sampler
from omegaconf import OmegaConf

def create_dit_model(width, depth=12, token_dim=768, num_heads=None):
    """Create a DiT model with specified width."""
    if num_heads is None:
        num_heads = max(1, width // 64)
    
    model = LightningDiT(
        input_size=16,
        patch_size=1,
        in_channels=token_dim,
        hidden_size=width,
        depth=depth,
        num_heads=num_heads,
        mlp_ratio=4.0,
        num_classes=10,  # CIFAR-10 has 10 classes
        use_qknorm=False,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
        wo_shift=False,
        use_gembed=True,
    )
    return model

def load_cifar10_subset(num_samples=1000, batch_size=32):
    """Load a subset of CIFAR-10 dataset."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    
    full_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=False, transform=transform
    )
    
    # Use a random subset
    indices = np.random.choice(len(full_dataset), num_samples, replace=False)
    subset = Subset(full_dataset, indices)
    
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return dataloader, len(subset)

def precompute_latents(rae, dataloader, device):
    """Precompute RAE latents for the entire dataset."""
    latents = []
    labels = []
    
    print("Encoding CIFAR-10 images to latents...")
    rae.eval()
    with torch.no_grad():
        for images, lbls in tqdm(dataloader, desc="Encoding"):
            images = images.to(device)
            z = rae.encode(images)
            latents.append(z.cpu())
            labels.append(lbls)
    
    latents = torch.cat(latents, dim=0)
    labels = torch.cat(labels, dim=0)
    
    print(f"Latent shape: {latents.shape}")
    return latents, labels

def train_dit(dit_model, latents, labels, transport, num_epochs, lr, device):
    """Train DiT model on precomputed latents."""
    optimizer = AdamW(dit_model.parameters(), lr=lr)
    
    # Create dataloader for latents
    dataset = torch.utils.data.TensorDataset(latents, labels)
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    
    losses = []
    dit_model.train()
    
    for epoch in range(num_epochs):
        epoch_losses = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_latents, batch_labels in pbar:
            batch_latents = batch_latents.to(device)
            batch_labels = batch_labels.to(device)
            
            optimizer.zero_grad()
            model_kwargs = dict(y=batch_labels)
            loss_dict = transport.training_losses(dit_model, batch_latents, model_kwargs)
            loss = loss_dict["loss"].mean()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1} - Average Loss: {avg_loss:.4f}")
    
    return losses

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load RAE
    print("Loading RAE...")
    rae_config = OmegaConf.create({
        'target': 'stage1.RAE',
        'params': {
            'encoder_cls': 'Dinov2withNorm',
            'encoder_config_path': 'facebook/dinov2-with-registers-base',
            'encoder_input_size': 224,
            'encoder_params': {
                'dinov2_path': 'facebook/dinov2-with-registers-base',
                'normalize': True
            },
            'decoder_config_path': 'configs/decoder/ViTXL',
            'pretrained_decoder_path': 'models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt',
            'noise_tau': 0.0,
            'reshape_to_2d': True,
            'normalization_stat_path': 'models/stats/dinov2/wReg_base/imagenet1k/stat.pt'
        }
    })
    
    from utils.model_utils import instantiate_from_config
    rae = instantiate_from_config(rae_config).to(device)
    rae.eval()
    
    # Load CIFAR-10 subset
    print(f"\nLoading {args.num_samples} CIFAR-10 images...")
    dataloader, num_samples = load_cifar10_subset(args.num_samples, batch_size=32)
    
    # Precompute latents
    latents, labels = precompute_latents(rae, dataloader, device)
    token_dim = latents.shape[1]
    
    # Calculate alpha
    effective_dim = token_dim * (16 * 16)
    base_dim = 4096
    alpha = math.sqrt(effective_dim / base_dim)
    
    print(f"\n{'='*60}")
    print(f"Dataset: CIFAR-10 ({num_samples} images)")
    print(f"Token dimension: {token_dim}")
    print(f"Effective dimension: {effective_dim:,}")
    print(f"Calculated alpha (shift): {alpha:.4f}")
    print(f"{'='*60}")
    
    # Experiment A: WITHOUT shift
    print(f"\n{'='*60}")
    print("EXPERIMENT A: Training WITHOUT noise schedule shift...")
    print(f"{'='*60}")
    
    transport_no_shift = create_transport(
        path_type='Linear',
        prediction='velocity',
        loss_weight='velocity',
        time_dist_type='uniform',
        time_dist_shift=1.0,  # No shift
    )
    
    dit_no_shift = create_dit_model(width=args.width, depth=args.depth, token_dim=token_dim).to(device)
    losses_no_shift = train_dit(
        dit_no_shift, latents, labels, transport_no_shift,
        num_epochs=args.num_epochs, lr=args.lr, device=device
    )
    
    # Experiment B: WITH shift
    print(f"\n{'='*60}")
    print(f"EXPERIMENT B: Training WITH noise schedule shift (α={alpha:.4f})...")
    print(f"{'='*60}")
    
    transport_with_shift = create_transport(
        path_type='Linear',
        prediction='velocity',
        loss_weight='velocity',
        time_dist_type='uniform',
        time_dist_shift=alpha,  # Dimension-dependent shift
    )
    
    dit_with_shift = create_dit_model(width=args.width, depth=args.depth, token_dim=token_dim).to(device)
    losses_with_shift = train_dit(
        dit_with_shift, latents, labels, transport_with_shift,
        num_epochs=args.num_epochs, lr=args.lr, device=device
    )
    
    # Plot results
    os.makedirs(args.output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    epochs = np.arange(1, len(losses_no_shift) + 1)
    plt.plot(epochs, losses_no_shift, 'o-', label=f'WITHOUT shift (α=1.0) - Final: {losses_no_shift[-1]:.4f}', linewidth=2)
    plt.plot(epochs, losses_with_shift, 's-', label=f'WITH shift (α={alpha:.2f}) - Final: {losses_with_shift[-1]:.4f}', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Average Loss', fontsize=12)
    plt.title(f'DiT Training on CIFAR-10 ({num_samples} images): Noise Schedule Shift Comparison', fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    loss_path = os.path.join(args.output_dir, 'cifar10_loss_comparison.png')
    plt.savefig(loss_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved loss comparison to {loss_path}")
    
    # Print summary
    improvement = ((losses_no_shift[-1] - losses_with_shift[-1]) / losses_no_shift[-1]) * 100
    
    print(f"\n{'='*60}")
    print("                 EXPERIMENT SUMMARY")
    print(f"{'='*60}")
    print(f"Dataset: CIFAR-10 subset ({num_samples} images)")
    print(f"Training: {args.num_epochs} epochs")
    print(f"DiT: width={args.width}, depth={args.depth}")
    print(f"\nFinal Loss WITHOUT shift (α=1.0):    {losses_no_shift[-1]:.4f}")
    print(f"Final Loss WITH shift (α={alpha:.2f}):  {losses_with_shift[-1]:.4f}")
    print(f"\nImprovement: {improvement:.1f}%")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-samples', type=int, default=2000, help='Number of CIFAR-10 samples to use')
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--width', type=int, default=768, help='DiT hidden dimension')
    parser.add_argument('--depth', type=int, default=12, help='DiT depth')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='cifar10_experiment', help='Output directory')
    args = parser.parse_args()
    main(args)

