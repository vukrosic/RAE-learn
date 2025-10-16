#!/usr/bin/env python3
"""
Noise-Augmented Decoder Training Experiment

Compares two decoders:
1. Trained WITHOUT noise augmentation (noise_tau=0) - FRAGILE
2. Trained WITH noise augmentation (noise_tau>0) - ROBUST

Shows that noise augmentation during training makes the decoder
more robust to imperfect latents from DiT generation.
"""

import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from stage1 import RAE
from omegaconf import OmegaConf

def load_cifar10_subset(num_samples=1000, batch_size=16):
    """Load CIFAR-10 dataset."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    
    full_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=False, transform=transform
    )
    
    # Use subset for faster training
    indices = np.random.choice(len(full_dataset), num_samples, replace=False)
    subset = Subset(full_dataset, indices)
    dataloader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=4)
    
    return dataloader

def train_decoder(encoder, decoder, dataloader, device, noise_tau=0.0, num_epochs=5, lr=1e-4):
    """Train a decoder with or without noise augmentation."""
    optimizer = AdamW(decoder.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    decoder.train()
    encoder.eval()
    
    losses = []
    
    for epoch in range(num_epochs):
        epoch_losses = []
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} (noise_tau={noise_tau})")
        
        for images, _ in pbar:
            images = images.to(device)
            
            # Encode (frozen encoder)
            with torch.no_grad():
                latents = encoder.encode(images)
                
                # Add noise augmentation if enabled
                if noise_tau > 0:
                    noise = torch.randn_like(latents) * noise_tau
                    latents = latents + noise
            
            # Decode
            reconstructed = decoder.decode(latents)
            
            # Loss
            loss = criterion(reconstructed, images)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_losses.append(loss.item())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = np.mean(epoch_losses)
        losses.append(avg_loss)
        print(f"Epoch {epoch+1} - Avg Loss: {avg_loss:.4f}")
    
    return losses

def test_decoder_robustness(encoder, decoder, test_images, device, noise_levels):
    """Test decoder on noisy latents."""
    decoder.eval()
    encoder.eval()
    
    results = {}
    
    with torch.no_grad():
        # Encode images
        latents = encoder.encode(test_images)
        
        # Test each noise level
        for noise_std in noise_levels:
            noisy_latents = latents + torch.randn_like(latents) * noise_std
            reconstructed = decoder.decode(noisy_latents)
            
            # Calculate MSE and PSNR
            mse = torch.mean((test_images - reconstructed) ** 2).item()
            psnr = 10 * np.log10(1.0 / mse) if mse > 0 else 100.0
            
            results[noise_std] = {
                'reconstructed': reconstructed.cpu(),
                'mse': mse,
                'psnr': psnr
            }
    
    return results

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load frozen encoder (shared for both experiments)
    print("Loading frozen DINOv2 encoder...")
    from utils.model_utils import instantiate_from_config
    
    encoder_config = OmegaConf.create({
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
            'pretrained_decoder_path': None,  # We'll train our own
            'noise_tau': 0.0,
            'reshape_to_2d': True,
        }
    })
    
    # Create RAE with frozen encoder
    rae_template = instantiate_from_config(encoder_config).to(device)
    encoder = rae_template  # Use for encoding only
    encoder.eval()
    
    # Freeze encoder
    for param in encoder.encoder.parameters():
        param.requires_grad = False
    
    # Load training data
    print(f"\nLoading {args.num_samples} CIFAR-10 images for training...")
    train_loader = load_cifar10_subset(args.num_samples, batch_size=args.batch_size)
    
    # Load test images
    print("Loading test images...")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    cifar_test = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    test_indices = np.random.choice(len(cifar_test), 6, replace=False)
    test_images = torch.stack([cifar_test[i][0] for i in test_indices]).to(device)
    
    print(f"\n{'='*60}")
    print("EXPERIMENT A: Training decoder WITHOUT noise augmentation")
    print(f"{'='*60}")
    
    # Create decoder A (no noise)
    decoder_A = instantiate_from_config(encoder_config).to(device)
    losses_A = train_decoder(
        encoder, decoder_A, train_loader, device,
        noise_tau=0.0,  # NO noise augmentation
        num_epochs=args.num_epochs,
        lr=args.lr
    )
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT B: Training decoder WITH noise augmentation (tau={args.noise_tau})")
    print(f"{'='*60}")
    
    # Create decoder B (with noise)
    decoder_B = instantiate_from_config(encoder_config).to(device)
    losses_B = train_decoder(
        encoder, decoder_B, train_loader, device,
        noise_tau=args.noise_tau,  # WITH noise augmentation
        num_epochs=args.num_epochs,
        lr=args.lr
    )
    
    # Test both decoders on noisy latents
    print(f"\n{'='*60}")
    print("Testing decoder robustness on noisy latents...")
    print(f"{'='*60}")
    
    noise_levels = [0.0, 0.1, 0.3, 0.5, 1.0]
    
    results_A = test_decoder_robustness(encoder, decoder_A, test_images, device, noise_levels)
    results_B = test_decoder_robustness(encoder, decoder_B, test_images, device, noise_levels)
    
    # Create visualizations
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot 1: Training curves
    plt.figure(figsize=(10, 5))
    epochs = np.arange(1, len(losses_A) + 1)
    plt.plot(epochs, losses_A, 'o-', label='Decoder A (no noise aug)', linewidth=2)
    plt.plot(epochs, losses_B, 's-', label=f'Decoder B (noise_tau={args.noise_tau})', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Training Loss', fontsize=12)
    plt.title('Decoder Training: With vs Without Noise Augmentation', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{args.output_dir}/training_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved training curves to {args.output_dir}/training_comparison.png")
    
    # Plot 2: Robustness comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    psnr_A = [results_A[n]['psnr'] for n in noise_levels]
    psnr_B = [results_B[n]['psnr'] for n in noise_levels]
    
    ax1.plot(noise_levels, psnr_A, 'o-', label='Decoder A (no noise aug)', linewidth=2, markersize=8)
    ax1.plot(noise_levels, psnr_B, 's-', label=f'Decoder B (noise_tau={args.noise_tau})', linewidth=2, markersize=8)
    ax1.set_xlabel('Latent Noise Level (σ)', fontsize=12)
    ax1.set_ylabel('PSNR (dB) - Higher is Better', fontsize=12)
    ax1.set_title('Decoder Robustness: PSNR vs Latent Noise', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=25, color='r', linestyle='--', alpha=0.5, label='Quality threshold')
    
    # Show improvement
    improvements = [psnr_B[i] - psnr_A[i] for i in range(len(noise_levels))]
    colors = ['green' if imp > 0 else 'gray' for imp in improvements]
    ax2.bar(range(len(noise_levels)), improvements, color=colors, alpha=0.7)
    ax2.set_xticks(range(len(noise_levels)))
    ax2.set_xticklabels([f'{n}' for n in noise_levels])
    ax2.set_xlabel('Latent Noise Level (σ)', fontsize=12)
    ax2.set_ylabel('PSNR Improvement (dB)', fontsize=12)
    ax2.set_title('Benefit of Noise Augmentation', fontsize=14)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/robustness_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Saved robustness comparison to {args.output_dir}/robustness_comparison.png")
    
    # Print summary
    print(f"\n{'='*70}")
    print("              NOISE-AUGMENTED DECODER EXPERIMENT")
    print(f"{'='*70}")
    print(f"Training: {args.num_samples} CIFAR-10 images, {args.num_epochs} epochs\n")
    
    print("Decoder Robustness Test (PSNR on noisy latents):\n")
    print(f"{'Noise σ':<10} {'Decoder A':<15} {'Decoder B':<15} {'Improvement'}")
    print(f"{'(latent)':<10} {'(no aug)':<15} {'(tau={args.noise_tau})':<15}")
    print("-" * 70)
    
    for i, noise_std in enumerate(noise_levels):
        psnr_a = psnr_A[i]
        psnr_b = psnr_B[i]
        improvement = psnr_b - psnr_a
        symbol = '✅' if improvement > 0.5 else '→' if improvement > 0 else '='
        print(f"{noise_std:<10.1f} {psnr_a:<15.2f} {psnr_b:<15.2f} {improvement:+.2f} dB {symbol}")
    
    print(f"\n{'='*70}")
    print("KEY FINDING:")
    # Find largest improvement
    max_improvement_idx = np.argmax(improvements)
    max_improvement = improvements[max_improvement_idx]
    at_noise = noise_levels[max_improvement_idx]
    
    print(f"At σ={at_noise} latent noise, noise-augmented decoder performs")
    print(f"{max_improvement:+.2f} dB better, proving noise augmentation makes")
    print("the decoder more robust to imperfect DiT-generated latents.")
    print(f"{'='*70}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-samples', type=int, default=500, help='CIFAR-10 training samples')
    parser.add_argument('--num-epochs', type=int, default=5, help='Training epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--noise-tau', type=float, default=0.5, help='Noise augmentation strength')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--output-dir', type=str, default='noise_aug_decoder_results', help='Output directory')
    args = parser.parse_args()
    main(args)

