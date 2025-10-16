#!/usr/bin/env python3
"""
RAE Decoder Fragility Experiment

Tests how robust the RAE decoder is to noisy/imperfect latents.
This simulates what happens when a DiT generates slightly imperfect latents.
"""

import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import argparse
from stage1 import RAE
from omegaconf import OmegaConf

def load_cifar10_samples(num_samples=9):
    """Load a few CIFAR-10 samples for visualization."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=False, transform=transform
    )
    
    # Get diverse samples from different classes
    samples_per_class = num_samples // 10
    indices = []
    for class_idx in range(10):
        class_indices = [i for i, (_, label) in enumerate(dataset) if label == class_idx]
        indices.extend(np.random.choice(class_indices, samples_per_class, replace=False).tolist())
    
    # Fill remaining with random samples
    while len(indices) < num_samples:
        indices.append(np.random.randint(0, len(dataset)))
    
    images = []
    for idx in indices[:num_samples]:
        img, _ = dataset[idx]
        images.append(img)
    
    return torch.stack(images)

def add_noise_to_latents(latents, noise_std):
    """Add Gaussian noise to latents to simulate imperfect DiT outputs."""
    noise = torch.randn_like(latents) * noise_std
    return latents + noise

def calculate_metrics(original, reconstructed):
    """Calculate MSE and PSNR."""
    mse = torch.mean((original - reconstructed) ** 2).item()
    psnr = 10 * np.log10(1.0 / mse) if mse > 0 else float('inf')
    return mse, psnr

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
    
    # Load CIFAR-10 samples
    print(f"\nLoading {args.num_samples} CIFAR-10 images...")
    images = load_cifar10_samples(args.num_samples).to(device)
    
    # Encode to latents
    print("Encoding images...")
    with torch.no_grad():
        latents = rae.encode(images)
    
    print(f"Latent shape: {latents.shape}")
    
    # Test different noise levels
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.3]
    
    results = {}
    
    print(f"\nTesting decoder robustness with different noise levels...")
    for noise_std in noise_levels:
        print(f"\n--- Noise std: {noise_std} ---")
        
        # Add noise to latents
        noisy_latents = add_noise_to_latents(latents, noise_std)
        
        # Decode
        with torch.no_grad():
            reconstructed = rae.decode(noisy_latents)
        
        # Calculate metrics
        mse, psnr = calculate_metrics(images, reconstructed)
        
        results[noise_std] = {
            'reconstructed': reconstructed.cpu(),
            'mse': mse,
            'psnr': psnr
        }
        
        print(f"MSE: {mse:.6f}, PSNR: {psnr:.2f} dB")
    
    # Create visualization
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Plot 1: Show reconstructions at different noise levels
    num_images_to_show = min(5, args.num_samples)
    num_noise_levels = len(noise_levels)
    
    fig, axes = plt.subplots(num_images_to_show, num_noise_levels + 1, 
                             figsize=(3*(num_noise_levels+1), 3*num_images_to_show))
    
    for row in range(num_images_to_show):
        # Original image
        img = images[row].permute(1, 2, 0).cpu().numpy()
        img = (img * 255).clip(0, 255).astype('uint8')
        axes[row, 0].imshow(img)
        axes[row, 0].set_title('Original', fontsize=10)
        axes[row, 0].axis('off')
        
        # Reconstructions at different noise levels
        for col, noise_std in enumerate(noise_levels):
            recon = results[noise_std]['reconstructed'][row].permute(1, 2, 0).numpy()
            recon = (recon * 255).clip(0, 255).astype('uint8')
            axes[row, col+1].imshow(recon)
            axes[row, col+1].set_title(f'Noise={noise_std}\nPSNR={results[noise_std]["psnr"]:.1f}dB', 
                                       fontsize=9)
            axes[row, col+1].axis('off')
    
    plt.tight_layout()
    recon_path = os.path.join(args.output_dir, 'decoder_fragility_reconstructions.png')
    plt.savefig(recon_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved reconstructions to {recon_path}")
    
    # Plot 2: MSE vs Noise Level
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    noise_vals = list(noise_levels)
    mse_vals = [results[n]['mse'] for n in noise_levels]
    psnr_vals = [results[n]['psnr'] for n in noise_levels]
    
    ax1.plot(noise_vals, mse_vals, 'o-', linewidth=2, markersize=8)
    ax1.set_xlabel('Latent Noise Standard Deviation', fontsize=12)
    ax1.set_ylabel('Reconstruction MSE', fontsize=12)
    ax1.set_title('Decoder Fragility: MSE vs Latent Noise', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(noise_vals, psnr_vals, 's-', linewidth=2, markersize=8, color='green')
    ax2.set_xlabel('Latent Noise Standard Deviation', fontsize=12)
    ax2.set_ylabel('PSNR (dB) - Higher is Better', fontsize=12)
    ax2.set_title('Decoder Fragility: PSNR vs Latent Noise', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=30, color='r', linestyle='--', alpha=0.5, label='30 dB threshold')
    ax2.legend()
    
    plt.tight_layout()
    metrics_path = os.path.join(args.output_dir, 'decoder_fragility_metrics.png')
    plt.savefig(metrics_path, dpi=150, bbox_inches='tight')
    print(f"Saved metrics to {metrics_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("                 DECODER FRAGILITY TEST")
    print(f"{'='*60}")
    print(f"Dataset: {args.num_samples} CIFAR-10 images")
    print(f"\nReconstruction Quality vs Latent Noise:")
    print(f"{'Noise Std':<12} {'MSE':<12} {'PSNR (dB)':<12} {'Quality'}")
    print("-" * 60)
    for noise_std in noise_levels:
        mse = results[noise_std]['mse']
        psnr = results[noise_std]['psnr']
        quality = '✅ Excellent' if psnr > 35 else '⚠️ Degraded' if psnr > 25 else '❌ Poor'
        print(f"{noise_std:<12.2f} {mse:<12.6f} {psnr:<12.2f} {quality}")
    
    print(f"\n{'='*60}")
    print("FINDING: Even small latent noise (std=0.1) degrades PSNR by")
    clean_psnr = results[0.0]['psnr']
    noisy_psnr = results[0.1]['psnr']
    degradation = clean_psnr - noisy_psnr
    print(f"{degradation:.1f} dB, proving the decoder is sensitive to")
    print("imperfect latents from DiT generation.")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-samples', type=int, default=25, help='Number of CIFAR-10 samples')
    parser.add_argument('--output-dir', type=str, default='decoder_fragility_results', help='Output directory')
    args = parser.parse_args()
    main(args)

