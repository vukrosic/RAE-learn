#!/usr/bin/env python3
"""
Noise-Robust Decoder Experiment (Memory Efficient)

Instead of training full decoders from scratch, we:
1. Take a pretrained decoder
2. Freeze all layers except the final projection
3. Fine-tune TWO versions on CIFAR-10:
   - Version A: WITHOUT noise augmentation (noise_tau=0)
   - Version B: WITH noise augmentation (noise_tau=0.5)
4. Test both on noisy latents to show B is more robust

This approach avoids OOM while demonstrating the key concept.
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
import copy

def load_cifar10_subset(num_samples=200, batch_size=8):
    """Load small CIFAR-10 subset."""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
    ])
    
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=False, transform=transform
    )
    
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    subset = Subset(dataset, indices)
    
    return DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2)

def freeze_decoder_except_final(decoder):
    """Freeze all decoder layers except the final output projection."""
    # Freeze everything first
    for param in decoder.parameters():
        param.requires_grad = False
    
    # Unfreeze the final layer(s)
    # Most ViT decoders have a final 'head' or 'out' layer
    if hasattr(decoder, 'out_proj'):
        for param in decoder.out_proj.parameters():
            param.requires_grad = True
    elif hasattr(decoder, 'head'):
        for param in decoder.head.parameters():
            param.requires_grad = True
    elif hasattr(decoder, 'decoder_pred'):
        for param in decoder.decoder_pred.parameters():
            param.requires_grad = True
    
    trainable_params = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in decoder.parameters())
    
    print(f"Trainable params: {trainable_params:,} / {total_params:,}")
    return decoder

def fine_tune_decoder(rae, dataloader, noise_tau, num_epochs, device):
    """Fine-tune decoder with or without noise augmentation."""
    # Clone the RAE to avoid modifying the original
    rae_copy = copy.deepcopy(rae)
    rae_copy.noise_tau = noise_tau
    
    # Freeze encoder and unfreeze decoder final layer
    for param in rae_copy.encoder.parameters():
        param.requires_grad = False
    
    rae_copy.decoder = freeze_decoder_except_final(rae_copy.decoder)
    rae_copy.to(device)
    rae_copy.train()
    
    # Only optimize decoder parameters
    optimizer = AdamW(
        [p for p in rae_copy.decoder.parameters() if p.requires_grad],
        lr=1e-4
    )
    
    criterion = nn.MSELoss()
    losses = []
    
    pbar = tqdm(range(num_epochs), desc=f"Fine-tuning (tau={noise_tau})")
    for epoch in pbar:
        epoch_loss = 0
        for batch_idx, (images, _) in enumerate(dataloader):
            images = images.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass through RAE (with noise augmentation if tau>0)
            with torch.no_grad():
                latents = rae_copy.encode(images)
            
            # Apply noise augmentation to latents if tau > 0
            if noise_tau > 0:
                noise = torch.randn_like(latents) * noise_tau
                latents_noisy = latents + noise
            else:
                latents_noisy = latents
            
            # Decode
            reconstructed = rae_copy.decode(latents_noisy)
            
            # Loss
            loss = criterion(reconstructed, images)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    rae_copy.eval()
    return rae_copy, losses

def test_decoder_robustness(rae, test_images, noise_levels, device):
    """Test decoder on latents with varying noise levels."""
    rae.eval()
    results = {}
    
    with torch.no_grad():
        # Encode once
        latents = rae.encode(test_images.to(device))
        
        for noise_sigma in noise_levels:
            # Add noise to latents
            if noise_sigma > 0:
                noisy_latents = latents + torch.randn_like(latents) * noise_sigma
            else:
                noisy_latents = latents
            
            # Decode
            reconstructed = rae.decode(noisy_latents)
            
            # Calculate PSNR
            mse = ((reconstructed - test_images.to(device)) ** 2).mean()
            psnr = 10 * torch.log10(1.0 / mse)
            
            results[noise_sigma] = {
                'psnr': psnr.item(),
                'reconstructed': reconstructed.cpu()
            }
    
    return results

def visualize_results(test_images, results_no_noise, results_with_noise, noise_levels, output_dir):
    """Create comparison visualization."""
    num_images = min(3, len(test_images))
    num_noise = len(noise_levels)
    
    fig, axes = plt.subplots(num_images, 1 + 2*num_noise, figsize=(4*(1+2*num_noise), 4*num_images))
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(num_images):
        # Original
        axes[i, 0].imshow(test_images[i].permute(1, 2, 0).cpu().numpy())
        axes[i, 0].set_title('Original' if i == 0 else '')
        axes[i, 0].axis('off')
        
        # For each noise level
        for j, sigma in enumerate(noise_levels):
            # Without noise training
            col_idx = 1 + 2*j
            img_no_noise = results_no_noise[sigma]['reconstructed'][i]
            axes[i, col_idx].imshow(img_no_noise.permute(1, 2, 0).numpy().clip(0, 1))
            if i == 0:
                psnr = results_no_noise[sigma]['psnr']
                axes[i, col_idx].set_title(f'No Aug\n(σ={sigma}, PSNR={psnr:.1f})')
            axes[i, col_idx].axis('off')
            
            # With noise training
            col_idx = 1 + 2*j + 1
            img_with_noise = results_with_noise[sigma]['reconstructed'][i]
            axes[i, col_idx].imshow(img_with_noise.permute(1, 2, 0).numpy().clip(0, 1))
            if i == 0:
                psnr = results_with_noise[sigma]['psnr']
                axes[i, col_idx].set_title(f'With Aug\n(σ={sigma}, PSNR={psnr:.1f})')
            axes[i, col_idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'robustness_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {output_dir}/robustness_comparison.png")

def plot_psnr_comparison(results_no_noise, results_with_noise, noise_levels, output_dir):
    """Plot PSNR vs noise level."""
    psnr_no_noise = [results_no_noise[s]['psnr'] for s in noise_levels]
    psnr_with_noise = [results_with_noise[s]['psnr'] for s in noise_levels]
    
    plt.figure(figsize=(10, 6))
    plt.plot(noise_levels, psnr_no_noise, 'o-', label='Decoder WITHOUT Noise Aug (tau=0)', linewidth=2, markersize=8)
    plt.plot(noise_levels, psnr_with_noise, 's-', label='Decoder WITH Noise Aug (tau=0.5)', linewidth=2, markersize=8)
    plt.xlabel('Latent Noise Level (σ)', fontsize=12)
    plt.ylabel('PSNR (dB)', fontsize=12)
    plt.title('Decoder Robustness to Noisy Latents', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'psnr_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved PSNR comparison to {output_dir}/psnr_comparison.png")

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load RAE with pretrained decoder
    print("Loading base RAE...")
    rae_config = OmegaConf.create({
        'target': 'stage1.RAE',
        'params': {
            'encoder_cls': 'Dinov2withNorm',
            'encoder_config_path': 'facebook/dinov2-with-registers-base',
            'encoder_input_size': 224,
            'encoder_params': {'dinov2_path': 'facebook/dinov2-with-registers-base', 'normalize': True},
            'decoder_config_path': 'configs/decoder/ViTXL',
            'pretrained_decoder_path': 'models/decoders/dinov2/wReg_base/ViTXL_n08/model.pt',
            'noise_tau': 0.0,
            'reshape_to_2d': True,
            'normalization_stat_path': 'models/stats/dinov2/wReg_base/imagenet1k/stat.pt',
        }
    })
    rae_base = RAE(**rae_config.params).to(device)
    rae_base.eval()
    
    # Load CIFAR-10 data
    print(f"Loading {args.num_samples} CIFAR-10 images...")
    train_loader = load_cifar10_subset(args.num_samples, args.batch_size)
    
    # Get test images
    test_images = next(iter(train_loader))[0][:args.num_test_images]
    
    print("\n" + "="*70)
    print("EXPERIMENT A: Fine-tune decoder WITHOUT noise augmentation (tau=0)")
    print("="*70)
    rae_no_noise, losses_no_noise = fine_tune_decoder(
        rae_base, train_loader, noise_tau=0.0, 
        num_epochs=args.num_epochs, device=device
    )
    
    print("\n" + "="*70)
    print("EXPERIMENT B: Fine-tune decoder WITH noise augmentation (tau=0.5)")
    print("="*70)
    rae_with_noise, losses_with_noise = fine_tune_decoder(
        rae_base, train_loader, noise_tau=0.5,
        num_epochs=args.num_epochs, device=device
    )
    
    # Test robustness
    print("\n" + "="*70)
    print("Testing decoder robustness on noisy latents...")
    print("="*70)
    
    noise_levels = [0.0, 0.3, 0.6, 1.0]
    
    results_no_noise = test_decoder_robustness(rae_no_noise, test_images, noise_levels, device)
    results_with_noise = test_decoder_robustness(rae_with_noise, test_images, noise_levels, device)
    
    # Print results
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print("\n| Latent Noise (σ) | No Aug PSNR | With Aug PSNR | Improvement |")
    print("|-----------------|-------------|---------------|-------------|")
    
    for sigma in noise_levels:
        psnr_no = results_no_noise[sigma]['psnr']
        psnr_with = results_with_noise[sigma]['psnr']
        improvement = psnr_with - psnr_no
        symbol = "✅" if improvement > 0.5 else ("⚠️" if improvement > 0 else "❌")
        print(f"| {sigma:<15.1f} | {psnr_no:>11.2f} | {psnr_with:>13.2f} | {improvement:>+10.2f} {symbol} |")
    
    # Visualizations
    print("\nGenerating visualizations...")
    visualize_results(test_images, results_no_noise, results_with_noise, noise_levels, args.output_dir)
    plot_psnr_comparison(results_no_noise, results_with_noise, noise_levels, args.output_dir)
    
    print(f"\n✅ Experiment complete! Results saved to {args.output_dir}/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-samples', type=int, default=200, help='Number of CIFAR-10 samples for fine-tuning')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--num-epochs', type=int, default=10, help='Number of fine-tuning epochs')
    parser.add_argument('--num-test-images', type=int, default=3, help='Number of test images to visualize')
    parser.add_argument('--output-dir', type=str, default='noise_robust_results', help='Output directory')
    args = parser.parse_args()
    main(args)

