#!/usr/bin/env python3
import os
import torch
from torch.optim import AdamW
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import argparse
from stage1 import RAE
from stage2.models.lightningDiT import LightningDiT
from stage2.transport import create_transport, Sampler
from omegaconf import OmegaConf

def load_single_image(image_path, size=256):
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])
    img = Image.open(image_path).convert('RGB')
    return transform(img).unsqueeze(0)

def create_dit_model(width, depth=12, token_dim=768, num_heads=None):
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
        num_classes=1000,
        use_qknorm=False,
        use_swiglu=True,
        use_rope=True,
        use_rmsnorm=True,
        wo_shift=False,
        use_gembed=True,
    )
    return model

def train_single_image(model, latent, device, num_steps=1200, lr=2e-4):
    model = model.to(device)
    latent = latent.to(device)
    
    optimizer = AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95))
    transport = create_transport(
        path_type='Linear',
        prediction='velocity',
        loss_weight=None,
        time_dist_shift=1.0,
    )
    
    losses = []
    model.train()
    y = torch.tensor([0], device=device)
    
    pbar = tqdm(range(num_steps), desc="Training")
    for step in pbar:
        optimizer.zero_grad()
        model_kwargs = dict(y=y)
        loss_dict = transport.training_losses(model, latent, model_kwargs)
        loss = loss_dict["loss"].mean()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if step % 100 == 0:
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    return losses

def sample_from_model(model, latent_shape, device, y=None):
    model.eval()
    with torch.no_grad():
        transport = create_transport(
            path_type='Linear',
            prediction='velocity',
            loss_weight=None,
            time_dist_shift=1.0,
        )
        sampler = Sampler(transport)
        ode_sampler = sampler.sample_ode(
            sampling_method='euler',
            num_steps=25,
        )
        z = torch.randn(1, *latent_shape, device=device)
        if y is None:
            y = torch.tensor([0], device=device)
        model_kwargs = dict(y=y)
        samples = ode_sampler(z, model.forward, **model_kwargs)
        return samples[-1]

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
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
    
    print(f"Loading image: {args.image_path}")
    img = load_single_image(args.image_path, size=256).to(device)
    with torch.no_grad():
        latent = rae.encode(img)
    
    token_dim = latent.shape[1]
    print(f"Latent shape: {latent.shape} (token_dim={token_dim})")
    
    widths = args.widths
    results = {}
    
    for width in widths:
        print(f"\n{'='*60}")
        print(f"Testing width={width} (token_dim={token_dim})")
        print(f"{'='*60}")
        
        model = create_dit_model(
            width=width,
            depth=args.depth,
            token_dim=token_dim,
        )
        
        param_count = sum(p.numel() for p in model.parameters()) / 1e6
        print(f"Model parameters: {param_count:.2f}M")
        
        losses = train_single_image(
            model, latent, device,
            num_steps=args.num_steps,
            lr=args.lr
        )
        
        final_loss = losses[-1]
        print(f"Final loss: {final_loss:.6f}")
        
        print("Generating sample...")
        sample_latent = sample_from_model(
            model, latent.shape[1:], device
        )
        
        with torch.no_grad():
            reconstructed = rae.decode(sample_latent)
        
        results[width] = {
            'losses': losses,
            'final_loss': final_loss,
            'reconstructed': reconstructed.cpu(),
            'param_count': param_count
        }
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    for width in widths:
        losses = results[width]['losses']
        plt.plot(losses, label=f'width={width}')
    
    theoretical_lb = (token_dim - min(widths)) / token_dim if min(widths) < token_dim else 0
    plt.axhline(y=theoretical_lb, color='r', linestyle='--', label=f'Theoretical lower bound')
    plt.xlabel('Training Steps')
    plt.ylabel('Loss')
    plt.title(f'Single Image Overfitting: DiT Width vs Token Dimension ({token_dim})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{args.output_dir}/loss_curves.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved loss curves to {args.output_dir}/loss_curves.png")
    
    n_widths = len(widths)
    fig, axes = plt.subplots(2, n_widths + 1, figsize=(4*(n_widths+1), 8))
    
    original_np = img[0].cpu().permute(1, 2, 0).numpy()
    axes[0, 0].imshow(original_np)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    axes[1, 0].axis('off')
    
    for i, width in enumerate(widths):
        recon = results[width]['reconstructed'][0].permute(1, 2, 0).numpy()
        recon = np.clip(recon, 0, 1)
        
        axes[0, i+1].imshow(recon)
        axes[0, i+1].set_title(f'width={width}')
        axes[0, i+1].axis('off')
        
        final_loss = results[width]['final_loss']
        status = "OK" if width >= token_dim else "FAIL"
        text = f'{status} Loss: {final_loss:.4f}\n{results[width]["param_count"]:.1f}M params'
        axes[1, i+1].text(0.5, 0.5, text, ha='center', va='center', fontsize=12)
        axes[1, i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{args.output_dir}/reconstructions.png', dpi=150, bbox_inches='tight')
    print(f"Saved reconstructions to {args.output_dir}/reconstructions.png")
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Token dimension: {token_dim}")
    print(f"\nResults:")
    for width in widths:
        final_loss = results[width]['final_loss']
        converged = "YES" if final_loss < 0.1 else "NO"
        print(f"  Width {width:4d}: Loss={final_loss:.6f}, Converged={converged}, Params={results[width]['param_count']:.1f}M")
    
    print(f"\nTheory: DiT should only converge when width >= token_dim ({token_dim})")
    print(f"{'='*60}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Single image overfitting experiment")
    parser.add_argument("--image-path", type=str, default="test_data/train/class0/img_0.jpg")
    parser.add_argument("--widths", type=int, nargs='+', default=[384, 512, 640, 768, 896])
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num-steps", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output-dir", type=str, default="experiment_results")
    args = parser.parse_args()
    main(args)
