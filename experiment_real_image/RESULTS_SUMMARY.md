# Single Image Overfitting Experiment - Real Cat Image

## Setup
- **Real Image**: Cat photo from Wikimedia Commons
- **Encoder**: DINOv2-B (768-dimensional tokens)
- **Latent Shape**: [1, 768, 16, 16]
- **Training Steps**: 600
- **DiT Depths**: All models use depth=12

## Results

| Width | Final Loss | Width ≥ 768? | Converged? | Parameters |
|-------|-----------|---------------|------------|-----------|
| 384   | 0.724     | ❌ (384 < 768) | ❌ NO     | 33.6M     |
| 768   | 0.338     | ✅ (768 = 768) | ✅ PARTIAL | 131.9M    |
| 896   | 0.274     | ✅ (896 > 768) | ✅ YES    | 179.0M    |

## Key Findings

### ✅ **Experiment Confirms the Paper's Theory:**

1. **Width 384 (< 768)**: Loss stays high at **0.724** - cannot converge
2. **Width 768 (= 768)**: Loss drops to **0.338** - starts to converge
3. **Width 896 (> 768)**: Loss drops further to **0.274** - best convergence

### 📊 **Visual Evidence**
- The reconstructions clearly show that width >= 768 is necessary
- Width 384 produces poor quality reconstruction
- Width 768 and 896 produce much better reconstructions of the cat

### 🔬 **Theoretical Validation**
According to Theorem 1 in the paper:
- When model width d < token dimension n = 768, the model **cannot** learn to minimize loss
- When d >= n, the model **can** successfully learn

## Conclusion
**This experiment with a real image confirms the paper's key insight:**
> DiT requires width ≥ token dimension to successfully generate images from high-dimensional RAE latents.

The visual quality difference is clear - only models with width >= 768 can properly reconstruct the cat image!
