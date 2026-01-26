# Point-MAE Reconstruction Quality Analysis

## Current Issue

The reconstruction visualization looks sparse/poor because:

1. **High Mask Ratio (60%)**: The model masks 60% of groups during training/inference
2. **Group-based Reconstruction**: Only reconstructs points within masked groups (not full point cloud)
3. **Sparse Output**: 64 groups × 32 points = 2048 total points (vs 8192 input)

## Understanding the Visualization

```
Original Input (8192 pts) → Visible (40% = 832 pts) → Reconstructed (2048 pts)
                                   ↓
                           60% groups MASKED
                                   ↓
                           Model predicts these
```

The "Reconstructed" view shows:
- Visible points (green) + Predicted masked points
- Total = 2048 points (64 groups × 32 points per group)

---

## TESTED: Options to Improve WITHOUT Retraining

### Option 1: Lower Mask Ratio at Inference (RECOMMENDED) ✓ TESTED

You can use a **lower mask ratio during visualization** without retraining.
The model was trained with 60% masking, but for visualization you can use less.

**Command:**
```bash
python tools/vis_pretrain.py \
    --ckpt experiments/pretrain_modelnet_normals/cfgs/pretrain_6channel/ckpt-last.pth \
    --config cfgs/pretrain_modelnet_normals.yaml \
    --output_dir vis_results/ \
    --mask_ratio 0.3  # Override: use 30% instead of 60%
```

**Test Results:**

| Mask Ratio | Visible Points | Reconstructed | Quality |
|------------|----------------|---------------|---------|
| 0.6 (default) | 832 (40%) | 2048 | Sparse, hard to see shape |
| 0.3 | 1440 (70%) | 2048 | Better, shape visible |
| 0.1 | 1856 (90%) | 2048 | Best, clear airplane shape |

**Recommendation**: Use `--mask_ratio 0.1` or `--mask_ratio 0.2` for best visualization.

### Option 2: Multiple Forward Passes with Different Masks

Run the model multiple times with different random masks and combine results:

```python
all_reconstructions = []
for _ in range(5):
    dense_points, vis_points, centers = model(points, vis=True)
    all_reconstructions.append(dense_points)
# Combine all reconstructions
combined = torch.cat(all_reconstructions, dim=1)
```

### Option 3: Use Full Point Cloud for Visualization

Instead of showing the sparse reconstruction, overlay:
- Original input (blue)
- Reconstructed masked regions only (red)

This gives better visual comparison.

---

## Options Requiring Retraining

### Option A: Lower Mask Ratio During Training

```yaml
# cfgs/pretrain_modelnet_normals.yaml
transformer_config:
  mask_ratio: 0.4  # Instead of 0.6
```

**Trade-off**: Lower mask ratio = easier task = potentially weaker representations
**Note**: The model learns better representations with higher mask ratios (harder task)

### Option B: More Groups / Larger Group Size (RECOMMENDED for retraining)

```yaml
model:
  group_size: 32
  num_group: 128    # Instead of 64
```

**Effect**: 128 groups × 32 points = 4096 points output (denser)

### Option C: Full Coverage Configuration

```yaml
model:
  group_size: 64
  num_group: 128
```

**Effect**: 128 × 64 = 8192 points (matches input size)

---

## Quick Reference: Visualization Commands

```bash
# Best visualization (10% masking)
python tools/vis_pretrain.py \
    --ckpt experiments/pretrain_modelnet_normals/cfgs/pretrain_6channel/ckpt-last.pth \
    --config cfgs/pretrain_modelnet_normals.yaml \
    --output_dir vis_results_best/ \
    --mask_ratio 0.1

# Moderate (30% masking)
python tools/vis_pretrain.py \
    --ckpt experiments/pretrain_modelnet_normals/cfgs/pretrain_6channel/ckpt-last.pth \
    --config cfgs/pretrain_modelnet_normals.yaml \
    --output_dir vis_results_moderate/ \
    --mask_ratio 0.3

# Default (60% masking - as trained)
python tools/vis_pretrain.py \
    --ckpt experiments/pretrain_modelnet_normals/cfgs/pretrain_6channel/ckpt-last.pth \
    --config cfgs/pretrain_modelnet_normals.yaml \
    --output_dir vis_results_default/
```

---

## For Retraining: Recommended Config

```yaml
# cfgs/pretrain_modelnet_normals_v2.yaml
model:
  NAME: Point_MAE
  input_channel: 6
  group_size: 32
  num_group: 128      # Doubled for denser output
  loss: cdl2
  transformer_config:
    mask_ratio: 0.6   # Keep high for better representations
    mask_type: 'rand'
    trans_dim: 384
    encoder_dims: 384
    depth: 12
    drop_path_rate: 0.1
    num_heads: 6
    decoder_depth: 4
    decoder_num_heads: 6
```

This would give: 128 groups × 32 points = 4096 points output
