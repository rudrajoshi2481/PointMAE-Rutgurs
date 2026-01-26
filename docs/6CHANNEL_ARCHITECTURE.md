# Point-MAE 6-Channel Architecture Documentation

## Overview

This document describes the modifications made to Point-MAE to support 6-channel input (XYZ + Normals) for pretraining on ModelNet40 mesh data.

## Architecture Flowchart

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         INPUT: Point Cloud (B, N, 6)                        │
│                         [x, y, z, nx, ny, nz] per point                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FPS (Farthest Point Sampling)                     │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • Uses ONLY xyz (first 3 channels) for distance computation        │    │
│  │  • Gathers ALL 6 channels for selected points                       │    │
│  │  • Output: (B, npoints, 6)                                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  File: utils/misc.py:fps()                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GROUP DIVIDER                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  1. FPS to select 64 group centers (using xyz only)                 │    │
│  │  2. KNN to find 32 nearest neighbors per center (using xyz only)    │    │
│  │  3. Gather all 6 channels for neighborhoods                         │    │
│  │  4. Normalize xyz relative to center, keep normals unchanged        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  Output: neighborhood (B, 64, 32, 6), center (B, 64, 3)                     │
│  File: models/Point_MAE.py:Group                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                               ENCODER                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • Input: (B, G, S, 6) → 6 channels per point                       │    │
│  │  • Conv1D layers: 6 → 128 → 256 → encoder_dims (384)                │    │
│  │  • Max pooling over points in each group                            │    │
│  │  • Output: (B, G, 384) token embeddings                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  File: models/Point_MAE.py:Encoder                                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            MASKING (60%)                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • Randomly mask 60% of groups (38 out of 64)                       │    │
│  │  • Keep 40% visible (26 groups)                                     │    │
│  │  • Visible tokens → Transformer Encoder                             │    │
│  │  • Masked tokens → Replaced with learnable mask token               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  File: models/Point_MAE.py:MaskTransformer                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
    ┌───────────────────────────┐       ┌───────────────────────────┐
    │   VISIBLE TOKENS (26)     │       │   MASKED TOKENS (38)      │
    │   → Transformer Encoder   │       │   → Mask Token + Pos Emb  │
    │   (12 layers, 6 heads)    │       │                           │
    └───────────────────────────┘       └───────────────────────────┘
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TRANSFORMER DECODER                                │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • Concatenate visible + masked tokens                              │    │
│  │  • 4 decoder layers, 6 heads                                        │    │
│  │  • Output: reconstructed token features (B, 38, 384)                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  File: models/Point_MAE.py:TransformerDecoder                               │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PREDICTION HEAD                                    │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • Conv1D: 384 → 3 * 32 = 96 (xyz only, NOT 6 channels!)            │    │
│  │  • Reshape to (B*38, 32, 3)                                         │    │
│  │  • Output: reconstructed xyz coordinates                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  File: models/Point_MAE.py:Point_MAE.increase_dim                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          CHAMFER DISTANCE LOSS                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  • Compare reconstructed xyz with ground truth xyz                  │    │
│  │  • L2 Chamfer Distance (bidirectional)                              │    │
│  │  • Normals are NOT reconstructed, only used as input features       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│  File: extensions/chamfer_dist/__init__.py:ChamferDistanceL2                │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Key Insight: Why Normals Help

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     HOW NORMALS CAPTURE INFORMATION                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  XYZ Only (3 channels):          XYZ + Normals (6 channels):                │
│  ┌─────────────────────┐         ┌─────────────────────────────────────┐    │
│  │  • Position only    │         │  • Position + Surface orientation   │    │
│  │  • No surface info  │         │  • Distinguishes flat vs curved     │    │
│  │  • Ambiguous edges  │         │  • Sharp edges have varying normals │    │
│  └─────────────────────┘         │  • Smooth surfaces have consistent  │    │
│                                  │    normals                           │    │
│                                  └─────────────────────────────────────┘    │
│                                                                             │
│  Example: A cube corner vs a sphere point                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  XYZ: Both might have similar local point distributions             │    │
│  │  Normals: Cube corner has 3 distinct normal directions              │    │
│  │           Sphere point has smoothly varying normals                 │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  The encoder learns to use normal information to create richer             │
│  embeddings that capture both geometry AND surface properties.             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Files Modified

### 1. `models/Point_MAE.py`

| Component | Change | Lines |
|-----------|--------|-------|
| `Encoder` | Accept `input_channel` parameter (default 6) | 17-48 |
| `Group` | Use xyz for FPS/KNN, gather all 6 channels | 58-83 |
| `MaskTransformer` | Pass `input_channel` to Encoder | 219-221 |
| `Point_MAE` | Prediction head outputs 3 channels (xyz only) | 366-373 |
| `Point_MAE.forward` | Reconstruct xyz only, compare with xyz ground truth | 407-414 |

### 2. `utils/misc.py`

| Function | Change | Lines |
|----------|--------|-------|
| `fps()` | Use xyz for FPS distance, gather all channels | 22-33 |

### 3. `datasets/data_transforms.py`

| Transform | Change | Lines |
|-----------|--------|-------|
| `PointcloudRotate` | Also rotate normals when present | 6-23 |

### 4. `cfgs/pretrain_modelnet_normals.yaml`

New config file with:
- `input_channel: 6`
- `lr: 0.0001` (reduced for stability)
- `initial_epochs: 20` (extended warmup)
- `grad_norm_clip: 10`

### 5. `cfgs/dataset_configs/ModelNet40.yaml`

- `USE_NORMALS: True`
- `DATA_PATH: /data/joshi/modelnet40_pointmae`

## Training Progress

```
Epoch 0:  Loss = 613.08  (actual: 0.613)
Epoch 10: Loss = 135.21  (actual: 0.135)
Epoch 35: Loss = 130.92  (actual: 0.131)
Epoch 49: Loss = 130.40  (actual: 0.130)
```

Loss stabilized around 130 (0.13 actual), indicating successful training.

## Critical Fix: Loss Explosion Bug

### Problem
The original 6-channel implementation had the prediction head output 6 channels:
```python
nn.Conv1d(self.trans_dim, self.input_channel * self.group_size, 1)  # 6 * 32 = 192
```

But the Chamfer loss only supervised 3 channels (xyz), leaving 3 channels (normals) unsupervised. This caused gradient instability and loss explosion.

### Solution
Changed prediction head to always output 3 channels:
```python
nn.Conv1d(self.trans_dim, 3 * self.group_size, 1)  # 3 * 32 = 96
```

Normals are used as **input features only**, not as reconstruction targets.

## Embedding Extraction

See `tools/extract_embeddings.py` for a script to extract embeddings from the pretrained model.

### Embedding Locations

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      EMBEDDING EXTRACTION POINTS                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. ENCODER OUTPUT (after Group + Encoder, before Transformer)             │
│     Shape: (B, 64, 384)                                                     │
│     Use: Local patch features                                               │
│                                                                             │
│  2. TRANSFORMER ENCODER OUTPUT (after all 12 layers)                        │
│     Shape: (B, 26, 384) for visible tokens                                  │
│     Use: Contextualized patch features                                      │
│                                                                             │
│  3. CLS TOKEN (if using classification head)                                │
│     Shape: (B, 384)                                                         │
│     Use: Global shape representation                                        │
│                                                                             │
│  4. POOLED FEATURES (max/mean pool over all tokens)                         │
│     Shape: (B, 384)                                                         │
│     Use: Global shape representation for downstream tasks                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```
