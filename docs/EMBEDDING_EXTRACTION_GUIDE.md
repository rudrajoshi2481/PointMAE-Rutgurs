# Embedding Extraction Guide

## Quick Start

```bash
# After training completes, extract embeddings:
cd /home/joshi/experiments/Point-MAE

# Activate environment
source /home/joshi/experiments/.pointMAEenv/bin/activate

# Extract test set embeddings
python tools/extract_embeddings.py \
    --ckpt experiments/pretrain_modelnet_normals/cfgs/pretrain_6channel/ckpt-last.pth \
    --config cfgs/pretrain_modelnet_normals.yaml \
    --data_path /data/joshi/modelnet40_pointmae \
    --output_dir embeddings/ \
    --split test

# Extract train set embeddings
python tools/extract_embeddings.py \
    --ckpt experiments/pretrain_modelnet_normals/cfgs/pretrain_6channel/ckpt-last.pth \
    --config cfgs/pretrain_modelnet_normals.yaml \
    --data_path /data/joshi/modelnet40_pointmae \
    --output_dir embeddings/ \
    --split train
```

## Output Format

The script saves embeddings as a `.npz` file with the following arrays:

| Array | Shape | Description |
|-------|-------|-------------|
| `encoder_features` | (N, 64, 384) | Local patch features before transformer |
| `transformer_features` | (N, 26, 384) | Contextualized features (visible tokens only) |
| `global_max` | (N, 384) | Max-pooled global features |
| `global_mean` | (N, 384) | Mean-pooled global features |
| `centers` | (N, 64, 3) | Group center coordinates |
| `labels` | (N,) | Class labels |

## Loading Embeddings

```python
import numpy as np

# Load embeddings
data = np.load('embeddings/embeddings_test.npz')

# Access different embedding types
encoder_features = data['encoder_features']      # (640, 64, 384)
transformer_features = data['transformer_features']  # (640, 26, 384)
global_max = data['global_max']                  # (640, 384)
global_mean = data['global_mean']                # (640, 384)
labels = data['labels']                          # (640,)

print(f"Number of samples: {len(labels)}")
print(f"Global feature dimension: {global_max.shape[1]}")
```

## Embedding Types Explained

### 1. Encoder Features (Local Patch Features)

```
Shape: (B, 64, 384)
       └─ 64 groups, 384-dim features per group
```

These are features extracted by the encoder **before** the transformer. Each group of 32 points is encoded into a 384-dimensional vector. These capture **local geometry** within each patch.

**Use cases:**
- Part segmentation
- Local feature matching
- Point-wise tasks

### 2. Transformer Features (Contextualized Features)

```
Shape: (B, 26, 384)
       └─ 26 visible tokens (40% of 64 groups)
```

These are features **after** the transformer encoder. They capture **global context** through self-attention. Only visible tokens (non-masked) are returned.

**Use cases:**
- When you need features that understand global shape context
- Scene understanding

### 3. Global Features (Max/Mean Pooled)

```
Shape: (B, 384)
       └─ Single 384-dim vector per shape
```

These are the most commonly used for downstream tasks. They aggregate all token features into a single vector representing the entire shape.

**Use cases:**
- Shape classification
- Shape retrieval
- Clustering

## Example: Shape Classification with SVM

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Load embeddings
train_data = np.load('embeddings/embeddings_train.npz')
test_data = np.load('embeddings/embeddings_test.npz')

# Use global max features
X_train = train_data['global_max']
y_train = train_data['labels']
X_test = test_data['global_max']
y_test = test_data['labels']

# Train SVM
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Classification accuracy: {acc:.4f}")
```

## Example: t-SNE Visualization

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load embeddings
data = np.load('embeddings/embeddings_test.npz')
features = data['global_max']
labels = data['labels']

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
features_2d = tsne.fit_transform(features)

# Plot
plt.figure(figsize=(12, 10))
scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                      c=labels, cmap='tab20', alpha=0.7, s=10)
plt.colorbar(scatter)
plt.title('t-SNE of Point-MAE Embeddings (6-channel)')
plt.savefig('tsne_embeddings.png', dpi=150)
plt.show()
```

## Programmatic Extraction (Custom Usage)

```python
import torch
import sys
sys.path.insert(0, '/home/joshi/experiments/Point-MAE')

from tools.extract_embeddings import load_model, EmbeddingExtractor

# Load model
model, config = load_model(
    'experiments/pretrain_modelnet_normals/cfgs/pretrain_6channel/ckpt-last.pth',
    'cfgs/pretrain_modelnet_normals.yaml',
    device='cuda'
)
extractor = EmbeddingExtractor(model).cuda()
extractor.eval()

# Your point cloud (B, N, 6) - xyz + normals
points = torch.randn(4, 8192, 6).cuda()

with torch.no_grad():
    # Option 1: Get all features at once
    features = extractor.extract_all_features(points)
    print(f"Encoder features: {features['encoder_features'].shape}")
    print(f"Transformer features: {features['transformer_features'].shape}")
    print(f"Global max: {features['global_max'].shape}")
    
    # Option 2: Get only global features
    global_feat = extractor.extract_global_features(points, pooling='max')
    print(f"Global features: {global_feat.shape}")
```

## Are Normals Capturing Information?

To verify that normals are contributing to the learned representations:

```python
import torch
import numpy as np
sys.path.insert(0, '/home/joshi/experiments/Point-MAE')

from tools.extract_embeddings import load_model, EmbeddingExtractor

model, config = load_model(
    'experiments/pretrain_modelnet_normals/cfgs/pretrain_6channel/ckpt-last.pth',
    'cfgs/pretrain_modelnet_normals.yaml'
)
extractor = EmbeddingExtractor(model).cuda()
extractor.eval()

# Load a sample
import pickle
with open('/data/joshi/modelnet40_pointmae/modelnet40_test_8192pts_fps.dat', 'rb') as f:
    data = pickle.load(f)

points = torch.from_numpy(data[0][0]).unsqueeze(0).cuda().float()

with torch.no_grad():
    # Extract with original normals
    feat_original = extractor.extract_global_features(points)
    
    # Extract with zeroed normals
    points_no_normals = points.clone()
    points_no_normals[:, :, 3:] = 0
    feat_no_normals = extractor.extract_global_features(points_no_normals)
    
    # Extract with random normals
    points_random_normals = points.clone()
    points_random_normals[:, :, 3:] = torch.randn_like(points[:, :, 3:])
    feat_random = extractor.extract_global_features(points_random_normals)

# Compare features
diff_zero = torch.norm(feat_original - feat_no_normals).item()
diff_random = torch.norm(feat_original - feat_random).item()

print(f"Feature difference (original vs zero normals): {diff_zero:.4f}")
print(f"Feature difference (original vs random normals): {diff_random:.4f}")

# If normals are being used, these differences should be significant
if diff_zero > 1.0:
    print("✓ Normals ARE contributing to the embeddings!")
else:
    print("✗ Normals may not be contributing significantly")
```

## Checkpoint Locations

| Checkpoint | Path |
|------------|------|
| Latest | `experiments/pretrain_modelnet_normals/cfgs/pretrain_6channel/ckpt-last.pth` |
| Best (if saved) | `experiments/pretrain_modelnet_normals/cfgs/pretrain_6channel/ckpt-best.pth` |
