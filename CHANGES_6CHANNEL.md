# Point-MAE 6-Channel (XYZ + Normals) Modifications

This document tracks all changes made to support 6-channel input (xyz + normals) instead of the original 3-channel (xyz only).

## Summary of Changes

### 1. Model Architecture (`models/Point_MAE.py`)

#### Encoder Class (Line 16-48)
- **Added `input_channel` parameter** (default=6)
- Changed first convolution from `nn.Conv1d(3, 128, 1)` to `nn.Conv1d(input_channel, 128, 1)`
- Updated forward pass to handle variable channel count

```python
# Before
def __init__(self, encoder_channel):
    self.first_conv = nn.Sequential(
        nn.Conv1d(3, 128, 1),  # Fixed 3 channels
        ...
    )

# After
def __init__(self, encoder_channel, input_channel=6):
    self.input_channel = input_channel
    self.first_conv = nn.Sequential(
        nn.Conv1d(input_channel, 128, 1),  # Configurable channels
        ...
    )
```

#### Group Class (Line 51-83)
- Modified to handle 6-channel data
- FPS and KNN still use only xyz (first 3 channels) for spatial operations
- Full 6-channel data is gathered for neighborhoods
- Only xyz coordinates are normalized (center subtracted)

```python
# Key changes:
xyz = pts[:, :, :3]  # Extract xyz for FPS and KNN
neighborhood[:, :, :, :3] = neighborhood[:, :, :, :3] - center.unsqueeze(2)  # Normalize xyz only
```

#### MaskTransformer Class (Line 207-330)
- Added `input_channel` config parameter
- Passes `input_channel` to Encoder

#### Point_MAE Class (Line 333-427)
- Added `input_channel` config parameter (default=6)
- Updated prediction head to output `input_channel * group_size` instead of `3 * group_size`
- Chamfer distance loss computed on xyz only (first 3 channels)

```python
# Prediction head now outputs 6 channels per point
nn.Conv1d(self.trans_dim, self.input_channel*self.group_size, 1)

# Loss computed on xyz only
loss1 = self.loss_func(rebuild_points[:, :, :3], gt_points[:, :, :3])
```

#### PointTransformer Class (Line 430-560)
- Added `input_channel` config parameter
- Passes `input_channel` to Encoder

---

### 2. Dataset Configuration (`cfgs/dataset_configs/ModelNet40.yaml`)

```yaml
# Before
NAME: ModelNet
DATA_PATH: data/ModelNet/modelnet40_normal_resampled
N_POINTS: 8192
NUM_CATEGORY: 40
USE_NORMALS: FALSE

# After
NAME: ModelNet
DATA_PATH: /data/joshi/modelnet40_pointmae
N_POINTS: 8192
NUM_CATEGORY: 40
USE_NORMALS: True
```

---

### 3. New Configuration Files

#### `cfgs/pretrain_modelnet_normals.yaml`
- Pretraining config for 6-channel data
- `input_channel: 6` in model config
- Uses ModelNet40 dataset with normals

#### `cfgs/finetune_modelnet_normals.yaml`
- Finetuning config for 6-channel data
- `input_channel: 6` in model config
- 40 classes for ModelNet40

---

### 4. New Tools

#### `tools/run_conversion.py`
PLY to Point-MAE format conversion utility:
- Converts PLY mesh files to txt format (x,y,z,nx,ny,nz)
- Samples 8192 points from mesh surface with normals
- Creates train/test splits
- Generates .dat cache files
- Uses multiprocessing (255 cores)

**Usage:**
```bash
python tools/run_conversion.py \
    --input /data/joshi/modelnet40_meshes \
    --output /data/joshi/modelnet40_pointmae \
    --n_points 8192 \
    --n_workers 255
```

---

## Training Commands

### Step 1: Convert PLY to Point-MAE Format
```bash
cd /home/joshi/experiments/Point-MAE
python tools/run_conversion.py \
    --input /data/joshi/modelnet40_meshes \
    --output /data/joshi/modelnet40_pointmae \
    --n_points 8192 \
    --n_workers 255
```

### Step 2: Pretrain with 8 GPUs
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node=8 \
    main.py \
    --config cfgs/pretrain_modelnet_normals.yaml \
    --launcher pytorch \
    --exp_name pretrain_6channel
```

### Step 3: Finetune
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node=8 \
    main.py \
    --config cfgs/finetune_modelnet_normals.yaml \
    --finetune_model \
    --ckpts experiments/pretrain_modelnet_normals/pretrain/ckpt-last.pth \
    --launcher pytorch \
    --exp_name finetune_6channel
```

---

## Data Format

### Input Format (txt files)
Each line contains 6 comma-separated values:
```
x,y,z,nx,ny,nz
```

Example:
```
0.123456,0.234567,0.345678,0.577350,0.577350,0.577350
...
```

### Cache Format (.dat files)
Pickle files containing:
- `list_of_points`: List of (N, 6) numpy arrays
- `list_of_labels`: List of (1,) numpy arrays with class indices

---

## Backward Compatibility

To use the original 3-channel mode, set `input_channel: 3` in the model config:

```yaml
model : {
  NAME: Point_MAE,
  input_channel: 3,  # Use xyz only
  ...
}
```

And set `USE_NORMALS: False` in the dataset config.

---

## Files Modified

1. `models/Point_MAE.py` - Core model architecture
2. `cfgs/dataset_configs/ModelNet40.yaml` - Dataset config
3. `cfgs/pretrain_modelnet_normals.yaml` - New pretrain config (created)
4. `cfgs/finetune_modelnet_normals.yaml` - New finetune config (created)
5. `tools/run_conversion.py` - PLY conversion utility (created)
6. `CHANGES_6CHANNEL.md` - This documentation (created)
