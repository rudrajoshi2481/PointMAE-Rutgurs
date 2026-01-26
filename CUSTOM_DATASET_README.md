# Point-MAE Custom Dataset Training Guide

This guide explains how to finetune Point-MAE on your custom PLY mesh dataset for 3D object classification.

## Overview

The pipeline consists of three main steps:
1. **Preprocessing**: Convert PLY meshes to Point-MAE format
2. **Finetuning**: Train the classifier using pretrained Point-MAE weights
3. **Classification**: Use the trained model to classify new objects

**Important**: Finetuning and classification are separate steps. You first train the model (finetuning), then use the trained model to classify new objects.

---

## Prerequisites

### 1. Install Dependencies

```bash
pip install -r requirements.txt

# Install CUDA extensions
cd extensions/chamfer_dist && python setup.py install --user && cd ../..
cd extensions/emd && python setup.py install --user && cd ../..

# Install PointNet++ ops
pip install "git+https://github.com/erikwijmans/Pointnet2_PyTorch.git#egg=pointnet2_ops&subdirectory=pointnet2_ops_lib"

# Install KNN CUDA
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl

# Install trimesh for PLY processing
pip install trimesh
```

### 2. Data Requirements

Your PLY files should be organized in a single directory:
```
/path/to/meshes/
├── Table_NW.ply
├── Table_NE.ply
├── Monitor_Cart.ply
├── Book_NE.ply
└── ...
```

The script automatically extracts class names from filenames:
- `Table_NW.ply` → class: `table`
- `Monitor_Cart.ply` → class: `monitor`
- `Book2_NE.ply` → class: `book`

---

## Step 1: Preprocess Your Data

Convert PLY meshes to Point-MAE format:

```bash
python tools/preprocess_custom_dataset.py \
    --input /path/to/your/meshes \
    --output /path/to/processed/data \
    --n_points 8192 \
    --train_ratio 0.8 \
    --dataset_name custom
```

### Arguments:
| Argument | Description | Default |
|----------|-------------|---------|
| `--input` | Directory containing PLY files | Required |
| `--output` | Output directory for processed data | Required |
| `--n_points` | Points to sample per mesh | 8192 |
| `--train_ratio` | Train/test split ratio | 0.8 |
| `--dataset_name` | Prefix for output files | custom |
| `--n_workers` | Parallel workers | Auto |

### Output Structure:
```
/path/to/processed/data/
├── custom_shape_names.txt          # List of class names
├── custom_train.txt                # Training sample list
├── custom_test.txt                 # Test sample list
├── custom_train_8192pts_fps.dat    # Training cache (fast loading)
├── custom_test_8192pts_fps.dat     # Test cache
├── table/
│   ├── table_0001.txt
│   ├── table_0002.txt
│   └── ...
├── monitor/
│   └── ...
└── ...
```

---

## Step 2: Update Configuration Files

### 2.1 Update Dataset Config

Edit `cfgs/dataset_configs/CustomDataset.yaml`:

```yaml
NAME: CustomDataset
DATA_PATH: /path/to/processed/data    # <-- Your output path from Step 1
N_POINTS: 8192
NUM_CATEGORY: 18                       # <-- Number of classes detected
USE_NORMALS: True
DATASET_NAME: custom
```

### 2.2 Update Finetune Config

Edit `cfgs/finetune_custom.yaml`:

```yaml
model : {
  NAME: PointTransformer,
  input_channel: 6,          # 6 for xyz+normals
  trans_dim: 384,
  depth: 12,
  drop_path_rate: 0.1,
  cls_dim: 18,               # <-- MUST match NUM_CATEGORY
  num_heads: 6,
  group_size: 32,
  num_group: 64,
  encoder_dims: 384,
}
```

**Critical**: `cls_dim` in the model config MUST equal `NUM_CATEGORY` in the dataset config!

---

## Step 3: Finetune the Model

### Option A: Using the Script

```bash
# Make script executable
chmod +x scripts/run_finetune_custom.sh

# Edit the script to set your pretrained checkpoint path
# Then run:
./scripts/run_finetune_custom.sh
```

### Option B: Direct Command

**Single GPU:**
```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --config cfgs/finetune_custom.yaml \
    --finetune_model \
    --ckpts /path/to/pretrained.pth \
    --exp_name finetune_custom
```

**Multi-GPU (8 GPUs):**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port=29501 \
    main.py \
    --config cfgs/finetune_custom.yaml \
    --finetune_model \
    --ckpts /path/to/pretrained.pth \
    --launcher pytorch \
    --exp_name finetune_custom
```

### Pretrained Checkpoints

You can use:
1. **Your own pretrained model**: `experiments/pretrain_modelnet_normals/cfgs/pretrain_6channel/ckpt-last.pth`
2. **Official Point-MAE pretrained**: Download from [Point-MAE releases](https://github.com/Pang-Yatian/Point-MAE/releases)

### Training Output

Checkpoints are saved to:
```
experiments/finetune_custom/cfgs/finetune_custom/
├── ckpt-best.pth      # Best validation accuracy
├── ckpt-last.pth      # Latest checkpoint
└── *.log              # Training logs
```

---

## Step 4: Evaluate on Test Set

```bash
CUDA_VISIBLE_DEVICES=0 python main.py \
    --test \
    --config cfgs/finetune_custom.yaml \
    --ckpts experiments/finetune_custom/cfgs/finetune_custom/ckpt-best.pth \
    --exp_name test_custom
```

Or use the script:
```bash
chmod +x scripts/run_test.sh
./scripts/run_test.sh
```

---

## Step 5: Classify New Objects

### Single File:
```bash
python tools/classify_single.py \
    --config cfgs/finetune_custom.yaml \
    --ckpts experiments/finetune_custom/cfgs/finetune_custom/ckpt-best.pth \
    --input /path/to/new_object.ply
```

### Directory of Files:
```bash
python tools/classify_single.py \
    --config cfgs/finetune_custom.yaml \
    --ckpts experiments/finetune_custom/cfgs/finetune_custom/ckpt-best.pth \
    --input /path/to/new_meshes/ \
    --output predictions.csv
```

### Output Example:
```
Table_new.ply:
  Prediction: table (95.2%)
  Top-3:
    - table: 95.2%
    - cabinet: 3.1%
    - box: 1.7%
```

---

## Complete Workflow Example

```bash
# 1. Preprocess data
python tools/preprocess_custom_dataset.py \
    --input /data/meshes \
    --output /data/processed \
    --n_points 8192

# 2. Update configs (edit the YAML files)
# - cfgs/dataset_configs/CustomDataset.yaml: Set DATA_PATH and NUM_CATEGORY
# - cfgs/finetune_custom.yaml: Set cls_dim to match NUM_CATEGORY

# 3. Finetune
CUDA_VISIBLE_DEVICES=0 python main.py \
    --config cfgs/finetune_custom.yaml \
    --finetune_model \
    --ckpts pretrain.pth \
    --exp_name finetune_custom

# 4. Test
CUDA_VISIBLE_DEVICES=0 python main.py \
    --test \
    --config cfgs/finetune_custom.yaml \
    --ckpts experiments/finetune_custom/cfgs/finetune_custom/ckpt-best.pth \
    --exp_name test_custom

# 5. Classify new objects
python tools/classify_single.py \
    --config cfgs/finetune_custom.yaml \
    --ckpts experiments/finetune_custom/cfgs/finetune_custom/ckpt-best.pth \
    --input /path/to/new_meshes/ \
    --output predictions.csv
```

---

## FAQ

### Q: Do I need to run finetuning separately from classification?
**A: Yes!** Finetuning trains the model on your dataset. Classification uses the trained model to predict classes for new objects. You must finetune first, then classify.

### Q: What if I have very few samples per class?
**A:** Point-MAE works well with limited data due to self-supervised pretraining. However, for best results:
- Use data augmentation (enabled by default)
- Consider few-shot learning config: `cfgs/fewshot.yaml`
- Ensure at least 2-3 samples per class for train/test split

### Q: How do I add more classes later?
**A:** 
1. Add new PLY files to your input directory
2. Re-run preprocessing
3. Update `NUM_CATEGORY` and `cls_dim` in configs
4. Re-train the model (finetuning from scratch or continue training)

### Q: Can I use 3-channel (xyz only) instead of 6-channel?
**A:** Yes, set:
- `USE_NORMALS: False` in dataset config
- `input_channel: 3` in model config

### Q: Training is slow, how can I speed it up?
**A:**
- Use multiple GPUs with distributed training
- Reduce `max_epoch` for quick experiments
- Increase batch size if GPU memory allows

---

## Files Created

| File | Purpose |
|------|---------|
| `tools/preprocess_custom_dataset.py` | Convert PLY to Point-MAE format |
| `tools/classify_single.py` | Classify new PLY files |
| `datasets/CustomDataset.py` | Dataset loader |
| `cfgs/dataset_configs/CustomDataset.yaml` | Dataset configuration |
| `cfgs/finetune_custom.yaml` | Finetuning configuration |
| `scripts/run_finetune_custom.sh` | Finetuning script |
| `scripts/run_classify.sh` | Classification script |
| `scripts/run_test.sh` | Test evaluation script |

---

## Troubleshooting

### Error: "Category file not found"
- Ensure preprocessing completed successfully
- Check `DATA_PATH` in `CustomDataset.yaml` points to correct directory

### Error: "cls_dim mismatch"
- `cls_dim` in `finetune_custom.yaml` must equal number of classes
- Check `custom_shape_names.txt` for actual class count

### Error: "CUDA out of memory"
- Reduce batch size in config (`bs: 2` instead of `bs: 4`)
- Reduce `n_points` (e.g., 4096 instead of 8192)

### Error: "No PLY files found"
- Ensure input directory contains `.ply` files
- Check file permissions
