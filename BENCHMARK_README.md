# Point-MAE Inference Benchmark Guide

This guide explains how to run inference timing benchmarks on the **REAL Point-MAE architecture** using GPU.

## What This Benchmark Measures

1. **Encoder Forward Pass** - Time for Point-MAE encoder to process 1 point cloud
2. **Full Classification Inference** - Complete inference including classification head
3. **Component Breakdown**:
   - FPS + KNN Grouping (CUDA)
   - Point Encoder (Mini-PointNet)
   - Positional Embedding
   - Transformer Blocks
   - Classification Head

## Understanding the Metrics

| Metric | Meaning |
|--------|---------|
| **Mean ± Std** | Average latency across N runs with standard deviation |
| **Median** | Middle value (50th percentile) - robust to outliers |
| **Min/Max** | Fastest and slowest runs |
| **P95** | 95th percentile - 95% of runs are faster than this |
| **P99** | 99th percentile - 99% of runs are faster than this |
| **Throughput** | Samples processed per second (1000 / mean_ms) |

### Why ± (Standard Deviation)?

The benchmark runs inference **100 times** on the same input and measures each run. The `±` value is the **standard deviation** - it shows how much the timing varies between runs. Lower std = more consistent performance.

---

## Running on NVIDIA Server (8x A100 GPUs)

### Prerequisites

```bash
# Ensure CUDA is available
nvidia-smi

# Activate your environment
conda activate pointmae  # or your environment name
```

### Option 1: Single GPU Benchmark (Recommended for Timing)

For accurate timing, use **single GPU** (multi-GPU adds communication overhead):

```bash
cd /app/sid_gigs/nvidia_brev_pont_mae_class/PointMAE_classification

# Use GPU 0
CUDA_VISIBLE_DEVICES=0 python tools/benchmark_real_model.py \
    --config cfgs/finetune_modelnet.yaml \
    --ckpts experiments/finetune_cpu/finetune_50epochs/ckpt-best.pth \
    --n_points 8192 \
    --num_runs 100 \
    --num_warmup 50 \
    --batch_size 1 \
    --output_dir benchmark_results
```

### Option 2: Benchmark on Different GPUs

Run on each A100 separately to compare:

```bash
# GPU 0
CUDA_VISIBLE_DEVICES=0 python tools/benchmark_real_model.py \
    --config cfgs/finetune_modelnet.yaml \
    --n_points 8192 \
    --num_runs 100 \
    --output_dir benchmark_results/gpu0

# GPU 1
CUDA_VISIBLE_DEVICES=1 python tools/benchmark_real_model.py \
    --config cfgs/finetune_modelnet.yaml \
    --n_points 8192 \
    --num_runs 100 \
    --output_dir benchmark_results/gpu1
```

### Option 3: Batch Size Scaling Test

Test how throughput scales with batch size:

```bash
# Batch size 1
CUDA_VISIBLE_DEVICES=0 python tools/benchmark_real_model.py \
    --config cfgs/finetune_modelnet.yaml \
    --n_points 8192 \
    --batch_size 1 \
    --output_dir benchmark_results/batch1

# Batch size 8
CUDA_VISIBLE_DEVICES=0 python tools/benchmark_real_model.py \
    --config cfgs/finetune_modelnet.yaml \
    --n_points 8192 \
    --batch_size 8 \
    --output_dir benchmark_results/batch8

# Batch size 16
CUDA_VISIBLE_DEVICES=0 python tools/benchmark_real_model.py \
    --config cfgs/finetune_modelnet.yaml \
    --n_points 8192 \
    --batch_size 16 \
    --output_dir benchmark_results/batch16

# Batch size 32
CUDA_VISIBLE_DEVICES=0 python tools/benchmark_real_model.py \
    --config cfgs/finetune_modelnet.yaml \
    --n_points 8192 \
    --batch_size 32 \
    --output_dir benchmark_results/batch32
```

---

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--config` | `cfgs/finetune_modelnet.yaml` | Model config file |
| `--ckpts` | None | Checkpoint path (optional) |
| `--n_points` | 8192 | Number of input points |
| `--num_runs` | 100 | Number of benchmark runs |
| `--num_warmup` | 50 | Warmup runs before timing |
| `--batch_size` | 1 | Batch size for inference |
| `--output_dir` | `benchmark_results` | Output directory |
| `--gpu` | 0 | GPU ID to use |

---

## Output Files

After running, you'll get:

```
benchmark_results/
├── real_model_architecture_YYYYMMDD_HHMMSS.txt   # Full model layer tree
├── real_benchmark_results_YYYYMMDD_HHMMSS.json   # Raw timing data (JSON)
└── real_paper_summary_YYYYMMDD_HHMMSS.md         # Paper-ready summary
```

### Example Output (Paper Summary)

```markdown
## Encoder Forward Pass (Single Point Cloud, Batch Size = 1)

| Metric | Value |
|--------|-------|
| Mean latency | 5.23 ± 0.45 ms |
| Median latency | 5.12 ms |
| 95th percentile | 6.01 ms |
| Throughput | 191.2 samples/sec |

## Component-wise Latency Breakdown

| Component | Latency (ms) | % of Encoder |
|-----------|-------------|--------------|
| FPS + KNN Grouping | 1.82 ± 0.12 | 34.8% |
| Point Encoder | 0.95 ± 0.08 | 18.2% |
| Positional Embedding | 0.05 ± 0.01 | 1.0% |
| Transformer Blocks | 2.38 ± 0.25 | 45.5% |
| Classification Head | 0.03 ± 0.01 | N/A |
```

---

## Expected Results on A100

Based on A100 specifications, you should expect approximately:

| Metric | Expected Range |
|--------|----------------|
| Encoder (batch=1) | 3-8 ms |
| Classification (batch=1) | 4-10 ms |
| Throughput (batch=1) | 100-300 samples/sec |
| Throughput (batch=32) | 500-2000 samples/sec |

**Note**: Actual results depend on:
- GPU utilization
- Memory bandwidth
- CUDA/cuDNN versions
- Other processes on the GPU

---

## Troubleshooting

### CUDA Out of Memory

Reduce batch size:
```bash
python tools/benchmark_real_model.py --batch_size 1
```

### KNN CUDA Not Found

Ensure knn_cuda is installed:
```bash
pip install --upgrade https://github.com/unlimblue/KNN_CUDA/releases/download/0.2/KNN_CUDA-0.2-py3-none-any.whl
```

### Config File Not Found

Use the correct config path:
```bash
python tools/benchmark_real_model.py --config cfgs/finetune_custom.yaml
```

---

## For Paper: Recommended Benchmark Protocol

1. **Warmup**: 50 runs (ensures GPU is at full clock speed)
2. **Measurement**: 100 runs (statistically significant)
3. **Batch size**: Report both batch=1 (latency) and batch=32 (throughput)
4. **Report**: Mean ± std, median, P95, throughput
5. **Hardware**: Specify GPU model, CUDA version, PyTorch version

### Example Paper Text

> We benchmark inference latency on a single NVIDIA A100 GPU (40GB). 
> For a single point cloud with 8,192 points, the Point-MAE encoder 
> achieves a mean latency of **X.XX ± X.XX ms** (median: X.XX ms, 
> P95: X.XX ms), corresponding to a throughput of **XXX samples/sec**. 
> The transformer blocks account for XX% of the total encoder time, 
> while FPS+KNN grouping accounts for XX%.

---

## Quick Start

```bash
# SSH to NVIDIA server
ssh user@nvidia-server

# Navigate to project
cd /app/sid_gigs/nvidia_brev_pont_mae_class/PointMAE_classification

# Run benchmark
CUDA_VISIBLE_DEVICES=0 python tools/benchmark_real_model.py \
    --config cfgs/finetune_modelnet.yaml \
    --n_points 8192 \
    --num_runs 100 \
    --output_dir benchmark_results

# View results
cat benchmark_results/real_paper_summary_*.md
```
