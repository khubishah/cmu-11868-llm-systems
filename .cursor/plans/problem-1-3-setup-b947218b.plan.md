<!-- b947218b-8022-4491-ad50-48ad353c0efc cda2573c-9ebd-4013-a603-8927a339fc3c -->
# Problem 1.3: Performance Comparison Setup

## Overview

Set up infrastructure to compare single-GPU vs 2-GPU training with proper logging, warmup handling, and background job submission.

## Implementation Steps

### 1. Add --workdir Command-Line Argument

**File**: `project/run_data_parallel.py`

- Add `parser.add_argument('--workdir', type=str, default='./workdir')` to argparse section (around line 185-191)
- Pass `workdir` parameter to `run_dp()` function call (line 210-220)
- Update `run_dp()` function signature to accept `workdir` parameter (line 57-64)
- Use the `workdir` parameter instead of hardcoded `'./workdir'` (line 65)

### 2. Update plot.py for Manual Warmup Exclusion

**File**: `project/plot.py`

Current structure has placeholder variables. Update to:

- Add code to read JSON files from workdir directories
- Load epochs 1-4 (exclude epoch 0 as warmup) from `rank{rank}_results_epoch{epoch}.json`
- Compute mean and std for training_time and tokens_per_sec
- Create two plots:
  - Plot 1: Training time comparison (GPU0, GPU1, Single GPU)
  - Plot 2: Throughput comparison (2-GPU sum vs Single GPU)
- Save plots to `submit_figures/` directory

### 3. Create Test Commands for Interactive Session

**Command 1 - Single GPU (1 epoch test)**:

```bash
python project/run_data_parallel.py --world_size 1 --batch_size 64 --n_epochs 1 --workdir workdir_single_test
```

**Command 2 - 2 GPUs (1 epoch test)**:

```bash
python project/run_data_parallel.py --world_size 2 --batch_size 128 --n_epochs 1 --workdir workdir_2gpu_test
```

### 4. Create Sbatch Scripts for Background Submission

**File**: `job_single_gpu.sh`

- Request 1 GPU (GPU-shared partition)
- Run for 5 epochs with world_size=1, batch_size=64
- Save to `workdir_single`
- Time limit: 2 hours

**File**: `job_2gpu.sh`

- Request 2 GPUs (GPU-shared partition)
- Run for 5 epochs with world_size=2, batch_size=128
- Save to `workdir_2gpu`
- Time limit: 2 hours

### 5. Directory Structure

- `workdir_single_test/` - single GPU test results (1 epoch)
- `workdir_2gpu_test/` - 2-GPU test results (1 epoch)
- `workdir_single/` - single GPU full results (5 epochs)
- `workdir_2gpu/` - 2-GPU full results (5 epochs)
- `submit_figures/` - final plots

## Expected Workflow

1. Test both configurations with 1 epoch in interactive session
2. Verify metrics are being logged correctly
3. Submit both sbatch jobs to run 5 epochs in background
4. Continue with Problem 2 while training runs
5. After training completes, run plot.py to generate figures

## Key Metrics

- **Training Time**: Average per-epoch time (excluding epoch 0) for each GPU
- **Throughput**: Sum of tokens_per_sec across GPUs (excluding epoch 0)