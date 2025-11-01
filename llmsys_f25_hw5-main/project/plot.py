import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np
import json
import os
from pathlib import Path

def plot_training_time(means, stds, labels, fig_name):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('GPT2 Execution Time (Second)')
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)

def plot_throughput(means, stds, labels, fig_name):
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('Tokens per Second')
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)

def load_metrics(workdir, rank, start_epoch=1, end_epoch=None):
    """Load training metrics from JSON files, excluding warmup epoch (epoch 0)"""
    training_times = []
    tokens_per_sec = []
    
    epoch = start_epoch
    while True:
        filepath = f'{workdir}/rank{rank}_results_epoch{epoch}.json'
        if not os.path.exists(filepath):
            break
        if end_epoch is not None and epoch > end_epoch:
            break
            
        with open(filepath, 'r') as f:
            data = json.load(f)
            training_times.append(data['training_time'])
            tokens_per_sec.append(data['tokens_per_sec'])
        epoch += 1
    
    return training_times, tokens_per_sec

# Fill the data points here
if __name__ == '__main__':
    # Create submit_figures directory if it doesn't exist
    os.makedirs('submit_figures', exist_ok=True)
    
    # Load data from workdir_single (single GPU, excluding epoch 0)
    single_times, single_tokens = load_metrics('workdir_single', rank=0, start_epoch=1)
    single_mean = np.mean(single_times)
    single_std = np.std(single_times)
    single_tokens_mean = np.mean(single_tokens)
    single_tokens_std = np.std(single_tokens)
    
    # Load data from workdir_2gpu (2 GPUs, excluding epoch 0)
    gpu0_times, gpu0_tokens = load_metrics('workdir_2gpu', rank=0, start_epoch=1)
    gpu1_times, gpu1_tokens = load_metrics('workdir_2gpu', rank=1, start_epoch=1)
    
    gpu0_mean = np.mean(gpu0_times)
    gpu0_std = np.std(gpu0_times)
    gpu1_mean = np.mean(gpu1_times)
    gpu1_std = np.std(gpu1_times)
    
    # Throughput is the sum of tokens_per_sec across all GPUs
    gpu0_tokens_mean = np.mean(gpu0_tokens)
    gpu0_tokens_std = np.std(gpu0_tokens)
    gpu1_tokens_mean = np.mean(gpu1_tokens)
    gpu1_tokens_std = np.std(gpu1_tokens)
    
    # Total throughput for 2-GPU setup
    total_2gpu_tokens_mean = gpu0_tokens_mean + gpu1_tokens_mean
    # For standard deviation of sum, we use sqrt(std1^2 + std2^2) assuming independence
    total_2gpu_tokens_std = np.sqrt(gpu0_tokens_std**2 + gpu1_tokens_std**2)
    
    print("="*60)
    print("TRAINING TIME STATISTICS (excluding epoch 0 warmup)")
    print("="*60)
    print(f"Single GPU: {single_mean:.2f} ± {single_std:.2f} seconds")
    print(f"Data Parallel GPU0: {gpu0_mean:.2f} ± {gpu0_std:.2f} seconds")
    print(f"Data Parallel GPU1: {gpu1_mean:.2f} ± {gpu1_std:.2f} seconds")
    print()
    print("="*60)
    print("THROUGHPUT STATISTICS (excluding epoch 0 warmup)")
    print("="*60)
    print(f"Single GPU: {single_tokens_mean:.2f} ± {single_tokens_std:.2f} tokens/sec")
    print(f"Data Parallel (2 GPUs): {total_2gpu_tokens_mean:.2f} ± {total_2gpu_tokens_std:.2f} tokens/sec")
    print(f"  - GPU0: {gpu0_tokens_mean:.2f} ± {gpu0_tokens_std:.2f} tokens/sec")
    print(f"  - GPU1: {gpu1_tokens_mean:.2f} ± {gpu1_tokens_std:.2f} tokens/sec")
    print()
    
    # Plot 1: Training time comparison
    plot_training_time(
        [gpu0_mean, gpu1_mean, single_mean],
        [gpu0_std, gpu1_std, single_std],
        ['Data Parallel - GPU0', 'Data Parallel - GPU1', 'Single GPU'],
        'submit_figures/ddp_vs_single_time.png'
    )
    print("Generated: submit_figures/ddp_vs_single_time.png")
    
    # Plot 2: Throughput comparison
    plot_throughput(
        [total_2gpu_tokens_mean, single_tokens_mean],
        [total_2gpu_tokens_std, single_tokens_std],
        ['Data Parallel (2 GPUs)', 'Single GPU'],
        'submit_figures/ddp_vs_single_throughput.png'
    )
    print("Generated: submit_figures/ddp_vs_single_throughput.png")