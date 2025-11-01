import matplotlib.pyplot as plt
plt.switch_backend('Agg')
import numpy as np
import json
import os
from pathlib import Path

def plot_training_time(means, stds, labels, fig_name):
    """Plot training time comparison"""
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
    print(f"Generated: {fig_name}")

def plot_throughput(means, stds, labels, fig_name):
    """Plot throughput comparison"""
    fig, ax = plt.subplots()
    ax.bar(np.arange(len(means)), means, yerr=stds,
           align='center', alpha=0.5, ecolor='red', capsize=10, width=0.6)
    ax.set_ylabel('GPT2 Throughput (Tokens per Second)')
    ax.set_xticks(np.arange(len(means)))
    ax.set_xticklabels(labels)
    ax.yaxis.grid(True)
    plt.tight_layout()
    plt.savefig(fig_name)
    plt.close(fig)
    print(f"Generated: {fig_name}")

def load_metrics(workdir, start_epoch=1, end_epoch=None):
    """Load training metrics from JSON files, excluding warmup epoch (epoch 0)"""
    training_times = []
    tokens_per_sec = []
    
    epoch = start_epoch
    while True:
        filepath = f'{workdir}/results_epoch{epoch}.json'
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

if __name__ == '__main__':
    # Create submit_figures directory if it doesn't exist
    os.makedirs('submit_figures', exist_ok=True)
    
    # Define workdirs
    model_parallel_dir = 'workdir_model_parallel'
    pipeline_parallel_dir = 'workdir_pipeline_parallel'
    
    # Check if directories exist
    if not os.path.exists(model_parallel_dir):
        print(f"Error: {model_parallel_dir} not found!")
        print("Please run: python project/run_pipeline.py --model_parallel_mode='model_parallel' --n_epochs=3")
        exit(1)
    
    if not os.path.exists(pipeline_parallel_dir):
        print(f"Error: {pipeline_parallel_dir} not found!")
        print("Please run: python project/run_pipeline.py --model_parallel_mode='pipeline_parallel' --n_epochs=3")
        exit(1)
    
    # Load data from model_parallel (excluding epoch 0 for warmup)
    mp_times, mp_tokens = load_metrics(model_parallel_dir, start_epoch=1)
    
    if len(mp_times) == 0:
        print(f"Error: No metrics found in {model_parallel_dir}!")
        print("Make sure to run with --n_epochs > 1 to have data after warmup epoch 0")
        exit(1)
    
    mp_time_mean = np.mean(mp_times)
    mp_time_std = np.std(mp_times)
    mp_tokens_mean = np.mean(mp_tokens)
    mp_tokens_std = np.std(mp_tokens)
    
    # Load data from pipeline_parallel (excluding epoch 0 for warmup)
    pp_times, pp_tokens = load_metrics(pipeline_parallel_dir, start_epoch=1)
    
    if len(pp_times) == 0:
        print(f"Error: No metrics found in {pipeline_parallel_dir}!")
        print("Make sure to run with --n_epochs > 1 to have data after warmup epoch 0")
        exit(1)
    
    pp_time_mean = np.mean(pp_times)
    pp_time_std = np.std(pp_times)
    pp_tokens_mean = np.mean(pp_tokens)
    pp_tokens_std = np.std(pp_tokens)
    
    # Print statistics
    print("="*70)
    print("TRAINING TIME STATISTICS (excluding epoch 0 warmup)")
    print("="*70)
    print(f"Model Parallel:    {mp_time_mean:.2f} ± {mp_time_std:.2f} seconds")
    print(f"Pipeline Parallel: {pp_time_mean:.2f} ± {pp_time_std:.2f} seconds")
    speedup_time = mp_time_mean / pp_time_mean if pp_time_mean > 0 else 0
    print(f"Speedup (time):    {speedup_time:.2f}x")
    print()
    
    print("="*70)
    print("THROUGHPUT STATISTICS (excluding epoch 0 warmup)")
    print("="*70)
    print(f"Model Parallel:    {mp_tokens_mean:.2f} ± {mp_tokens_std:.2f} tokens/sec")
    print(f"Pipeline Parallel: {pp_tokens_mean:.2f} ± {pp_tokens_std:.2f} tokens/sec")
    speedup_throughput = pp_tokens_mean / mp_tokens_mean if mp_tokens_mean > 0 else 0
    print(f"Speedup (throughput): {speedup_throughput:.2f}x")
    print()
    
    # Plot 1: Training time comparison
    plot_training_time(
        [mp_time_mean, pp_time_mean],
        [mp_time_std, pp_time_std],
        ['Model Parallel', 'Pipeline Parallel'],
        'submit_figures/pp_vs_mp_time.png'
    )
    
    # Plot 2: Throughput comparison
    plot_throughput(
        [mp_tokens_mean, pp_tokens_mean],
        [mp_tokens_std, pp_tokens_std],
        ['Model Parallel', 'Pipeline Parallel'],
        'submit_figures/pp_vs_mp_throughput.png'
    )
    
    print()
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Pipeline Parallel is {speedup_time:.2f}x faster in training time")
    print(f"Pipeline Parallel has {speedup_throughput:.2f}x higher throughput")
    print("="*70)

