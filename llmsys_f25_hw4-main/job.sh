#!/bin/bash
#SBATCH -N 1
#SBATCH -p GPU-shared
#SBATCH -t 24:00:00
#SBATCH --gpus=v100-32:4
#SBATCH --output=output_%j.log      # Standard output file (%j will be replaced with job ID)
#SBATCH --error=error_%j.log        # Standard error file (%j will be replaced with job ID)

# load conda
module load anaconda3/2024.10-1

module load cuda/12.4

# activate environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate hw4

# ensure conda's C++ runtime is preferred (fixes CXXABI mismatch for PyCUDA)
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH}"
nvidia-smi
cd /jet/home/kshah10/cmu-11868-llm-systems/llmsys_f25_hw4-main

# build CUDA kernels needed by tests
bash compile_cuda.sh

# execute your test
echo "Non-FUSED kernel training started"
python project/run_machine_translation.py --use_fused_kernel=False --n_epochs=1
echo "Non-FUSED kernel training complete"

echo "FUSED kernel training started"
python project/run_machine_translation.py --use_fused_kernel=True --n_epochs=1
echo "FUSED kernel training complete"

# or start a notebook server to attach your .ipynb
#jupyter notebook --no-browser --ip=0.0.0.0 --port=8888
