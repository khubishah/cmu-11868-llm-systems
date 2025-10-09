import numpy as np
import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps

np.random.seed(10)
backend = minitorch.TensorBackend(CudaKernelOps)

print("Testing softmax backward in isolation...")

# Simulate what the backward pass receives
out_grad_data = np.random.rand(1, 1, 2, 2).astype(np.float32)
soft_inp_data = np.random.rand(1, 1, 2, 2).astype(np.float32)

out_grad = minitorch.tensor_from_numpy(out_grad_data, backend, False)
soft_inp = minitorch.tensor_from_numpy(soft_inp_data, backend, False)

print(f"out_grad shape: {out_grad.shape}")
print(f"soft_inp shape: {soft_inp.shape}")

# Call the backward kernel directly
grad_inp = out_grad.f.attn_softmax_bw(out_grad, soft_inp)

print(f"grad_inp shape: {grad_inp.shape}")
print(f"Are they the same object (out_grad, grad_inp)? {out_grad is grad_inp}")
print(f"Do they share storage? {out_grad._tensor._storage is grad_inp._tensor._storage}")

print("\nBackward kernel completed successfully!")

