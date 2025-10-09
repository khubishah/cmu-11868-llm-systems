import numpy as np
import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps

np.random.seed(10)
backend = minitorch.TensorBackend(CudaKernelOps)

print("Testing Attn_Softmax alone...")
data = np.random.rand(1, 1, 2, 2).astype(np.float32)
X = minitorch.tensor_from_numpy(data, backend, True)

# Create causal mask (1, 1, 2, 2)
mask_data = np.array([[[[0., -1e8], [0., 0.]]]], dtype=np.float32)
mask = minitorch.tensor_from_numpy(mask_data, backend, False)

print("Running forward...")
result = minitorch.Attn_Softmax.apply(X, mask)
print(f"Forward successful! Result shape: {result.shape}")

print("Running backward...")
result.sum().backward()
print("Backward successful!")
print(f"X.grad shape: {X.grad.shape if X.grad is not None else 'None'}")

