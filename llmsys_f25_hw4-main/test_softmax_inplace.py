import numpy as np
import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps

np.random.seed(10)
backend = minitorch.TensorBackend(CudaKernelOps)

print("Testing if attn_softmax modifies input in-place...")
data = np.random.rand(1, 1, 2, 2).astype(np.float32)
print(f"Original data: {data}")

X = minitorch.tensor_from_numpy(data.copy(), backend, True)
print(f"Input tensor before: {X.to_numpy()}")

# Create causal mask (1, 1, 2, 2)
mask_data = np.array([[[[0., -1e8], [0., 0.]]]], dtype=np.float32)
mask = minitorch.tensor_from_numpy(mask_data, backend, False)

result = minitorch.Attn_Softmax.apply(X, mask)
print(f"Result tensor: {result.to_numpy()}")
print(f"Input tensor after: {X.to_numpy()}")
print(f"Are they the same object? {result is X}")
print(f"Do they share storage? {result._tensor._storage is X._tensor._storage}")

print("\nTest completed successfully (no backward)")

