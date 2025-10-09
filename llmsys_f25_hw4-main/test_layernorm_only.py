import numpy as np
import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps

np.random.seed(10)
backend = minitorch.TensorBackend(CudaKernelOps)

print("Testing LayerNorm alone...")
data = np.random.rand(4, 64).astype(np.float32)
X = minitorch.tensor_from_numpy(data, backend, True)
gamma = minitorch.tensor([1.0] * 64, backend=backend, requires_grad=True)
beta = minitorch.tensor([0.0] * 64, backend=backend, requires_grad=True)

print("Running forward...")
result = minitorch.LayerNorm.apply(X, gamma, beta)
print(f"Forward successful! Result shape: {result.shape}")

print("Running backward...")
result.sum().backward()
print("Backward successful!")
print(f"X.grad shape: {X.grad.shape if X.grad is not None else 'None'}")

