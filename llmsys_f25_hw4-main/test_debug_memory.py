import numpy as np
import minitorch
from minitorch.cuda_kernel_ops import CudaKernelOps

np.random.seed(10)
backend = minitorch.TensorBackend(CudaKernelOps)

print("=" * 60)
print("Test 1: LayerNorm alone")
print("=" * 60)
try:
    data = np.random.rand(4, 64).astype(np.float32)
    X = minitorch.tensor_from_numpy(data, backend, True)
    gamma = minitorch.tensor([1.0] * 64, backend=backend, requires_grad=True)
    beta = minitorch.tensor([0.0] * 64, backend=backend, requires_grad=True)
    
    result = minitorch.LayerNorm.apply(X, gamma, beta)
    print(f"✓ Forward pass successful! Result shape: {result.shape}")
    
    result.sum().backward()
    print(f"✓ Backward pass successful!")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test 2: Attn_Softmax alone (with causal mask)")
print("=" * 60)
try:
    data = np.random.rand(1, 1, 2, 2).astype(np.float32)
    X = minitorch.tensor_from_numpy(data, backend, True)
    
    # Create causal mask (1, 1, 2, 2)
    mask_data = np.array([[[[0., -1e8], [0., 0.]]]], dtype=np.float32)
    mask = minitorch.tensor_from_numpy(mask_data, backend, False)
    
    result = minitorch.Attn_Softmax.apply(X, mask)
    print(f"✓ Forward pass successful! Result shape: {result.shape}")
    
    result.sum().backward()
    print(f"✓ Backward pass successful!")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test 3: LayerNorm → Attn_Softmax (chained)")
print("=" * 60)
try:
    data = np.random.rand(2, 4).astype(np.float32)
    X = minitorch.tensor_from_numpy(data, backend, True)
    gamma = minitorch.tensor([1.0] * 4, backend=backend, requires_grad=True)
    beta = minitorch.tensor([0.0] * 4, backend=backend, requires_grad=True)
    
    # LayerNorm
    ln_out = minitorch.LayerNorm.apply(X, gamma, beta)
    print(f"✓ LayerNorm forward successful! Shape: {ln_out.shape}")
    
    # Reshape for softmax (2, 4) → (1, 1, 2, 4)
    ln_reshaped = ln_out.view(1, 1, 2, 4)
    
    # Create zero mask
    mask_data = np.zeros((1, 1, 1, 4), dtype=np.float32)
    mask = minitorch.tensor_from_numpy(mask_data, backend, False)
    
    # Attn_Softmax
    sm_out = minitorch.Attn_Softmax.apply(ln_reshaped, mask)
    print(f"✓ Attn_Softmax forward successful! Shape: {sm_out.shape}")
    
    # Backward
    sm_out.sum().backward()
    print(f"✓ Backward pass successful!")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("Test 4: Simple MultiHeadAttention (minimal case)")
print("=" * 60)
try:
    data = np.random.rand(1, 2, 64).astype(np.float32)
    X = minitorch.tensor_from_numpy(data, backend, True)
    
    layer = minitorch.MultiHeadAttention(64, 1, True, 0.0, bias=False, backend=backend)
    
    result = layer(X)
    print(f"✓ Forward pass successful! Result shape: {result.shape}")
    
    result.sum().backward()
    print(f"✓ Backward pass successful!")
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("All tests completed!")
print("=" * 60)

