import numpy as np
import minitorch
import torch
from minitorch.cuda_kernel_ops import CudaKernelOps

np.random.seed(10)
torch.manual_seed(10)

backend = minitorch.TensorBackend(CudaKernelOps)
batch_size = 2
seq_len = 32
n_embd = 128
num_heads = 4
bias = False

data = np.random.randn(batch_size, seq_len, n_embd)
X = minitorch.tensor_from_numpy(data.copy(), backend, True)
X_ = torch.tensor(data, dtype=torch.float32, requires_grad=True)

layer_ = torch.nn.TransformerEncoderLayer(
    d_model=n_embd, nhead=num_heads, dim_feedforward=256, dropout=0,
    activation=lambda x: torch.nn.functional.gelu(x, approximate='tanh'),
    batch_first=True, norm_first=True, bias=bias, dtype=torch.float32
)

layer = minitorch.TransformerLayer(
    n_embd=n_embd, n_head=num_heads, p_dropout=0, ln_eps=layer_.norm1.eps, 
    bias=bias, backend=backend
)

# FFN Weights
w_ffn_in = layer_.linear1.weight.detach().numpy().T.copy()
w_ffn_out = layer_.linear2.weight.detach().numpy().T.copy()

# Transformer Weights
w_qkv = layer_.self_attn.in_proj_weight.detach().numpy().T.copy()
w_q_, w_k_, w_v_ = [w.copy() for w in np.split(w_qkv, 3, -1)]
w_out_ = layer_.self_attn.out_proj.weight.detach().numpy().T.copy()

# Set weights
layer.attention.q_projection.weights.value = minitorch.tensor_from_numpy(w_q_, backend=backend, requires_grad=True)
layer.attention.k_projection.weights.value = minitorch.tensor_from_numpy(w_k_, backend=backend, requires_grad=True)
layer.attention.v_projection.weights.value = minitorch.tensor_from_numpy(w_v_, backend=backend, requires_grad=True)
layer.attention.out_projection.weights.value = minitorch.tensor_from_numpy(w_out_, backend=backend, requires_grad=True)
layer.ff.linear_in.weights.value = minitorch.tensor_from_numpy(w_ffn_in, backend=backend, requires_grad=True)
layer.ff.linear_out.weights.value = minitorch.tensor_from_numpy(w_ffn_out, backend=backend, requires_grad=True)

# Mask for Torch
M = torch.triu(-float("inf")*torch.ones(seq_len, seq_len),1)

print("Running forward pass...")
result = layer(X)
result_ = layer_(X_, M)

diff = np.abs(result.to_numpy() - result_.detach().numpy())
print(f'\nForward pass results:')
print(f'  Max absolute diff: {diff.max():.8e}')
print(f'  Mean absolute diff: {diff.mean():.8e}')
print(f'  Mismatched elements (>1e-5): {(diff > 1e-5).sum()}/{diff.size} ({100*(diff > 1e-5).sum()/diff.size:.2f}%)')
print(f'  Forward pass tolerance check (atol=1e-5, rtol=1e-5): ', end='')
try:
    np.testing.assert_allclose(result.to_numpy(), result_.detach().numpy(), atol=1e-5, rtol=1e-5)
    print("PASS")
except AssertionError:
    print("FAIL")

print("\nRunning backward pass...")
result.sum().backward()
result_.sum().backward()

grad_diff = np.abs(X.grad.to_numpy() - X_.grad.detach().numpy())
print(f'\nBackward pass results:')
print(f'  Max absolute diff: {grad_diff.max():.8e}')
print(f'  Mean absolute diff: {grad_diff.mean():.8e}')
print(f'  Mismatched elements (>1e-5): {(grad_diff > 1e-5).sum()}/{grad_diff.size} ({100*(grad_diff > 1e-5).sum()/grad_diff.size:.2f}%)')
print(f'  Backward pass tolerance check (atol=1e-5, rtol=1e-5): ', end='')
try:
    np.testing.assert_allclose(X.grad.to_numpy(), X_.grad.detach().numpy(), atol=1e-5, rtol=1e-5)
    print("PASS")
except AssertionError as e:
    print("FAIL")
    print(f'\n  Error: {str(e).split(chr(10))[1]}')  # Print just the summary line

