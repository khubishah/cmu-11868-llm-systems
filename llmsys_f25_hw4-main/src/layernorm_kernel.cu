#include "includes/block_reduce.h"
#include "includes/kernels.h"
#include "includes/cuda_util.h"

#include <cooperative_groups.h>
#include <cstddef>

namespace cg = cooperative_groups;
namespace lightseq {
namespace cuda {

const float LN_EPSILON = 1e-8f;
#define TILE_DIM 32


/**
@brief: ker_layer_norm
Standard layer normalization.
It will not only output the layer norm result,
  but also outputs variance.
  may also output means, depends on whether
  the means argument is nullptr

@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
ln_res: [batch_size * seq_len, hidden_size], ln result.
vars: [batch_size * seq_len], variance per token
means: [batch_size * seq_len], means per token, can be nullput
inp: [batch_size * seq_len, hidden_size], ln input.
scale: [hidden_size], ln scale
bias: [hidden_size], ln bias
*/
template <typename T>
__global__ void ker_layer_norm(T *ln_res, T *vars, T *means, const T *inp,
                               const T *scale, const T *bias, int hidden_size) {
  
  /// BEGIN ASSIGN4_2_1
  // Step 1: Compute sum of x and sum of x^2 concurrently using float4
  float l_sum = 0.0f;        // sum of x
  float l_sum_of_squares = 0.0f;  // sum of x^2
  
  const float4 *inp_f4 = reinterpret_cast<const float4 *>(inp) + blockIdx.x * hidden_size;  
  
  // Each thread processes multiple float4 elements
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 val = inp_f4[idx];
    
    // Process all 4 components
    l_sum += val.x + val.y + val.z + val.w;
    l_sum_of_squares += val.x * val.x + val.y * val.y + val.z * val.z + val.w * val.w;
  }

  // Step 2: Block reduction to compute global sums
  // Reduce both sums together to avoid race conditions with shared memory
  float sums[2] = {l_sum, l_sum_of_squares};
  blockReduce<ReduceType::kSum, 2>(sums);
  l_sum = sums[0];
  l_sum_of_squares = sums[1];
  
  // Compute mean and variance (only thread 0 has the correct sums)
  __shared__ float s_mean, s_variance;
  if (threadIdx.x == 0) {
    int total_elements = hidden_size * 4;  // hidden_size is divided by 4, so multiply back
    s_mean = l_sum / total_elements;
    float mean_of_squares = l_sum_of_squares / total_elements;
    // Ensure variance is non-negative before sqrt to avoid NaN
    float variance = fmaxf(mean_of_squares - s_mean * s_mean, 0.0f);
    s_variance = sqrtf(variance + LN_EPSILON);
    
    // Store mean and variance for output
    if (means != nullptr) {
      means[blockIdx.x] = s_mean;
    }
    vars[blockIdx.x] = s_variance;
  }
  __syncthreads();

  // Step 3: Apply LayerNorm transformation using float4 for speedup
  float4 *ln_res_f4 = reinterpret_cast<float4 *>(ln_res) + blockIdx.x * hidden_size;
  const float4 *scale_f4 = reinterpret_cast<const float4 *>(scale);
  const float4 *bias_f4 = reinterpret_cast<const float4 *>(bias);
  
  // Ensure s_variance is not too small to avoid Inf/NaN
  float safe_variance = fmaxf(s_variance, LN_EPSILON);
  float inv_variance = 1.0f / safe_variance;
  
  for (uint idx = threadIdx.x; idx < hidden_size; idx += blockDim.x) {
    float4 val = inp_f4[idx];
    float4 scale_val = scale_f4[idx];
    float4 bias_val = bias_f4[idx];
    
    // Apply LayerNorm: y = γ * (x - μ) / σ + β
    float4 result;
    result.x = scale_val.x * (val.x - s_mean) * inv_variance + bias_val.x;
    result.y = scale_val.y * (val.y - s_mean) * inv_variance + bias_val.y;
    result.z = scale_val.z * (val.z - s_mean) * inv_variance + bias_val.z;
    result.w = scale_val.w * (val.w - s_mean) * inv_variance + bias_val.w;
    
    ln_res_f4[idx] = result;
  }
  /// END ASSIGN4_2_1
}

extern "C" {
void launch_layernorm(float *ln_res, float *vars, float *means,
                              const float *inp, const float *scale,
                              const float *bias, int batch_size, int hidden_dim,
                              cudaStream_t stream) {
  if (hidden_dim % 4 != 0) {
    throw std::runtime_error("violate hidden_dim % 4 = 0");
  }
  int float_size = sizeof(float);
  int input_size = batch_size * hidden_dim * float_size;
  int scale_size = hidden_dim * float_size;
  int bias_size = hidden_dim * float_size;
  int output_size = batch_size * hidden_dim * float_size;
  int mean_size = batch_size * float_size;
  int var_size = batch_size * float_size;


  float *d_ln_res, *d_vars, *d_means, *d_inp, *d_scale, *d_bias;
  cudaMalloc((void **)&d_ln_res, output_size);
  cudaMalloc((void **)&d_vars, var_size);
  cudaMalloc((void **)&d_means, mean_size);
  cudaMalloc((void **)&d_inp, input_size);
  cudaMalloc((void **)&d_scale, scale_size);
  cudaMalloc((void **)&d_bias, bias_size);

  cudaMemcpy(d_inp, inp, input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_scale, scale, scale_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_bias, bias, bias_size, cudaMemcpyHostToDevice);

  // For using float4
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  dim3 grid_dim(batch_size);
  dim3 block_dim(nthread);

  ker_layer_norm<float><<<grid_dim, block_dim, 0, stream>>>(
      d_ln_res, d_vars, d_means, d_inp, d_scale, d_bias, hidden_dim);

  // Copy back to the host
  cudaMemcpy(ln_res, d_ln_res, output_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(vars, d_vars, var_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(means, d_means, mean_size, cudaMemcpyDeviceToHost);
  cudaDeviceSynchronize();

  // Check CUDA execution
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm Error: %s\n", cudaGetErrorString(err));
    // Handle the error (e.g., by exiting the program)
    exit(EXIT_FAILURE);
  }

  // Free memory on device
  cudaFree(d_ln_res);
  cudaFree(d_vars);
  cudaFree(d_means);
  cudaFree(d_inp);
  cudaFree(d_scale);
  cudaFree(d_bias);

}
}

/**
@brief: ker_ln_bw_dgamma_dbetta
Layer norm backword kernel, compute the gradient of gamma and betta.
dbetta = sum(dout, dim=0)
dgamma = sum(xhat * dout, dim=0)
xhat = (input - mean) * rsqrt(var) or
  (output - betta) / gamma

@thread
gridDim.x = hidden_size / 32
blockDim.x = 32
blockDim.y = 32

@param
gamma_grad: [hidden_size], gradient of gamma
betta_grad: [hidden_size], gradient of betta
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat, maybe nullptr
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat, maybe nullptr
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
(gamma && betta) ^ (vars && means) should be true
*/
template <typename T>
__global__ void ker_ln_bw_dgamma_dbetta(T *gamma_grad, T *betta_grad,
                                        const T *out_grad,
                                        const T *inp, const T *gamma,
                                        const T *betta, const T *vars,
                                        const T *means, int rows, int width) {

  /// BEGIN ASSIGN4_2_2
  __shared__ float betta_buffer[TILE_DIM][TILE_DIM];
  __shared__ float gamma_buffer[TILE_DIM][TILE_DIM];

  cg::thread_block b = cg::this_thread_block();
  cg::thread_block_tile<TILE_DIM> g = cg::tiled_partition<TILE_DIM>(b);

  // Step 1: Compute the partial gradients by looping across inp rows
  int col = blockIdx.x * TILE_DIM + threadIdx.x;
  
  float local_betta_grad = 0.0f;
  float local_gamma_grad = 0.0f;
  
  if (col < width) {
    for (int row = threadIdx.y; row < rows; row += TILE_DIM) {
      int idx = row * width + col;
      
      // Compute xhat = (input - mean) / sqrt(var)
      float inp_val = (float)inp[idx];
      float mean_val = (float)means[row];
      float var_val = (float)vars[row];
      float xhat = (inp_val - mean_val) / var_val;  // vars already contains sqrt(var + epsilon)
      
      float out_grad_val = (float)out_grad[idx];
      
      // Accumulate gradients
      local_betta_grad += out_grad_val;                    // ∇β = Σ(∇y)
      local_gamma_grad += out_grad_val * xhat;             // ∇γ = Σ(∇y * x̂)
    }
  }

  // Step 2: Store the partial gradients in the shared memory arrays
  betta_buffer[threadIdx.y][threadIdx.x] = local_betta_grad;
  gamma_buffer[threadIdx.y][threadIdx.x] = local_gamma_grad;
  __syncthreads();

  // Step 3: Reduce across threadIdx.y dimension using shared memory
  // Note: Could be optimized further with g.shfl_down for warp-level efficiency
  if (threadIdx.y == 0 && col < width) {
    float final_betta_grad = 0.0f;
    float final_gamma_grad = 0.0f;
    
    // Sum across all threadIdx.y values
    for (int i = 0; i < TILE_DIM; i++) {
      final_betta_grad += betta_buffer[i][threadIdx.x];
      final_gamma_grad += gamma_buffer[i][threadIdx.x];
    }
    
    // Step 4: Assign the final result to the correct position in the global output
    betta_grad[col] = (T)final_betta_grad;
    gamma_grad[col] = (T)final_gamma_grad;
  }
  /// END ASSIGN4_2_2
}

/**
@brief: ker_ln_bw_dinp
Layer norm backword kernel, compute the gradient of input.
dinp = (dxhat - (sum(dxhat) + xhat * sum(dxhat * xhat)) / hidden_dim)
  * rsqrt(var)
xhat = (input - mean) * rsqrt(var) if mean is not nullptr
       (output - betta) / gamma if mean is nullptr
dxhat = dout * gamma


@thread
gridDim.x = batch_size * seq_len
blockDim.x = hidden_size

@param
inp_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
out_grad: [batch_size * seq_len, hidden_size], gradient of betta ln output
residual_grad: [batch_size * seq_len, hidden_size], gradient of residual input,
  usually appear in pre-layer-norm for transformer layer, maybe nullptr
inp_or_out: [batch_size * seq_len, hidden_size], ln output if means is nullptr
  ln input if means is not nullptr
gamma: [hidden_size], gamma of ln,
  used to compute xhat and dxhat
betta: [hidden_size], betta of ln,
  used to compute xhat, maybe nullptr
vars: [batch_size * seq_len], variance of ln forward,
  used to compute xhat and dinp
means: [batch_size * seq_len], mean of ln forward,
  used to compute xhat, maybe nullptr
*/
template <typename T>
__global__ void ker_ln_bw_dinp(T *inp_grad, const T *out_grad, const T *inp,
                               const T *gamma, const T *betta, const T *vars,
                               const T *means, int hidden_dim) {
  
  /// BEGIN ASSIGN4_2_2
  int batch_idx = blockIdx.x;
  float mean_val = (float)means[batch_idx];
  float var_val = (float)vars[batch_idx];  // This is sqrt(var + epsilon)
  // Ensure var_val is not too small to avoid Inf/NaN
  float safe_var = fmaxf(var_val, LN_EPSILON);
  float inv_var = 1.0f / safe_var;
  
  // Step 1 & 2: Compute dxhat=dy*w and xhat with float4 for speedup
  const float4 *out_grad_f4 = reinterpret_cast<const float4 *>(out_grad) + batch_idx * hidden_dim;
  const float4 *gamma_f4 = reinterpret_cast<const float4 *>(gamma);
  const float4 *inp_f4 = reinterpret_cast<const float4 *>(inp) + batch_idx * hidden_dim;
  
  float l_sum_dxhat = 0.0f;
  float l_sum_dxhat_xhat = 0.0f;
  
  // First pass: compute partial sums
  for (uint idx = threadIdx.x; idx < hidden_dim; idx += blockDim.x) {
    float4 out_grad_val = out_grad_f4[idx];
    float4 gamma_val = gamma_f4[idx];
    float4 inp_val = inp_f4[idx];
    
    // Compute xhat = (input - mean) / sqrt(var + epsilon)
    float xhat_x = (inp_val.x - mean_val) * inv_var;
    float xhat_y = (inp_val.y - mean_val) * inv_var;
    float xhat_z = (inp_val.z - mean_val) * inv_var;
    float xhat_w = (inp_val.w - mean_val) * inv_var;
    
    // Compute dxhat = dout * gamma
    float dxhat_x = out_grad_val.x * gamma_val.x;
    float dxhat_y = out_grad_val.y * gamma_val.y;
    float dxhat_z = out_grad_val.z * gamma_val.z;
    float dxhat_w = out_grad_val.w * gamma_val.w;
    
    // Accumulate sums
    l_sum_dxhat += dxhat_x + dxhat_y + dxhat_z + dxhat_w;
    l_sum_dxhat_xhat += dxhat_x * xhat_x + dxhat_y * xhat_y + dxhat_z * xhat_z + dxhat_w * xhat_w;
  }
  
  // Step 3: Compute reduce sum for dxhat and dxhat*xhat with blockReduce
  blockReduce<ReduceType::kSum, 1>(&l_sum_dxhat);
  blockReduce<ReduceType::kSum, 1>(&l_sum_dxhat_xhat);
  
  __shared__ float s_sum_dxhat, s_sum_dxhat_xhat;
  if (threadIdx.x == 0) {
    int total_elements = hidden_dim * 4;  // hidden_dim is divided by 4
    s_sum_dxhat = l_sum_dxhat / total_elements;
    s_sum_dxhat_xhat = l_sum_dxhat_xhat / total_elements;
  }
  __syncthreads();
  
  // Step 4: Compute final gradient
  // dinp = (dxhat - (sum(dxhat) + xhat * sum(dxhat * xhat)) / hidden_dim) * rsqrt(var)
  float4 *inp_grad_f4 = reinterpret_cast<float4 *>(inp_grad) + batch_idx * hidden_dim;
  
  for (uint idx = threadIdx.x; idx < hidden_dim; idx += blockDim.x) {
    float4 out_grad_val = out_grad_f4[idx];
    float4 gamma_val = gamma_f4[idx];
    float4 inp_val = inp_f4[idx];
    
    // Recompute xhat and dxhat (they're cheap to compute)
    float xhat_x = (inp_val.x - mean_val) * inv_var;
    float xhat_y = (inp_val.y - mean_val) * inv_var;
    float xhat_z = (inp_val.z - mean_val) * inv_var;
    float xhat_w = (inp_val.w - mean_val) * inv_var;
    
    float dxhat_x = out_grad_val.x * gamma_val.x;
    float dxhat_y = out_grad_val.y * gamma_val.y;
    float dxhat_z = out_grad_val.z * gamma_val.z;
    float dxhat_w = out_grad_val.w * gamma_val.w;
    
    float4 result;
    result.x = (dxhat_x - (s_sum_dxhat + xhat_x * s_sum_dxhat_xhat)) * inv_var;
    result.y = (dxhat_y - (s_sum_dxhat + xhat_y * s_sum_dxhat_xhat)) * inv_var;
    result.z = (dxhat_z - (s_sum_dxhat + xhat_z * s_sum_dxhat_xhat)) * inv_var;
    result.w = (dxhat_w - (s_sum_dxhat + xhat_w * s_sum_dxhat_xhat)) * inv_var;
    
    inp_grad_f4[idx] = result;
  }
  /// END ASSIGN4_2_2
}
extern "C" {
void launch_layernorm_bw(float *gamma_grad, float *betta_grad, float *inp_grad,
                         const float *out_grad, const float *inp, const float *gamma,
                         const float *betta, const float *vars,
                         const float *means, int batch_size, int hidden_dim,
                         cudaStream_t stream_1, cudaStream_t stream_2) {
  
  // Allocate device memory
  float *d_gamma_grad, *d_betta_grad, *d_inp_grad, *d_out_grad, *d_inp, *d_gamma, *d_betta, *d_vars, *d_means;
  int grad_output_size = batch_size * hidden_dim * sizeof(float);
  int gamma_betta_size = hidden_dim * sizeof(float);
  int vars_means_size = batch_size * sizeof(float);

  cudaMalloc((void **)&d_gamma_grad, gamma_betta_size);
  cudaMalloc((void **)&d_betta_grad, gamma_betta_size);
  cudaMalloc((void **)&d_inp_grad, grad_output_size);
  cudaMalloc((void **)&d_out_grad, grad_output_size);
  cudaMalloc((void **)&d_inp, grad_output_size);
  cudaMalloc((void **)&d_gamma, gamma_betta_size);
  cudaMalloc((void **)&d_betta, gamma_betta_size);
  cudaMalloc((void **)&d_vars, vars_means_size);
  cudaMalloc((void **)&d_means, vars_means_size);

  // Copy memory to device
  cudaMemcpy((void *)d_out_grad, out_grad, grad_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_inp, inp, grad_output_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_gamma, gamma, gamma_betta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_betta, betta, gamma_betta_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_vars, vars, vars_means_size, cudaMemcpyHostToDevice);
  cudaMemcpy((void *)d_means, means, vars_means_size, cudaMemcpyHostToDevice);

  // Launch kernels
  // Compute grad of gamma and betta
  // This calculates the number of blocks needed to cover the data along the specified dimension, rounds it up.
  // The result is then multiplied by TILE_DIM to ensure that the grid size is a multiple of TILE_DIM.
  dim3 grid_dim(((hidden_dim + TILE_DIM - 1) / TILE_DIM) * TILE_DIM);
  dim3 block_dim(TILE_DIM, TILE_DIM);
  ker_ln_bw_dgamma_dbetta<float><<<grid_dim, block_dim, 0, stream_1>>>(
      d_gamma_grad, d_betta_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars,
      d_means, batch_size, hidden_dim);

  // Compute grad of input
  if (hidden_dim % 4 != 0 || hidden_dim > 4096) {
    throw std::runtime_error("hidden_dim % 4 != 0 || hidden_dim > 4096");
  }
  hidden_dim >>= 2;
  int nthread = min(((hidden_dim + 31) / 32) * 32, MAX_THREADS);
  ker_ln_bw_dinp<<<batch_size, nthread, 0, stream_2>>>(
      d_inp_grad, d_out_grad, d_inp, d_gamma, d_betta, d_vars, d_means, hidden_dim);

  // Synchronize and check for errors
  cudaDeviceSynchronize();
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    fprintf(stderr, "launch_layernorm_bw Error: %s\n", cudaGetErrorString(err));
    exit(EXIT_FAILURE);
  }

  // Copy back to host
  cudaMemcpy(gamma_grad, d_gamma_grad, gamma_betta_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(betta_grad, d_betta_grad, gamma_betta_size, cudaMemcpyDeviceToHost);
  cudaMemcpy(inp_grad, d_inp_grad, grad_output_size, cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_gamma_grad);
  cudaFree(d_betta_grad);
  cudaFree(d_inp_grad);
  cudaFree((void *)d_out_grad);
  cudaFree((void *)d_inp);
  cudaFree((void *)d_gamma);
  cudaFree((void *)d_betta);
  cudaFree((void *)d_vars);
  cudaFree((void *)d_means);
}}
}}
