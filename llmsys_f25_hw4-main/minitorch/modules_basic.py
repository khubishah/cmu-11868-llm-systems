"""
For additional transformer related

Sequential
Embedding

"""
import numpy as np

from .module import Module, Parameter
from .tensor_functions import (zeros, ones, rand, tensor, tensor_from_numpy, zeros_tensor_from_numpy, ones_tensor_from_numpy)
from .nn import one_hot
from .tensor_ops import TensorBackend
from .tensor import Tensor

from typing import Any, Dict, Optional, Sequence, Tuple


class Embedding(Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, backend: TensorBackend):
        super().__init__()
        """
        Maps one-hot word vectors from a dictionary of fixed size to embeddings.

        Args:
            num_embeddings : The vocabulary size
            embedding_dim : The size of each embedding vector

        Attributes:
            weight : The learnable weights of shape (num_embeddings, embedding_dim) initialized from N(0, 1).
        """
        self.backend = backend
        self.num_embeddings = num_embeddings # Vocab size
        self.embedding_dim  = embedding_dim  # Embedding Dimension
        ### BEGIN ASSIGN3_2
        # Initialize weight matrix from N(0, 1) distribution
        weight_data = np.random.randn(num_embeddings, embedding_dim).astype(np.float32)
        self.weights = Parameter(tensor_from_numpy(weight_data, backend=backend))
        ### END ASSIGN3_2
    
    def forward(self, x: Tensor):
        """Maps word indices to one-hot vectors, and projects to embedding vectors.

        Args:
            x : Tensor of shape (batch_size, seq_len)

        Returns:
            output : Tensor of shape (batch_size, seq_len, embedding_dim)
        """
        bs, seq_len = x.shape
        ### BEGIN ASSIGN3_2
        # Reshape input to 1D for one_hot conversion
        x_flat = x.view(bs * seq_len)
        
        # Convert indices to one-hot vectors
        # one_hot produces shape (bs * seq_len, num_embeddings)
        one_hot_flat = one_hot(x_flat, self.num_embeddings)
        
        # Project one-hot vectors to embeddings using weight matrix
        # (bs * seq_len, num_embeddings) @ (num_embeddings, embedding_dim) = (bs * seq_len, embedding_dim)
        embeddings_flat = one_hot_flat @ self.weights.value
        
        # Reshape back to (batch_size, seq_len, embedding_dim)
        output = embeddings_flat.view(bs, seq_len, self.embedding_dim)
        
        return output
        ### END ASSIGN3_2

    
class Dropout(Module):
    def __init__(self, p_dropout: float=0.1):
        super().__init__()
        """During training, randomly zeroes some of the elements of the input tensor with probability :attr:`p_dropout`.

        Attributes: 
            p_dropout : Probability an element will be zeroed.
        """
        self.p_dropout = p_dropout

    def forward(self, x: Tensor) -> Tensor: 
        """During training, randomly zero out elements of a tensor and scale by (1 - p_dropout)
        
        Args: 
            x : Tensor of shape (*)
        
        Returns: 
            output : Tensor of shape (*)
        """
        ### BEGIN ASSIGN3_2
        # If not training, return input unchanged
        if not self.training:
            return x
        
        # During training, apply dropout
        if self.p_dropout == 0.0:
            return x
        
        # Use np.random.binomial as specified in README to match autograder seed
        keep_prob = 1.0 - self.p_dropout
        mask = np.random.binomial(1, keep_prob, size=x.shape)
        mask_tensor = tensor_from_numpy(mask.astype(np.float32), backend=x.backend)
        
        # Apply mask and scale by 1/(1-p_dropout) to maintain expected value
        return (x * mask_tensor) / keep_prob
        ### END ASSIGN3_2


class Linear(Module):
    def __init__(self, in_size: int, out_size: int, bias: bool, backend: TensorBackend):
        super().__init__()
        """Applies a linear transformation to the incoming data. (Same as PyTorch)

        Parameters:
            in_size  - The size of the dimension the transformation will be applied to
            out_size - The size of the resulting transformation's dimension
            bias     - If True, then add an additive bias

        Attributes:
            weights - The learnable weights of shape (in_size, out_size) initialized from Uniform(-1/sqrt(in_size), 1/sqrt(in_size)).
            bias   - The learnable weights of shape (out_size, ) initialized from Uniform(-1/sqrt(in_size), 1/sqrt(in_size)).
        """
        self.out_size = out_size
        self.in_size = in_size
        self.use_bias = bias
        ### BEGIN ASSIGN3_2
        # Initialize weights with Uniform(-1/sqrt(in_size), 1/sqrt(in_size))
        bound = 1.0 / (in_size ** 0.5)
        self.weights = Parameter(2 * bound * rand((in_size, out_size), backend=backend) - bound)
        
        if bias:
            # Initialize bias with the same distribution
            self.bias = Parameter(2 * bound * rand((out_size,), backend=backend) - bound)
        else:
            self.bias = None
        ### END ASSIGN3_2

    def forward(self, x: Tensor):
        """Applies a linear transformation to the incoming data.
        
        Args: 
            x : Tensor of shape (n, in_size) or (batch_size, seq_len, in_size)
        
        Returns:
            output : Tensor of shape (n, out_size) or (batch_size, seq_len, out_size)
        """
        ### BEGIN ASSIGN3_2
        # Handle both 2D and 3D inputs (for transformer layers)
        input_shape = x.shape
        
        if len(input_shape) == 2:
            # Standard 2D case: (batch, in_size)
            output = x @ self.weights.value
        elif len(input_shape) == 3:
            # 3D case for transformers: (batch_size, seq_len, in_size)
            batch_size, seq_len, in_size = input_shape
            # Reshape to 2D, apply linear transformation, then reshape back
            x_reshaped = x.view(batch_size * seq_len, in_size)
            output_reshaped = x_reshaped @ self.weights.value
            output = output_reshaped.view(batch_size, seq_len, self.out_size)
        else:
            raise ValueError(f"Linear layer expects 2D or 3D input, got {len(input_shape)}D")
        
        # Add bias if present
        if self.use_bias:
            output = output + self.bias.value
            
        return output
        ### END ASSIGN3_2


class LayerNorm1d(Module):
    def __init__(self, dim: int, eps: float, backend: TensorBackend):
        super().__init__()
        """Applies Layer Normalization over a mini-batch of 1-dimensional inputs.
        
        Args: 
            dim : Expected size of the last dimension to apply layer normalization.
            eps : A value added for numerical stability.
        
        Attributes: 
            weights : the learnable weights of the module of shape (self.dim, ) initialized to 1.
            bias    : the learnable bias of the module of shape (self.dim, ) initialized to 0.
        """
        self.dim = dim
        self.eps = eps
        ### BEGIN ASSIGN3_2
        # Initialize weights to 1 and bias to 0
        self.weights = Parameter(ones((dim,), backend=backend))
        self.bias = Parameter(zeros((dim,), backend=backend))
        ### END ASSIGN3_2

    def forward(self, x: Tensor) -> Tensor:
        """Applies Layer Normalization over a mini-batch of inputs. 
        NOTE: You can assume the input to this layer is a 2D tensor of shape (batch_size, dim)
        You will use implicit broadcasting in miniTorch to use the weight and bias.
        
        Input: 
            x - Tensor of shape (bs, dim)
        
        Output: 
            output - Tensor of shape (bs, dim)
        """
        batch, dim = x.shape
        ### BEGIN ASSIGN3_2
        # Compute mean and variance along the feature dimension (dim=1)
        mean = x.mean(dim=1)  # Shape: (batch, 1)
        
        # Compute variance: E[(x - mean)^2]
        variance = ((x - mean) ** 2).mean(dim=1)  # Shape: (batch, 1)
        
        # Normalize: (x - mean) / sqrt(variance + eps)
        normalized = (x - mean) / ((variance + self.eps) ** 0.5)
        
        # Apply learnable scale and shift using implicit broadcasting
        # weights and bias have shape (dim,), normalized has shape (batch, dim)
        output = normalized * self.weights.value + self.bias.value
        
        return output
        ### END ASSIGN3_2