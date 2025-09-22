import numpy as np
from .tensor import tensor, tensor_from_numpy
from .module import Module, Parameter
from .modules_basic import (
    Embedding,
    Dropout,
    LayerNorm1d,
    Linear
)
from .tensor_ops import TensorBackend
from .nn import (
    max,
    softmax,
    dropout,
    GELU,
)
from typing import Any, Dict, Optional, Sequence, Tuple

datatype = np.float32


class MultiHeadAttention(Module):
    def __init__(self, n_embd: int, n_head: int, causal: bool=True, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        """Implements Multi-Head Attention as described in "Attention Is All You Need"

        Args:
            n_embd: Dimensionality of embeddings and hidden states
            n_head: Number of heads
            p_dropout: Dropout ratio for dropout layer
            causal: If True, then apply a causal mask during self-attention
            bias: If True, then apply a bias in Linear layers
        
        Attributes:
            q_projection: Linear layer projecting input to Q matrix
            k_projection: Linear layer projecting input to K matrix
            v_projection: Linear layer projecting input to V matrix
            out_projection: Linear output projection layer
            dropout: Dropout layer
        """
        self.backend = backend
        self.n_embd = n_embd 
        self.n_head = n_head
        self.causal = causal
        self.attn_hidden_dim = n_embd // n_head

        ### BEGIN ASSIGN3_3
        # Ensure n_embd is divisible by n_head for proper head splitting
        assert n_embd % n_head == 0, f"n_embd ({n_embd}) must be divisible by n_head ({n_head})"
        
        # Create linear projection layers for Q, K, V (each projects n_embd -> n_embd)
        self.q_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.k_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        self.v_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        
        # Output projection layer to combine heads
        self.out_projection = Linear(n_embd, n_embd, bias=bias, backend=backend)
        
        # Dropout layer
        self.dropout = Dropout(p_dropout)
        ### END ASSIGN3_3


    def create_causal_mask(self, seq_len):
        """
        Create a causal mask for self-attention to prevent information leakage.
        
        Generates a triangular mask where each position can only attend to previous
        positions and itself. Upper triangle contains -inf, lower triangle contains 0.

        Args:
            seq_len (int): Length of the sequence

        Returns:
            Tensor: Causal mask of shape (1, 1, seq_len, seq_len) with -inf above
                    diagonal and 0 on/below diagonal. Will be broadcasted to full
                    attention tensor shape during computation.
        """
        # Returns a 1x1xTxt triangular causal mask for Q @ K^T (You will implicitly broadcast it to BxHxTxT)
        # Use -1e9 instead of -np.finfo(datatype).max for numerical stability
        mask = -1e9 * np.triu(np.ones((1, 1, seq_len, seq_len), dtype=datatype), 1)
        # CRITICAL: Mask should not participate in gradient computation
        return tensor_from_numpy(mask, backend=self.backend, requires_grad=False)

    def project_to_query_key_value(self, x):
        """
        Project input embeddings to Query, Key, and Value matrices for self-attention.
        
        Args:
            x (Tensor): Input embeddings of shape (batch_size, seq_len, n_embd)

        Returns:
            tuple: (q, kT, v) where:
                - q: Query matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)
                - kT: Transposed key matrix of shape (batch_size, num_heads, attn_hidden_dim, seq_len)
                - v: Value matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)
        """
        batch_size, seq_len, n_embd = x.shape
        ### BEGIN ASSIGN3_3
        # Step 1: Project input to Q, K, V using linear layers (README: X ∈ R^(B×S×D) → Q,K,V ∈ R^(B×S×D))
        q = self.q_projection(x)  # (batch_size, seq_len, n_embd)
        k = self.k_projection(x)  # (batch_size, seq_len, n_embd) 
        v = self.v_projection(x)  # (batch_size, seq_len, n_embd)
        
        # Step 2: Reshape for multi-head attention (README: unravel to Q ∈ R^(B×S×h×D_h))
        # Split n_embd into n_head * attn_hidden_dim
        q = q.view(batch_size, seq_len, self.n_head, self.attn_hidden_dim)  # (B, S, h, D_h)
        k = k.view(batch_size, seq_len, self.n_head, self.attn_hidden_dim)  # (B, S, h, D_h)
        v = v.view(batch_size, seq_len, self.n_head, self.attn_hidden_dim)  # (B, S, h, D_h)
        
        # Step 3: Permute to get (README: Q ∈ R^(B×S×h×D_h) → Q ∈ R^(B×h×S×D_h))
        q = q.permute(0, 2, 1, 3)  # (batch_size, n_head, seq_len, attn_hidden_dim)
        k = k.permute(0, 2, 1, 3)  # (batch_size, n_head, seq_len, attn_hidden_dim)
        v = v.permute(0, 2, 1, 3)  # (batch_size, n_head, seq_len, attn_hidden_dim)
        
        # Step 4: Transpose K for attention computation (README: take care to transpose K along last two dimensions)
        kT = k.permute(0, 1, 3, 2)  # (batch_size, n_head, attn_hidden_dim, seq_len)
        ### END ASSIGN3_3
        return q, kT, v
    
    def self_attention(self, q, kT, v):
        """
        Compute self-attention: softmax((q @ kT) / sqrt(attn_hidden_dim)) @ v.
        
        Args:
            q (Tensor): Query matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)
            kT (Tensor): Transposed key matrix of shape (batch_size, num_heads, attn_hidden_dim, seq_len)
            v (Tensor): Value matrix of shape (batch_size, num_heads, seq_len, attn_hidden_dim)

        Returns:
            Tensor: Attention output of shape (batch_size, seq_len, n_embd)
        """
        batch_size, num_head, queries_len, q_dim = q.shape
        _, _, k_dim, _ = kT.shape
        _, _, _, v_dim = v.shape
        assert q_dim == k_dim == v_dim
        result = None
        
        ### BEGIN ASSIGN3_3
        # Follow README formula exactly: softmax((Q_i K_i^T)/√D_h + M) V_i
        
        # Step 1: Compute Q_i K_i^T for each head i (README formula)
        # q: (batch_size, num_heads, seq_len, attn_hidden_dim)
        # kT: (batch_size, num_heads, attn_hidden_dim, seq_len)
        # Result: (batch_size, num_heads, seq_len, seq_len)
        attention_scores = q @ kT
        
        # Step 2: Scale by 1/√D_h as per README formula (√D_h where D_h = attn_hidden_dim)
        scale_factor = (q_dim ** 0.5)  # q_dim = attn_hidden_dim 
        attention_scores = attention_scores / scale_factor
        
        # Step 3: Add causal mask M if enabled (README: + M)
        if self.causal:
            mask = self.create_causal_mask(queries_len)  # Shape: (1, 1, seq_len, seq_len)
            attention_scores = attention_scores + mask  # Broadcasting: (B,H,S,S) + (1,1,S,S)
        
        # Step 4: Apply softmax (README: softmax(...))
        attention_probs = softmax(attention_scores, dim=3)  # Softmax over last dimension (seq_len)
        
        # Step 5: Multiply by V_i (README: ... V_i)
        # attention_probs: (batch_size, num_heads, seq_len, seq_len) 
        # v: (batch_size, num_heads, seq_len, attn_hidden_dim)
        # Result: (batch_size, num_heads, seq_len, attn_hidden_dim)
        attn_output = attention_probs @ v
        
        # Step 6: Permute and reshape to combine heads (README: A ∈ R^(B×h×S×D_h) → A ∈ R^(B×S×h×D_h) → A ∈ R^(B×S×D))
        # Before: A ∈ R^(B×h×S×D_h)
        # After permute: A ∈ R^(B×S×h×D_h) 
        attn_output = attn_output.permute(0, 2, 1, 3).contiguous()  # (batch_size, seq_len, num_heads, attn_hidden_dim)
        # After reshape: A ∈ R^(B×S×D) where D = h × D_h = n_embd
        result = attn_output.view(batch_size, queries_len, self.n_embd)  # (batch_size, seq_len, n_embd)
        ### END ASSIGN3_3

        return result

    def forward(self, x):
        """
        Compute multi-head attention with optional causal masking.
        HEAD-BY-HEAD implementation to avoid 4D tensor autodiff issues.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape
        ### BEGIN ASSIGN3_3
        # Follow README structure exactly
        
        # Step 1: Project X into Q, K^T, V in the project_to_query_key_value function
        q, kT, v = self.project_to_query_key_value(x)
        
        # Step 2: Compute self-attention 
        attn_output = self.self_attention(q, kT, v)
        
        # Step 3: Pass self-attention output through the out projection layer
        output = self.out_projection(attn_output)
        
        return output
        ### END ASSIGN3_3


class FeedForward(Module):
    def __init__(self, n_embd: int, middle_dim: int=256, p_dropout: float=0.1, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        """
        Initialize a feed-forward network module.
        
        Args:
            n_embd (int): Input and output dimension
            middle_dim (int): Hidden layer dimension, default 256
            p_dropout (float): Dropout probability, default 0.1
            bias (bool): Whether to use bias in linear layers, default True
            backend (TensorBackend): Backend for tensor operations
            
        Attributes:
            linear_in (Linear): First linear layer
            linear_out (Linear): Second linear layer
            dropout (Dropout): Dropout layer
        """
        ### BEGIN ASSIGN3_3
        self.linear_in  = Linear(n_embd, middle_dim, bias=bias, backend=backend)
        self.linear_out = Linear(middle_dim, n_embd, bias=bias, backend=backend)
        self.dropout    = Dropout(p_dropout)
        ### END ASSIGN3_3

    def forward(self, x):
        """
        Forward pass through feed-forward network with  activation and dropout.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, n_embd)

        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape

        ### BEGIN ASSIGN3_3
        x = GELU(self.linear_in(x.view(batch_size * seq_len, n_embd)))
        x = self.dropout(self.linear_out(x)).view(batch_size, seq_len, n_embd)
        ### END ASSIGN3_3

        return x
    

class TransformerLayer(Module):
    def __init__(self, n_embd: int, n_head: int, p_dropout: float=0.1, ln_eps: float=1e-5, bias: bool=True, backend: TensorBackend=None):
        super().__init__()
        """
        Initialize a transformer layer with pre-layer normalization.
        
        Args:
            n_embd (int): Embedding dimension
            n_head (int): Number of attention heads
            p_dropout (float): Dropout probability, default 0.1
            ln_eps (float): Layer normalization epsilon, default 1e-5
            bias (bool): Whether to use bias in linear layers, default True
            backend (TensorBackend): Backend for tensor operations
            
        Attributes:
            ln_1 (LayerNorm1d): First layer normalization before attention
            ln_2 (LayerNorm1d): Second layer normalization after attention
            attention (MultiHeadAttention): Multi-head attention layer
            ff (FeedForward): Feed-forward network layer
        """
        ### BEGIN ASSIGN3_3
        # Pre-LN architecture components
        self.ln_1 = LayerNorm1d(n_embd, ln_eps, backend)
        self.ln_2 = LayerNorm1d(n_embd, ln_eps, backend)
        
        # Multi-head attention layer
        self.attention = MultiHeadAttention(n_embd, n_head, causal=True, p_dropout=p_dropout, bias=bias, backend=backend)
        
        # Feed-forward network layer (README mentions 16*n_embd for GPT-2 configuration)
        self.ff = FeedForward(n_embd, middle_dim=16*n_embd, p_dropout=p_dropout, bias=bias, backend=backend)
        ### END ASSIGN3_3

    def forward(self, x):
        """
        Forward pass through transformer layer with pre-layer normalization.
        
        Args:
            x (Tensor): Input tensor of shape (batch_size, seq_len, n_embd)
        
        Returns:
            Tensor: Output tensor of shape (batch_size, seq_len, n_embd)
        """
        batch_size, seq_len, n_embd = x.shape
        ### BEGIN YOUR SOLUTION
        # Pre-LN Architecture: LayerNorm -> Attention -> Add -> LayerNorm -> FeedForward -> Add
        
        # 1. First residual connection: x + attention(ln_1(x))
        # Apply layer norm before attention (reshape for LayerNorm1d: 3D -> 2D -> 3D)
        x_reshaped = x.view(batch_size * seq_len, n_embd)  # (B*S, D)
        ln1_output = self.ln_1(x_reshaped)  # (B*S, D)
        ln1_output = ln1_output.view(batch_size, seq_len, n_embd)  # (B, S, D)
        
        # Apply attention
        attn_output = self.attention(ln1_output) 
        # Add residual connection
        x = x + attn_output
        
        # 2. Second residual connection: x + feedforward(ln_2(x))
        # Apply layer norm before feedforward (reshape for LayerNorm1d: 3D -> 2D -> 3D)
        x_reshaped = x.view(batch_size * seq_len, n_embd)  # (B*S, D)
        ln2_output = self.ln_2(x_reshaped)  # (B*S, D)
        ln2_output = ln2_output.view(batch_size, seq_len, n_embd)  # (B, S, D)
        
        # Apply feedforward
        ff_output = self.ff(ln2_output)
        # Add residual connection
        x = x + ff_output
        
        return x
        ### END YOUR SOLUTION


class DecoderLM(Module):
    def __init__(
        self, 
        n_vocab: int,
        n_embd: int,
        n_head: int,
        n_positions: int,
        p_dropout: float=0.1,
        ln_eps: float=1e-5, 
        bias: bool=True,
        backend: TensorBackend=None
    ):
        super().__init__()
        """
        Initialize a decoder-only transformer language model.
        
        Args:
            n_vocab (int): Vocabulary size
            n_embd (int): Embedding dimension
            n_head (int): Number of attention heads
            n_positions (int): Maximum sequence length
            p_dropout (float): Dropout probability, default 0.1
            ln_eps (float): Layer normalization epsilon, default 1e-5
            bias (bool): Whether to use bias in linear layers, default True
            backend (TensorBackend): Backend for tensor operations
            
        Attributes:
            token_embeddings (Embedding): Token embedding layer
            position_embeddings (Embedding): Position embedding layer
            t_layer_1 (TransformerLayer): First transformer layer
            t_layer_2 (TransformerLayer): Second transformer layer
            t_layer_3 (TransformerLayer): Third transformer layer
            t_layer_4 (TransformerLayer): Fourth transformer layer
            dropout (Dropout): Dropout layer before transformer layers
            ln (LayerNorm1d): Final layer normalization
            lm_head (Linear): Language model head for vocabulary projection
        """
        self.backend = backend
        self.n_embd = n_embd
        self.n_vocab = n_vocab
        ### BEGIN ASSIGN3_3
        # Embedding layers
        self.token_embeddings = Embedding(n_vocab, n_embd, backend)
        self.position_embeddings = Embedding(n_positions, n_embd, backend)
        
        # Four transformer layers (GPT-2 architecture)
        self.t_layer_1 = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend)
        self.t_layer_2 = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend)
        self.t_layer_3 = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend)
        self.t_layer_4 = TransformerLayer(n_embd, n_head, p_dropout, ln_eps, bias, backend)
        
        # Dropout layer before transformer layers
        self.dropout = Dropout(p_dropout)
        
        # Final layer normalization
        self.ln = LayerNorm1d(n_embd, ln_eps, backend)
        
        # Language model head for vocabulary projection
        self.lm_head = Linear(n_embd, n_vocab, bias, backend)
        ### END ASSIGN3_3
    
    def forward(self, idx):
        """
        Forward pass through decoder-only transformer language model.
        
        Args:
            idx (Tensor): Input token indices of shape (batch_size, seq_len)
        
        Returns:
            Tensor: Logits of shape (batch_size, seq_len, n_vocab)
        """
        
        batch_size, seq_len = idx.shape

        ### BEGIN ASSIGN3_3
        # 1. Get token embeddings of shape (batch_size, seq_len, n_embd)
        token_emb = self.token_embeddings(idx)  # (batch_size, seq_len, n_embd)
        
        # 2. Create positional embeddings of shape (1, seq_len, n_embd):
        #    - Create position ids tensor [0, 1, 2, ..., seq_len-1] of shape (1, seq_len)
        pos_ids = tensor([[i for i in range(seq_len)]], backend=self.backend)  # (1, seq_len)
        #    - Pass through positional embedding layer
        pos_emb = self.position_embeddings(pos_ids)  # (1, seq_len, n_embd)
        
        # 3. Add token and positional embeddings (broadcasting handles (B,S,D) + (1,S,D))
        x = token_emb + pos_emb  # (batch_size, seq_len, n_embd)
        
        # 4. Apply dropout
        x = self.dropout(x)
        
        # 5. Pass through transformer layers (t_layer_1 to t_layer_4)
        x = self.t_layer_1(x)
        x = self.t_layer_2(x)
        x = self.t_layer_3(x)
        x = self.t_layer_4(x)
        
        # 6. Apply final layer normalization (reshape for LayerNorm1d: 3D -> 2D -> 3D)
        x_reshaped = x.view(batch_size * seq_len, self.n_embd)  # (B*S, D)
        x_ln = self.ln(x_reshaped)  # (B*S, D)
        x = x_ln.view(batch_size, seq_len, self.n_embd)  # (B, S, D)
        
        # 7. Project to vocabulary size using lm_head
        logits = self.lm_head(x)  # (batch_size, seq_len, n_vocab)
        
        return logits
        ### END ASSIGN3_3