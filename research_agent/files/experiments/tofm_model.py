"""
TOFM Model Module: Transformer-Based Order Flow Microstructure Model

This module implements the model architecture as specified in framework.md Section 8.2
Key features:
- Input embedding with linear projection
- Positional encoding (sinusoidal)
- Microstructure-informed attention bias
- Multi-head self-attention
- Classification and auxiliary heads
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding as specified in Section 3.2

    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    def __init__(self, d_model: int, max_len: int = 500, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch, seq_len, d_model)
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class MicrostructureAttentionBias(nn.Module):
    """
    Microstructure-Informed Attention Bias (Novel Contribution)

    As specified in Section 3.3:
    B_micro[i,j] = gamma_1 * Corr(OFI_i, OFI_j) + gamma_2 * |lambda_i - lambda_j|

    This biases attention toward:
    1. Time steps with correlated order flow patterns
    2. Similar adverse selection environments
    """
    def __init__(self, d_model: int = 128):
        super().__init__()
        # Learnable gamma parameters
        self.gamma = nn.Parameter(torch.zeros(3))
        nn.init.uniform_(self.gamma, -0.1, 0.1)

    def forward(self, x_raw: torch.Tensor) -> torch.Tensor:
        """
        Compute microstructure attention bias.

        Args:
            x_raw: Raw input features (batch, seq_len, d_input)
                   Expects OFI at index 0, kyle_lambda at index 5

        Returns:
            bias: Attention bias matrix (batch, seq_len, seq_len)
        """
        batch_size, seq_len, _ = x_raw.shape

        # Extract OFI (index 0) and Kyle's lambda (index 5)
        ofi = x_raw[:, :, 0]  # (batch, seq_len)
        kyle_lambda = x_raw[:, :, 5]  # (batch, seq_len)

        # Compute pairwise OFI correlation approximation
        # Use outer product of normalized OFI as correlation proxy
        ofi_norm = (ofi - ofi.mean(dim=1, keepdim=True)) / (ofi.std(dim=1, keepdim=True) + 1e-8)
        corr_proxy = torch.bmm(ofi_norm.unsqueeze(2), ofi_norm.unsqueeze(1))  # (batch, seq, seq)

        # Compute pairwise lambda difference
        lambda_diff = torch.abs(kyle_lambda.unsqueeze(2) - kyle_lambda.unsqueeze(1))  # (batch, seq, seq)

        # Compute volatility regime similarity (using RV at index 8)
        rv = x_raw[:, :, 8]  # (batch, seq_len)
        rv_median = rv.median(dim=1, keepdim=True).values
        regime = (rv > rv_median).float()  # Binary regime indicator
        regime_match = (regime.unsqueeze(2) == regime.unsqueeze(1)).float()

        # Combine with learnable weights
        bias = (self.gamma[0] * corr_proxy +
                self.gamma[1] * (-lambda_diff) +  # Negative so similar values get higher attention
                self.gamma[2] * regime_match)

        return bias


class MultiHeadAttentionWithBias(nn.Module):
    """
    Multi-Head Attention with optional microstructure bias.

    Attention_micro(Q, K, V) = softmax(Q * K^T / sqrt(d_k) + B_micro) * V
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, x: torch.Tensor,
                bias: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor (batch, seq_len, d_model)
            bias: Optional attention bias (batch, seq_len, seq_len)
            return_attention: Whether to return attention weights

        Returns:
            output: Transformed tensor (batch, seq_len, d_model)
            attention: Attention weights if return_attention=True
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        Q = self.W_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale  # (batch, n_heads, seq, seq)

        # Add microstructure bias if provided
        if bias is not None:
            # Expand bias for all heads
            bias = bias.unsqueeze(1)  # (batch, 1, seq, seq)
            scores = scores + bias

        # Apply softmax
        attention = F.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Apply to values
        context = torch.matmul(attention, V)  # (batch, n_heads, seq, d_k)

        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Output projection
        output = self.W_o(context)

        if return_attention:
            return output, attention
        return output, None


class TransformerBlock(nn.Module):
    """
    Transformer Block as specified in Section 3.5

    H' = LayerNorm(H + MultiHead(H))
    H^{l+1} = LayerNorm(H' + FFN(H'))
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()

        self.attention = MultiHeadAttentionWithBias(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                bias: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through transformer block.
        """
        # Self-attention with residual
        attn_out, attention = self.attention(x, bias, return_attention)
        x = self.norm1(x + self.dropout(attn_out))

        # FFN with residual
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x, attention


class TOFM(nn.Module):
    """
    Transformer-Based Order Flow Microstructure Model (TOFM)

    Full model architecture as specified in Section 8.2 of framework.md

    Components:
    - Input embedding layer
    - Positional encoding
    - N transformer blocks with optional microstructure attention bias
    - Classification head (3-class: down, stable, up)
    - Auxiliary head (predict OFI and lambda for regularization)
    """
    def __init__(self,
                 d_input: int,
                 d_model: int = 128,
                 n_heads: int = 8,
                 n_layers: int = 4,
                 d_ff: int = 512,
                 seq_len: int = 100,
                 n_classes: int = 3,
                 dropout: float = 0.1,
                 use_micro_bias: bool = True):
        """
        Args:
            d_input: Input feature dimension (9 + 2*L)
            d_model: Transformer hidden dimension (128)
            n_heads: Number of attention heads (8)
            n_layers: Number of transformer blocks (4)
            d_ff: Feedforward dimension (512)
            seq_len: Sequence length (100)
            n_classes: Number of output classes (3)
            dropout: Dropout rate (0.1)
            use_micro_bias: Whether to use microstructure attention bias
        """
        super().__init__()

        self.d_input = d_input
        self.d_model = d_model
        self.n_layers = n_layers
        self.use_micro_bias = use_micro_bias

        # Input embedding
        self.input_embedding = nn.Sequential(
            nn.Linear(d_input, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout)
        )

        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model, max_len=seq_len + 50, dropout=dropout)

        # Microstructure attention bias
        if use_micro_bias:
            self.micro_bias = MicrostructureAttentionBias(d_model)
        else:
            self.micro_bias = None

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Classification head
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_classes)
        )

        # Auxiliary head (predict OFI and Kyle's lambda)
        self.auxiliary_head = nn.Linear(d_model, 2)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize weights using Xavier/Glorot initialization."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor,
                return_attention: bool = False) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through TOFM.

        Args:
            x: Input tensor (batch, seq_len, d_input)
            return_attention: Whether to return attention weights from last layer

        Returns:
            logits: Classification logits (batch, n_classes)
            aux_pred: Auxiliary predictions for OFI and lambda (batch, 2)
            attention: Attention weights from last layer if return_attention=True
        """
        batch_size, seq_len, _ = x.shape

        # Store raw input for microstructure bias
        x_raw = x

        # Input embedding
        h = self.input_embedding(x)  # (batch, seq_len, d_model)

        # Add positional encoding
        h = self.positional_encoding(h)

        # Compute microstructure bias if enabled
        if self.use_micro_bias and self.micro_bias is not None:
            bias = self.micro_bias(x_raw)
        else:
            bias = None

        # Pass through transformer blocks
        attention = None
        for i, block in enumerate(self.transformer_blocks):
            return_attn = return_attention and (i == len(self.transformer_blocks) - 1)
            h, attention = block(h, bias, return_attention=return_attn)

        # Take last position representation
        z = h[:, -1, :]  # (batch, d_model)

        # Classification head
        logits = self.classification_head(z)  # (batch, n_classes)

        # Auxiliary head
        aux_pred = self.auxiliary_head(z)  # (batch, 2)

        return logits, aux_pred, attention

    def get_representation(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get the final hidden representation before classification.

        Useful for analysis and visualization.
        """
        x_raw = x
        h = self.input_embedding(x)
        h = self.positional_encoding(h)

        if self.use_micro_bias and self.micro_bias is not None:
            bias = self.micro_bias(x_raw)
        else:
            bias = None

        for block in self.transformer_blocks:
            h, _ = block(h, bias)

        return h[:, -1, :]


class LSTMBaseline(nn.Module):
    """
    LSTM baseline model for ablation study comparison.
    """
    def __init__(self, d_input: int, hidden_size: int = 128,
                 n_layers: int = 2, n_classes: int = 3, dropout: float = 0.1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=d_input,
            hidden_size=hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0,
            bidirectional=False
        )

        self.classification_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, n_classes)
        )

        self.auxiliary_head = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor,
                return_attention: bool = False) -> Tuple[torch.Tensor, torch.Tensor, None]:
        # LSTM forward
        output, (h_n, c_n) = self.lstm(x)

        # Take last hidden state
        z = output[:, -1, :]

        logits = self.classification_head(z)
        aux_pred = self.auxiliary_head(z)

        return logits, aux_pred, None


class MLPBaseline(nn.Module):
    """
    MLP baseline model for ablation study comparison.
    Flattens sequence and uses feedforward layers.
    """
    def __init__(self, d_input: int, seq_len: int = 100,
                 hidden_size: int = 256, n_classes: int = 3, dropout: float = 0.1):
        super().__init__()

        flat_size = d_input * seq_len

        self.mlp = nn.Sequential(
            nn.Linear(flat_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, hidden_size // 4),
            nn.ReLU(),
        )

        self.classification_head = nn.Linear(hidden_size // 4, n_classes)
        self.auxiliary_head = nn.Linear(hidden_size // 4, 2)

    def forward(self, x: torch.Tensor,
                return_attention: bool = False) -> Tuple[torch.Tensor, torch.Tensor, None]:
        # Flatten sequence
        batch_size = x.size(0)
        x_flat = x.view(batch_size, -1)

        z = self.mlp(x_flat)

        logits = self.classification_head(z)
        aux_pred = self.auxiliary_head(z)

        return logits, aux_pred, None


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test model creation
    d_input = 29  # 9 + 2*10
    batch_size = 32
    seq_len = 100

    # Create model
    model = TOFM(
        d_input=d_input,
        d_model=128,
        n_heads=8,
        n_layers=4,
        d_ff=512,
        seq_len=seq_len,
        n_classes=3,
        dropout=0.1,
        use_micro_bias=True
    )

    print("TOFM Model Architecture:")
    print(model)
    print(f"\nTotal parameters: {count_parameters(model):,}")

    # Test forward pass
    x = torch.randn(batch_size, seq_len, d_input)
    logits, aux_pred, attention = model(x, return_attention=True)

    print(f"\nInput shape: {x.shape}")
    print(f"Logits shape: {logits.shape}")
    print(f"Aux pred shape: {aux_pred.shape}")
    print(f"Attention shape: {attention.shape}")

    # Test baseline models
    lstm = LSTMBaseline(d_input, hidden_size=128, n_layers=2, n_classes=3)
    mlp = MLPBaseline(d_input, seq_len=seq_len, hidden_size=256, n_classes=3)

    print(f"\nLSTM parameters: {count_parameters(lstm):,}")
    print(f"MLP parameters: {count_parameters(mlp):,}")
