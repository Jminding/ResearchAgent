"""
Graph Neural Network implementations for homophily vs heterophily comparison.
All implementations are in pure PyTorch (no PyTorch Geometric dependency).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple
import numpy as np


def normalize_adj(edge_index: Tensor, num_nodes: int, add_self_loops: bool = True) -> Tuple[Tensor, Tensor]:
    """
    Compute normalized adjacency matrix weights.
    Returns edge_index and corresponding edge weights for D^{-1/2} A D^{-1/2}.
    """
    if add_self_loops:
        # Add self-loops
        self_loop_edge = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)]).to(edge_index.device)
        edge_index = torch.cat([edge_index, self_loop_edge], dim=1)

    row, col = edge_index
    deg = torch.zeros(num_nodes, dtype=torch.float, device=edge_index.device)
    deg.scatter_add_(0, row, torch.ones(row.size(0), device=edge_index.device))

    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

    edge_weight = deg_inv_sqrt[row] * deg_inv_sqrt[col]

    return edge_index, edge_weight


def sparse_mm(edge_index: Tensor, edge_weight: Tensor, x: Tensor, num_nodes: int) -> Tensor:
    """Sparse matrix multiplication: (adj * x)."""
    row, col = edge_index
    out = torch.zeros(num_nodes, x.size(1), dtype=x.dtype, device=x.device)
    message = x[col] * edge_weight.unsqueeze(1)
    out.scatter_add_(0, row.unsqueeze(1).expand(-1, x.size(1)), message)
    return out


class GCNConv(nn.Module):
    """Graph Convolutional Network layer (Kipf & Welling, 2017)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, num_nodes: int) -> Tensor:
        x = self.linear(x)
        return sparse_mm(edge_index, edge_weight, x, num_nodes)


class SAGEConv(nn.Module):
    """GraphSAGE layer (Hamilton et al., 2017)."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.linear_self = nn.Linear(in_channels, out_channels)
        self.linear_neigh = nn.Linear(in_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear_self.weight)
        nn.init.xavier_uniform_(self.linear_neigh.weight)
        nn.init.zeros_(self.linear_self.bias)
        nn.init.zeros_(self.linear_neigh.bias)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, num_nodes: int) -> Tensor:
        # Mean aggregation of neighbors
        neigh_agg = sparse_mm(edge_index, edge_weight, x, num_nodes)
        return self.linear_self(x) + self.linear_neigh(neigh_agg)


class GATConv(nn.Module):
    """Graph Attention Network layer (Velickovic et al., 2018)."""

    def __init__(self, in_channels: int, out_channels: int, heads: int = 4, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.out_channels = out_channels
        self.head_dim = out_channels // heads

        self.linear = nn.Linear(in_channels, heads * self.head_dim, bias=False)
        self.att_src = nn.Parameter(torch.Tensor(heads, self.head_dim))
        self.att_dst = nn.Parameter(torch.Tensor(heads, self.head_dim))
        self.dropout = dropout

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.xavier_uniform_(self.att_src.unsqueeze(0))
        nn.init.xavier_uniform_(self.att_dst.unsqueeze(0))

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, num_nodes: int) -> Tensor:
        H, D = self.heads, self.head_dim

        x = self.linear(x).view(-1, H, D)  # [N, H, D]

        row, col = edge_index

        # Compute attention scores
        alpha_src = (x * self.att_src).sum(dim=-1)  # [N, H]
        alpha_dst = (x * self.att_dst).sum(dim=-1)  # [N, H]

        alpha = alpha_src[row] + alpha_dst[col]  # [E, H]
        alpha = F.leaky_relu(alpha, negative_slope=0.2)

        # Softmax over neighbors
        alpha_max = torch.zeros(num_nodes, H, device=x.device)
        alpha_max.scatter_reduce_(0, row.unsqueeze(1).expand(-1, H), alpha, reduce='amax', include_self=False)
        alpha = alpha - alpha_max[row]
        alpha = alpha.exp()

        alpha_sum = torch.zeros(num_nodes, H, device=x.device)
        alpha_sum.scatter_add_(0, row.unsqueeze(1).expand(-1, H), alpha)
        alpha = alpha / (alpha_sum[row] + 1e-16)

        if self.training and self.dropout > 0:
            alpha = F.dropout(alpha, p=self.dropout, training=True)

        # Aggregate
        out = torch.zeros(num_nodes, H, D, device=x.device)
        message = x[col] * alpha.unsqueeze(-1)
        out.scatter_add_(0, row.unsqueeze(1).unsqueeze(2).expand(-1, H, D), message)

        return out.view(num_nodes, -1)


# ============== HETEROPHILY-AWARE GNN LAYERS ==============

class H2GCNConv(nn.Module):
    """
    H2GCN layer (Zhu et al., 2020): "Beyond Homophily in Graph Neural Networks".
    Key idea: Separate ego, neighbor, and higher-order neighbor representations.
    """

    def __init__(self, in_channels: int, out_channels: int, k_hops: int = 2):
        super().__init__()
        self.k_hops = k_hops
        # Separate projections for each hop
        self.linears = nn.ModuleList([
            nn.Linear(in_channels, out_channels) for _ in range(k_hops + 1)
        ])
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.linears:
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, num_nodes: int) -> Tensor:
        outputs = [self.linears[0](x)]  # Ego embedding

        current = x
        for k in range(1, self.k_hops + 1):
            # k-hop neighbors via repeated message passing
            current = sparse_mm(edge_index, edge_weight, current, num_nodes)
            outputs.append(self.linears[k](current))

        # Concatenate and combine (could also sum/mean)
        return torch.cat(outputs, dim=1)


class FAGCNConv(nn.Module):
    """
    FAGCN layer (Bo et al., 2021): "Beyond Low-frequency Information in GCNs".
    Key idea: Learn signed attention to capture both low and high frequency signals.
    """

    def __init__(self, in_channels: int, out_channels: int, eps: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(in_channels, out_channels)
        self.att = nn.Parameter(torch.Tensor(out_channels, 1))
        self.eps = eps  # Self-loop weight
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        nn.init.xavier_uniform_(self.att)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, num_nodes: int) -> Tensor:
        x = self.linear(x)

        row, col = edge_index

        # Compute signed attention: tanh allows negative weights
        alpha = torch.tanh((x[row] * x[col]).sum(dim=1))  # [-1, 1]

        # Aggregate with signed attention
        out = torch.zeros_like(x)
        message = x[col] * alpha.unsqueeze(1)
        out.scatter_add_(0, row.unsqueeze(1).expand_as(message), message)

        # Normalize by degree
        deg = torch.zeros(num_nodes, device=x.device)
        deg.scatter_add_(0, row, torch.ones_like(row, dtype=torch.float))
        deg = deg.clamp(min=1)

        out = out / deg.unsqueeze(1)

        # Add self-loop with learnable weight
        return self.eps * x + (1 - self.eps) * out


class GPRGNNConv(nn.Module):
    """
    GPR-GNN layer (Chien et al., 2021): "Adaptive Universal Generalized PageRank GNN".
    Key idea: Learn polynomial weights over propagation steps.
    """

    def __init__(self, in_channels: int, out_channels: int, K: int = 10, alpha: float = 0.1):
        super().__init__()
        self.K = K
        self.linear = nn.Linear(in_channels, out_channels)
        # Learnable coefficients for each propagation step
        self.gamma = nn.Parameter(torch.Tensor(K + 1))
        self.alpha = alpha
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)
        # Initialize with APPNP-like coefficients
        nn.init.constant_(self.gamma, 1.0 / (self.K + 1))

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, num_nodes: int) -> Tensor:
        x = self.linear(x)

        # Generalized PageRank propagation
        gamma = F.softmax(self.gamma, dim=0)

        h = x * gamma[0]
        current = x

        for k in range(1, self.K + 1):
            current = sparse_mm(edge_index, edge_weight, current, num_nodes)
            h = h + gamma[k] * current

        return h


class LINKXConv(nn.Module):
    """
    LINKX layer (Lim et al., 2021): "Large Scale Learning on Non-Homophilous Graphs".
    Key idea: Separate MLP for features and adjacency, then combine.
    """

    def __init__(self, in_channels: int, out_channels: int, hidden_channels: int = 64):
        super().__init__()
        self.mlp_x = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )
        self.mlp_a = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels)
        )
        self.combine = nn.Linear(2 * out_channels, out_channels)
        self.reset_parameters()

    def reset_parameters(self):
        for module in [self.mlp_x, self.mlp_a, self.combine]:
            if isinstance(module, nn.Sequential):
                for m in module:
                    if isinstance(m, nn.Linear):
                        nn.init.xavier_uniform_(m.weight)
                        nn.init.zeros_(m.bias)
            else:
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, num_nodes: int) -> Tensor:
        # Feature branch
        h_x = self.mlp_x(x)

        # Adjacency branch: aggregate then MLP
        agg = sparse_mm(edge_index, edge_weight, x, num_nodes)
        h_a = self.mlp_a(agg)

        # Combine
        return self.combine(torch.cat([h_x, h_a], dim=1))


# ============== FULL GNN MODELS ==============

class GCN(nn.Module):
    """Standard GCN model."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
        if num_layers > 1:
            self.convs.append(GCNConv(hidden_channels, out_channels))
        else:
            self.convs[0] = GCNConv(in_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor, num_nodes: int) -> Tensor:
        edge_index_norm, edge_weight = normalize_adj(edge_index, num_nodes)

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index_norm, edge_weight, num_nodes)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index_norm, edge_weight, num_nodes)
        return x


class GraphSAGE(nn.Module):
    """GraphSAGE model."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        if num_layers > 1:
            self.convs.append(SAGEConv(hidden_channels, out_channels))
        else:
            self.convs[0] = SAGEConv(in_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor, num_nodes: int) -> Tensor:
        edge_index_norm, edge_weight = normalize_adj(edge_index, num_nodes, add_self_loops=False)

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index_norm, edge_weight, num_nodes)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index_norm, edge_weight, num_nodes)
        return x


class GAT(nn.Module):
    """Graph Attention Network model."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, heads: int = 4, dropout: float = 0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        for _ in range(num_layers - 2):
            self.convs.append(GATConv(hidden_channels, hidden_channels, heads=heads, dropout=dropout))
        if num_layers > 1:
            self.convs.append(GATConv(hidden_channels, out_channels, heads=1, dropout=dropout))
        else:
            self.convs[0] = GATConv(in_channels, out_channels, heads=1, dropout=dropout)

    def forward(self, x: Tensor, edge_index: Tensor, num_nodes: int) -> Tensor:
        edge_index_norm, edge_weight = normalize_adj(edge_index, num_nodes)

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index_norm, edge_weight, num_nodes)
            x = F.elu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index_norm, edge_weight, num_nodes)
        return x


class H2GCN(nn.Module):
    """H2GCN model for heterophilic graphs."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, k_hops: int = 2, dropout: float = 0.5):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        # H2GCN concatenates k+1 embeddings per layer
        h2_out_dim = hidden_channels * (k_hops + 1)

        self.convs = nn.ModuleList()
        self.convs.append(H2GCNConv(in_channels, hidden_channels, k_hops=k_hops))
        for _ in range(num_layers - 2):
            self.convs.append(H2GCNConv(h2_out_dim, hidden_channels, k_hops=k_hops))

        if num_layers > 1:
            self.classifier = nn.Linear(h2_out_dim, out_channels)
        else:
            self.classifier = nn.Linear(h2_out_dim, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor, num_nodes: int) -> Tensor:
        edge_index_norm, edge_weight = normalize_adj(edge_index, num_nodes)

        for conv in self.convs:
            x = conv(x, edge_index_norm, edge_weight, num_nodes)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.classifier(x)


class FAGCN(nn.Module):
    """FAGCN model with signed attention."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, dropout: float = 0.5, eps: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(FAGCNConv(in_channels, hidden_channels, eps=eps))
        for _ in range(num_layers - 2):
            self.convs.append(FAGCNConv(hidden_channels, hidden_channels, eps=eps))
        if num_layers > 1:
            self.convs.append(FAGCNConv(hidden_channels, out_channels, eps=eps))
        else:
            self.convs[0] = FAGCNConv(in_channels, out_channels, eps=eps)

    def forward(self, x: Tensor, edge_index: Tensor, num_nodes: int) -> Tensor:
        edge_index_norm, edge_weight = normalize_adj(edge_index, num_nodes)

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index_norm, edge_weight, num_nodes)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index_norm, edge_weight, num_nodes)
        return x


class GPRGNN(nn.Module):
    """GPR-GNN model with learnable polynomial coefficients."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, K: int = 10, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout

        # GPR-GNN typically uses a single layer with K propagation steps
        self.encoder = nn.Linear(in_channels, hidden_channels)
        self.gpr = GPRGNNConv(hidden_channels, out_channels, K=K)

    def forward(self, x: Tensor, edge_index: Tensor, num_nodes: int) -> Tensor:
        edge_index_norm, edge_weight = normalize_adj(edge_index, num_nodes)

        x = self.encoder(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)

        return self.gpr(x, edge_index_norm, edge_weight, num_nodes)


class LINKX(nn.Module):
    """LINKX model separating feature and structure."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(LINKXConv(in_channels, hidden_channels, hidden_channels=hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(LINKXConv(hidden_channels, hidden_channels, hidden_channels=hidden_channels))
        if num_layers > 1:
            self.classifier = nn.Linear(hidden_channels, out_channels)
        else:
            self.classifier = nn.Linear(hidden_channels, out_channels)

    def forward(self, x: Tensor, edge_index: Tensor, num_nodes: int) -> Tensor:
        edge_index_norm, edge_weight = normalize_adj(edge_index, num_nodes)

        for conv in self.convs:
            x = conv(x, edge_index_norm, edge_weight, num_nodes)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        return self.classifier(x)


class MLP(nn.Module):
    """Baseline MLP (no graph structure)."""

    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int,
                 num_layers: int = 2, dropout: float = 0.5):
        super().__init__()
        self.dropout = dropout

        layers = []
        layers.append(nn.Linear(in_channels, hidden_channels))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))

        for _ in range(num_layers - 2):
            layers.append(nn.Linear(hidden_channels, hidden_channels))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))

        layers.append(nn.Linear(hidden_channels, out_channels))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor, edge_index: Tensor = None, num_nodes: int = None) -> Tensor:
        return self.net(x)


def get_model(name: str, in_channels: int, hidden_channels: int, out_channels: int,
              num_layers: int = 2, dropout: float = 0.5, **kwargs) -> nn.Module:
    """Factory function to get model by name."""
    models = {
        'GCN': GCN,
        'GraphSAGE': GraphSAGE,
        'GAT': GAT,
        'H2GCN': H2GCN,
        'FAGCN': FAGCN,
        'GPR-GNN': GPRGNN,
        'LINKX': LINKX,
        'MLP': MLP
    }

    if name not in models:
        raise ValueError(f"Unknown model: {name}. Available: {list(models.keys())}")

    return models[name](in_channels, hidden_channels, out_channels, num_layers=num_layers, dropout=dropout, **kwargs)


# Model categorization
HOMOPHILY_ASSUMING = ['GCN', 'GraphSAGE', 'GAT']
HETEROPHILY_AWARE = ['H2GCN', 'FAGCN', 'GPR-GNN', 'LINKX']
