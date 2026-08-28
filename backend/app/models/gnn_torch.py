"""GraphSAGE, GAT and the graph-free MLP ablation, in PyTorch.

This module is TRAINING-ONLY. PyTorch is not installed in the deployed image;
serving runs the identical forward pass in NumPy (`gnn_numpy.py`) over weights
exported by `export_npz`. `tests/test_model_parity.py` asserts the two agree.

Architecture, exactly as specified in the project documentation:

    node features ─┐
                   ├─> [encoder layer 1] ─> [encoder layer 2] ─> node embeddings
    graph edges ───┘                                                   │
                                                                       ▼
    edge (u,v) prediction = MLP( [ emb_u ‖ emb_v ‖ edge_features ‖ time_context ] )
                                                                       │
                                                                       ▼
                                                    predicted travel time (minutes)

Trained with Huber loss on log(1 + travel_time). Working in log space stops
40-minute bus rides from drowning out 4-minute walks; Huber keeps a single
GPS glitch from dominating the gradient.

Swapping GraphSAGE for MLPEncoder removes message passing and changes nothing
else -- same features, same capacity, same optimiser. That is baseline 4, the
ablation that decides whether this project has a finding.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..graph.features import (
    EDGE_FEATURE_DIM, NODE_FEATURE_DIM, TIME_FEATURE_DIM, feature_signature,
)

EncoderName = Literal["graphsage", "gat", "mlp"]


def scatter_mean(src_values: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Mean of `src_values` grouped by `index`, over `n` groups."""
    out = torch.zeros(n, src_values.size(-1), dtype=src_values.dtype,
                      device=src_values.device)
    out.index_add_(0, index, src_values)
    cnt = torch.zeros(n, 1, dtype=src_values.dtype, device=src_values.device)
    cnt.index_add_(0, index, torch.ones(index.size(0), 1, dtype=src_values.dtype,
                                        device=src_values.device))
    return out / cnt.clamp(min=1.0)


def scatter_softmax(logits: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    """Softmax over the edges sharing each destination node."""
    max_per = torch.full((n, logits.size(-1)), -1e30, dtype=logits.dtype,
                         device=logits.device)
    max_per = max_per.index_reduce(0, index, logits, "amax", include_self=True)
    ex = torch.exp(logits - max_per[index])
    denom = torch.zeros(n, logits.size(-1), dtype=logits.dtype, device=logits.device)
    denom.index_add_(0, index, ex)
    return ex / denom[index].clamp(min=1e-16)


# --------------------------------------------------------------------------
# encoders
# --------------------------------------------------------------------------
class SAGELayer(nn.Module):
    """GraphSAGE with a mean aggregator: combine what you are with the mean of
    what your neighbours are."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.lin_self = nn.Linear(dim_in, dim_out, bias=True)
        self.lin_neigh = nn.Linear(dim_in, dim_out, bias=False)

    def forward(self, h, src, dst, n):
        return self.lin_self(h) + self.lin_neigh(scatter_mean(h[src], dst, n))


class GATLayer(nn.Module):
    """Graph attention: learn how much to listen to each neighbour."""

    def __init__(self, dim_in: int, dim_out: int, heads: int = 2, slope: float = 0.2):
        super().__init__()
        self.heads, self.dim_out, self.slope = heads, dim_out, slope
        self.lin = nn.Linear(dim_in, heads * dim_out, bias=False)
        self.att_src = nn.Parameter(torch.empty(heads, dim_out))
        self.att_dst = nn.Parameter(torch.empty(heads, dim_out))
        self.bias = nn.Parameter(torch.zeros(heads * dim_out))
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, h, src, dst, n):
        wh = self.lin(h).view(-1, self.heads, self.dim_out)     # [N, H, D]
        a_src = (wh * self.att_src).sum(-1)                     # [N, H]
        a_dst = (wh * self.att_dst).sum(-1)
        logits = F.leaky_relu(a_src[src] + a_dst[dst], self.slope)   # [E, H]
        alpha = scatter_softmax(logits, dst, n)                 # [E, H]
        msg = wh[src] * alpha.unsqueeze(-1)                     # [E, H, D]
        out = torch.zeros(n, self.heads, self.dim_out, dtype=h.dtype, device=h.device)
        out.index_add_(0, dst, msg)
        return out.reshape(n, self.heads * self.dim_out) + self.bias


class MLPLayer(nn.Module):
    """Baseline 4: identical shape, message passing deleted."""

    def __init__(self, dim_in: int, dim_out: int):
        super().__init__()
        self.lin = nn.Linear(dim_in, dim_out, bias=True)

    def forward(self, h, src, dst, n):
        return self.lin(h)


class Encoder(nn.Module):
    def __init__(self, kind: EncoderName, dim_in: int, hidden: int, layers: int,
                 heads: int = 2, dropout: float = 0.1):
        super().__init__()
        self.kind, self.dropout = kind, dropout
        mods, d = [], dim_in
        for _ in range(layers):
            if kind == "graphsage":
                mods.append(SAGELayer(d, hidden)); d = hidden
            elif kind == "gat":
                per = max(hidden // heads, 4)
                mods.append(GATLayer(d, per, heads=heads)); d = per * heads
            elif kind == "mlp":
                mods.append(MLPLayer(d, hidden)); d = hidden
            else:
                raise ValueError(f"unknown encoder '{kind}'")
        self.layers = nn.ModuleList(mods)
        self.dim_out = d

    def forward(self, x, src, dst):
        n = x.size(0)
        h = x
        for i, layer in enumerate(self.layers):
            h = layer(h, src, dst, n)
            if i < len(self.layers) - 1:
                h = F.relu(h)
                h = F.dropout(h, self.dropout, self.training)
        return h


# --------------------------------------------------------------------------
# full model
# --------------------------------------------------------------------------
class EdgeTravelTimeModel(nn.Module):
    def __init__(self, encoder: EncoderName = "graphsage", hidden: int = 48,
                 layers: int = 2, heads: int = 2, head_hidden: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder_kind = encoder
        self.encoder = Encoder(encoder, NODE_FEATURE_DIM, hidden, layers, heads, dropout)
        head_in = 2 * self.encoder.dim_out + EDGE_FEATURE_DIM + TIME_FEATURE_DIM
        self.head = nn.Sequential(
            nn.Linear(head_in, head_hidden), nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden, head_hidden // 2), nn.ReLU(),
            nn.Linear(head_hidden // 2, 1),
        )
        self.config = dict(encoder=encoder, hidden=hidden, layers=layers,
                           heads=heads, head_hidden=head_hidden, dropout=dropout)

    def embed(self, node_x, src, dst):
        return self.encoder(node_x, src, dst)

    def forward(self, node_x, src, dst, edge_uv, edge_feats, time_feats):
        """Returns log1p(minutes). `edge_uv` is [B, 2] node indices."""
        emb = self.embed(node_x, src, dst)
        z = torch.cat([emb[edge_uv[:, 0]], emb[edge_uv[:, 1]], edge_feats, time_feats], dim=-1)
        return self.head(z).squeeze(-1)

    # -- export ---------------------------------------------------------
    def export_npz(self, path: str, norm: dict, metrics: dict | None = None,
                   extra: dict | None = None) -> dict:
        """Flatten every parameter into a NumPy archive that `gnn_numpy` can
        run without PyTorch installed."""
        arrays: dict[str, np.ndarray] = {}
        for name, p in self.state_dict().items():
            arrays[name] = p.detach().cpu().numpy().astype(np.float32)
        for k, v in norm.items():
            arrays[f"norm.{k}"] = np.asarray(v, dtype=np.float32)
        meta = dict(
            encoder=self.encoder_kind, config=self.config,
            features=feature_signature(), metrics=metrics or {}, **(extra or {}),
        )
        import json
        arrays["__meta__"] = np.frombuffer(
            json.dumps(meta).encode("utf-8"), dtype=np.uint8)
        np.savez_compressed(path, **arrays)
        return meta


def huber_log_loss(pred_log: torch.Tensor, target_min: torch.Tensor,
                   delta: float = 1.0) -> torch.Tensor:
    """Huber loss over log(1 + travel_time), as specified."""
    return F.huber_loss(pred_log, torch.log1p(target_min), delta=delta)


def to_minutes(pred_log: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.expm1(pred_log), min=0.05)
