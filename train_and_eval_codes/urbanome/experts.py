from typing import Dict

import torch
from torch import nn

from .router import select_topk


class ExpertBlock(nn.Module):
    def __init__(self, hidden_dim: int, expert_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, expert_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expert_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ExpertPool(nn.Module):
    def __init__(self, expert_names, hidden_dim: int, expert_dim: int, dropout: float = 0.1):
        super().__init__()
        self.expert_names = list(expert_names)
        self.experts = nn.ModuleDict(
            {
                name: ExpertBlock(hidden_dim, expert_dim, dropout)
                for name in self.expert_names
            }
        )

    def forward(self, x: torch.Tensor, weights: torch.Tensor, top_k: int) -> Dict[str, torch.Tensor]:
        top_values, top_indices = select_topk(weights, top_k=top_k)
        all_outputs = [self.experts[name](x).unsqueeze(1) for name in self.expert_names]
        stacked = torch.cat(all_outputs, dim=1)

        gather_idx = top_indices.unsqueeze(-1).expand(-1, -1, stacked.shape[-1])
        selected = torch.gather(stacked, dim=1, index=gather_idx)
        mixed = (selected * top_values.unsqueeze(-1)).sum(dim=1)
        return {
            "mixed": mixed,
            "all": stacked,
            "top_values": top_values,
            "top_indices": top_indices,
        }


class FactorizedComposer(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fuse_with_modality = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.fuse_without_modality = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        shared: torch.Tensor,
        modality_mix: torch.Tensor,
        region_mix: torch.Tensor,
        task_mix: torch.Tensor,
        include_modality: bool = True,
    ) -> torch.Tensor:
        if include_modality:
            concat = torch.cat([shared, modality_mix, region_mix, task_mix], dim=-1)
            return self.fuse_with_modality(concat)
        concat = torch.cat([shared, region_mix, task_mix], dim=-1)
        return self.fuse_without_modality(concat)
