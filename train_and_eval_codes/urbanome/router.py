from typing import Dict, List, Tuple

import torch
from torch import nn


class GatedRouter(nn.Module):
    def __init__(self, context_dim: int, num_experts: int, hidden_dim: int, temperature: float = 1.0):
        super().__init__()
        self.temperature = max(float(temperature), 1e-4)
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        logits = self.net(context) / self.temperature
        return torch.softmax(logits, dim=-1)


def select_topk(weights: torch.Tensor, top_k: int) -> Tuple[torch.Tensor, torch.Tensor]:
    values, indices = torch.topk(weights, k=min(top_k, weights.shape[-1]), dim=-1)
    values = values / values.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    return values, indices


class OpportunityRouter(nn.Module):
    def __init__(
        self,
        context_dim: int,
        hidden_dim: int,
        modality_names: List[str],
        region_names: List[str],
        task_cluster_names: List[str],
        num_task_experts_per_cluster: int,
        temperature: float = 1.0,
        use_modality_routing: bool = True,
    ):
        super().__init__()
        self.use_modality_routing = bool(use_modality_routing)
        self.num_modalities = len(modality_names)
        router_dim = context_dim + hidden_dim * 3
        self.modality_router = (
            GatedRouter(router_dim, len(modality_names), hidden_dim, temperature=1.0)
            if self.use_modality_routing
            else None
        )
        self.region_router = GatedRouter(router_dim, len(region_names), hidden_dim, temperature=temperature)
        self.task_cluster_router = GatedRouter(router_dim, len(task_cluster_names), hidden_dim, temperature=temperature)
        self.cluster_task_routers = nn.ModuleDict(
            {
                cluster_name: GatedRouter(
                    router_dim,
                    num_task_experts_per_cluster,
                    hidden_dim,
                    temperature=temperature,
                )
                for cluster_name in task_cluster_names
            }
        )

    def forward(
        self,
        context: torch.Tensor,
        visual_feature: torch.Tensor,
        task_prompt: torch.Tensor,
        task_cluster_name: str,
    ) -> Dict[str, torch.Tensor]:
        visual_task_interaction = visual_feature * task_prompt
        router_input = torch.cat([context, visual_feature, task_prompt, visual_task_interaction], dim=-1)
        if self.use_modality_routing:
            modality_weights = self.modality_router(router_input)
        else:
            modality_weights = torch.ones(
                router_input.shape[0],
                self.num_modalities,
                device=router_input.device,
                dtype=router_input.dtype,
            )
            modality_weights = modality_weights / modality_weights.shape[-1]
        return {
            "modality": modality_weights,
            "region": self.region_router(router_input),
            "task_cluster": self.task_cluster_router(router_input),
            "task": self.cluster_task_routers[task_cluster_name](router_input),
        }
