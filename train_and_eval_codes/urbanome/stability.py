from typing import Dict

import torch


def pairwise_agreement_score(preds: torch.Tensor) -> torch.Tensor:
    diffs = torch.cdist(preds, preds, p=1)
    agreement = torch.exp(-diffs.mean(dim=(-1, -2)))
    return agreement


def normalized_entropy(weights: torch.Tensor) -> torch.Tensor:
    entropy = -(weights * (weights.clamp_min(1e-8).log())).sum(dim=-1)
    max_entropy = torch.log(torch.tensor(weights.shape[-1], device=weights.device, dtype=weights.dtype))
    return 1.0 - entropy / max_entropy.clamp_min(1e-8)


class OpportunityStabilityScorer:
    def __init__(self, margin: float = 0.15):
        self.margin = margin

    def __call__(
        self,
        task_predictions: Dict[str, torch.Tensor],
        routing_weights: Dict[str, torch.Tensor],
        modality_predictions: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        task_stack = torch.stack(list(task_predictions.values()), dim=1)
        modality_stack = torch.stack(list(modality_predictions.values()), dim=1)

        cross_task = pairwise_agreement_score(task_stack)
        cross_modality = pairwise_agreement_score(modality_stack)

        route_parts = [
            normalized_entropy(routing_weights["region"]),
            normalized_entropy(routing_weights["task"]),
        ]
        if "modality" in routing_weights:
            route_parts.append(normalized_entropy(routing_weights["modality"]))
        route_conf = torch.stack(route_parts, dim=0).mean(dim=0)

        stability = (cross_task + cross_modality + route_conf) / 3.0
        stable_mask = stability >= (0.5 + self.margin)
        return {
            "stability": stability,
            "stable_mask": stable_mask,
            "cross_task": cross_task,
            "cross_modality": cross_modality,
            "route_confidence": route_conf,
        }
