from typing import Dict

import torch


def build_pseudo_labels(
    predictions: Dict[str, torch.Tensor],
    stability: torch.Tensor,
    threshold: float,
) -> Dict[str, torch.Tensor]:
    mask = stability >= threshold
    return {task: pred.detach() for task, pred in predictions.items()}, mask


def consistency_regularization(
    task_predictions: Dict[str, torch.Tensor],
    modality_predictions: Dict[str, torch.Tensor],
) -> torch.Tensor:
    task_stack = torch.stack(list(task_predictions.values()), dim=1)
    modality_stack = torch.stack(list(modality_predictions.values()), dim=1)
    task_mean = task_stack.mean(dim=1, keepdim=True)
    modality_mean = modality_stack.mean(dim=1, keepdim=True)
    task_loss = (task_stack - task_mean).pow(2).mean()
    modality_loss = (modality_stack - modality_mean).pow(2).mean()
    return task_loss + modality_loss


def masked_regression_loss(
    predictions: Dict[str, torch.Tensor],
    pseudo_labels: Dict[str, torch.Tensor],
    mask: torch.Tensor,
) -> torch.Tensor:
    if mask.numel() == 0 or not mask.any():
        device = next(iter(predictions.values())).device
        return torch.zeros((), device=device)

    total = 0.0
    valid_mask = mask.view(-1, 1)
    for task_name, pred in predictions.items():
        total = total + ((pred - pseudo_labels[task_name]) ** 2)[valid_mask].mean()
    return total
