from collections import defaultdict
from typing import Dict, List
import math

import pandas as pd
import torch
from torch import nn
from pandas.api.types import is_numeric_dtype

try:
    from tqdm.auto import tqdm
except ImportError:
    tqdm = None

from .config import UrbanOMEConfig
from .tta import build_pseudo_labels, consistency_regularization, masked_regression_loss


def round_nested_metrics(obj, decimals: int = 3):
    if isinstance(obj, dict):
        return {key: round_nested_metrics(value, decimals) for key, value in obj.items()}
    if isinstance(obj, list):
        return [round_nested_metrics(value, decimals) for value in obj]
    if isinstance(obj, float):
        return round(obj, decimals)
    return obj


def round_dataframe_numeric(df: pd.DataFrame, decimals: int = 3) -> pd.DataFrame:
    rounded = df.copy()
    for col in rounded.columns:
        if is_numeric_dtype(rounded[col]):
            rounded[col] = rounded[col].round(decimals)
    return rounded


def _print_debug_batch(
    batch: Dict[str, torch.Tensor],
    outputs: Dict[str, Dict[str, torch.Tensor]],
    config: UrbanOMEConfig,
    stage: str,
    batch_idx: int,
):
    active_task_counts = {
        task_name: int(batch["target_mask"][:, idx].sum().item())
        for idx, task_name in enumerate(config.task_names)
    }
    print(f"[debug][{stage}] batch_idx={batch_idx}")
    print(
        "[debug] input_shapes:",
        {
            "satellite": tuple(batch["satellite"].shape),
            "street_view": tuple(batch["street_view"].shape),
            "tabular": tuple(batch["tabular"].shape),
            "routing_context": tuple(batch["routing_context"].shape),
            "targets": tuple(batch["targets"].shape),
            "target_mask": tuple(batch["target_mask"].shape),
        },
    )
    print(
        "[debug] sample_meta:",
        {
            "country": batch["country"][0] if len(batch["country"]) > 0 else "",
            "city": batch["city"][0] if len(batch["city"]) > 0 else "",
            "grid": batch["grid"][0] if len(batch["grid"]) > 0 else "",
            "task_key": batch["task_key"][0] if len(batch["task_key"]) > 0 else "",
            "satellite_path": batch["satellite_path"][0] if len(batch["satellite_path"]) > 0 else "",
            "street_view_paths": batch["street_view_paths"][0] if len(batch["street_view_paths"]) > 0 else [],
        },
    )
    print("[debug] config_task_names:", config.task_names)
    print("[debug] active_task_counts:", active_task_counts)
    first_prompt = batch["task_prompt"][0] if len(batch["task_prompt"]) > 0 else ""
    print("[debug] first_task_prompt_full:")
    print(first_prompt)
    print(
        "[debug] encoded_shapes:",
        {name: tuple(value.shape) for name, value in outputs["encoded"].items()},
    )
    print(
        "[debug] prompt_embeddings_shape:",
        tuple(outputs["debug"]["prompt_embeddings"].shape),
    )
    print(
        "[debug] prompt_embedding_norms:",
        outputs["debug"]["prompt_embeddings"].norm(dim=-1).detach().cpu().tolist(),
    )
    print(
        "[debug] shared_shape:",
        tuple(outputs["debug"]["shared"].shape),
    )
    print(
        "[debug] routing_weights_first_sample:",
        {
            key: value[0].detach().cpu().tolist()
            for key, value in outputs["routing_weights"].items()
        },
    )
    print(
        "[debug] prediction_preview:",
        {
            task_name: float(pred[0].detach().cpu().item())
            for task_name, pred in outputs["task_predictions"].items()
        },
    )
    print(
        "[debug] target_preview:",
        {
            task_name: {
                "target": float(batch["targets"][0, idx].item()),
                "mask": float(batch["target_mask"][0, idx].item()),
            }
            for idx, task_name in enumerate(config.task_names)
        },
    )
    print(
        "[debug] stability_preview:",
        {
            key: value[: min(3, value.shape[0])].detach().cpu().tolist()
            for key, value in outputs["stability"].items()
        },
    )


def split_targets(targets: torch.Tensor, task_names) -> Dict[str, torch.Tensor]:
    return {
        task_name: targets[:, idx : idx + 1]
        for idx, task_name in enumerate(task_names)
    }


def split_target_masks(target_mask: torch.Tensor, task_names) -> Dict[str, torch.Tensor]:
    return {
        task_name: target_mask[:, idx : idx + 1]
        for idx, task_name in enumerate(task_names)
    }


def compute_supervised_loss(
    outputs: Dict[str, Dict[str, torch.Tensor]],
    targets: Dict[str, torch.Tensor],
    target_masks: Dict[str, torch.Tensor],
    config: UrbanOMEConfig,
    prediction_key: str = "task_predictions",
) -> torch.Tensor:
    loss = None
    total_weight = 0.0
    prediction_dict = outputs[prediction_key]
    for task_name, pred in prediction_dict.items():
        task_weight = 1.0
        if config.task_loss_weights is not None:
            task_weight = float(config.task_loss_weights.get(task_name, 1.0))
        mask = target_masks[task_name]
        valid_count = mask.sum()
        if valid_count.item() <= 0:
            continue
        squared_error = (pred - targets[task_name]).pow(2) * mask
        task_loss = squared_error.sum() / valid_count.clamp_min(1.0)
        loss = task_weight * task_loss if loss is None else loss + task_weight * task_loss
        total_weight += task_weight
    if loss is None:
        first_pred = next(iter(prediction_dict.values()))
        return torch.zeros((), dtype=first_pred.dtype, device=first_pred.device)
    if config.normalize_task_loss and total_weight > 0:
        loss = loss / total_weight
    return loss


def compute_residual_regularization(
    outputs: Dict[str, Dict[str, torch.Tensor]],
    targets: Dict[str, torch.Tensor],
    target_masks: Dict[str, torch.Tensor],
    config: UrbanOMEConfig,
) -> torch.Tensor:
    if config.residual_reg_loss_weight <= 0:
        first_pred = next(iter(outputs["task_predictions"].values()))
        return torch.zeros((), dtype=first_pred.dtype, device=first_pred.device)

    residual_predictions = outputs.get("residual_task_predictions", {})
    if not residual_predictions:
        first_pred = next(iter(outputs["task_predictions"].values()))
        return torch.zeros((), dtype=first_pred.dtype, device=first_pred.device)

    aux_outputs = {"task_predictions": residual_predictions}
    residual_supervised_loss = compute_supervised_loss(
        aux_outputs,
        targets,
        target_masks,
        config,
    )
    return config.residual_reg_loss_weight * residual_supervised_loss


def compute_region_aux_loss(
    outputs: Dict[str, Dict[str, torch.Tensor]],
    batch: Dict[str, torch.Tensor],
    config: UrbanOMEConfig,
    device: torch.device,
) -> torch.Tensor:
    aux_outputs = outputs.get("aux_outputs", {})
    total_loss = None

    if "region_logits" in aux_outputs:
        region_targets = batch["region_label"].to(device)
        valid = region_targets >= 0
        if valid.any():
            region_loss = nn.functional.cross_entropy(aux_outputs["region_logits"][valid], region_targets[valid])
            total_loss = config.region_aux_loss_weight * region_loss if total_loss is None else total_loss + config.region_aux_loss_weight * region_loss

    if "income_logits" in aux_outputs:
        income_targets = batch["income_label"].to(device)
        valid = income_targets >= 0
        if valid.any():
            income_loss = nn.functional.cross_entropy(aux_outputs["income_logits"][valid], income_targets[valid])
            total_loss = config.income_aux_loss_weight * income_loss if total_loss is None else total_loss + config.income_aux_loss_weight * income_loss

    if total_loss is None:
        first_pred = next(iter(outputs["task_predictions"].values()))
        return torch.zeros((), dtype=first_pred.dtype, device=first_pred.device)
    return total_loss


def compute_load_balance_loss(
    outputs: Dict[str, Dict[str, torch.Tensor]],
    config: UrbanOMEConfig,
    device: torch.device,
) -> torch.Tensor:
    routing_weights = outputs["routing_weights"]
    total_loss = None
    for key in ["region", "task_cluster", "task"]:
        weights = routing_weights[key]
        avg_usage = weights.mean(dim=0)
        target = torch.full_like(avg_usage, 1.0 / avg_usage.numel(), device=device)
        loss = ((avg_usage - target) ** 2).mean()
        total_loss = loss if total_loss is None else total_loss + loss
    if total_loss is None:
        first_pred = next(iter(outputs["task_predictions"].values()))
        return torch.zeros((), dtype=first_pred.dtype, device=first_pred.device)
    return config.load_balance_loss_weight * total_loss


def compute_expert_diversity_loss(
    outputs: Dict[str, Dict[str, torch.Tensor]],
    config: UrbanOMEConfig,
    device: torch.device,
) -> torch.Tensor:
    expert_outputs = outputs.get("expert_outputs", {})
    total_loss = None

    for key in ["modality", "region", "task_cluster", "task"]:
        family_outputs = expert_outputs.get(key, [])
        if not family_outputs:
            continue
        stacked = torch.stack(family_outputs, dim=0)
        mean_per_expert = stacked.mean(dim=(0, 1))
        if mean_per_expert.shape[0] <= 1:
            continue
        normalized = nn.functional.normalize(mean_per_expert, dim=-1)
        sim = normalized @ normalized.transpose(0, 1)
        eye = torch.eye(sim.shape[0], device=sim.device, dtype=sim.dtype)
        off_diag = sim * (1.0 - eye)
        denom = max(sim.numel() - sim.shape[0], 1)
        family_loss = off_diag.pow(2).sum() / denom
        total_loss = family_loss if total_loss is None else total_loss + family_loss

    if total_loss is None:
        first_pred = next(iter(outputs["task_predictions"].values()))
        return torch.zeros((), dtype=first_pred.dtype, device=device)
    return config.expert_diversity_loss_weight * total_loss


def compute_route_entropy_loss(
    outputs: Dict[str, Dict[str, torch.Tensor]],
    config: UrbanOMEConfig,
    device: torch.device,
) -> torch.Tensor:
    routing_weights = outputs["routing_weights"]
    total_loss = None
    counted = 0.0
    for key in ["region", "task_cluster", "task"]:
        weights = routing_weights[key]
        entropy = -(weights * weights.clamp_min(1e-8).log()).sum(dim=-1).mean()
        total_loss = entropy if total_loss is None else total_loss + entropy
        counted += 1.0
    if total_loss is None or counted <= 0:
        first_pred = next(iter(outputs["task_predictions"].values()))
        return torch.zeros((), dtype=first_pred.dtype, device=device)
    return config.route_entropy_loss_weight * (total_loss / counted)


def compute_task_cluster_consistency_loss(
    outputs: Dict[str, Dict[str, torch.Tensor]],
    config: UrbanOMEConfig,
    device: torch.device,
) -> torch.Tensor:
    if config.task_cluster_consistency_loss_weight <= 0:
        first_pred = next(iter(outputs["task_predictions"].values()))
        return torch.zeros((), dtype=first_pred.dtype, device=device)

    task_features = outputs.get("task_features", {})
    if not task_features:
        first_pred = next(iter(outputs["task_predictions"].values()))
        return torch.zeros((), dtype=first_pred.dtype, device=device)

    task_cluster_map = config.task_cluster_map or {}
    grouped_centers = defaultdict(list)
    task_centers = {}
    for task_name, feature in task_features.items():
        if feature.numel() <= 0:
            continue
        center = nn.functional.normalize(feature.mean(dim=0), dim=0)
        task_centers[task_name] = center
        cluster_name = task_cluster_map.get(task_name, "general_cluster")
        grouped_centers[cluster_name].append(center)

    if not grouped_centers:
        first_pred = next(iter(outputs["task_predictions"].values()))
        return torch.zeros((), dtype=first_pred.dtype, device=device)

    intra_loss = None
    intra_count = 0.0
    cluster_prototypes = {}
    for cluster_name, centers in grouped_centers.items():
        stacked = torch.stack(centers, dim=0)
        prototype = nn.functional.normalize(stacked.mean(dim=0), dim=0)
        cluster_prototypes[cluster_name] = prototype
        if stacked.shape[0] <= 1:
            continue
        cosine_distance = 1.0 - (stacked @ prototype.unsqueeze(-1)).squeeze(-1)
        cluster_loss = cosine_distance.mean()
        intra_loss = cluster_loss if intra_loss is None else intra_loss + cluster_loss
        intra_count += 1.0

    inter_loss = None
    inter_count = 0.0
    prototype_names = list(cluster_prototypes.keys())
    if len(prototype_names) > 1:
        for idx, cluster_i in enumerate(prototype_names):
            proto_i = cluster_prototypes[cluster_i]
            for cluster_j in prototype_names[idx + 1 :]:
                proto_j = cluster_prototypes[cluster_j]
                cosine_sim = torch.sum(proto_i * proto_j)
                pair_loss = torch.relu(cosine_sim - config.task_cluster_separation_margin)
                inter_loss = pair_loss if inter_loss is None else inter_loss + pair_loss
                inter_count += 1.0

    total_loss = None
    if intra_loss is not None and intra_count > 0:
        total_loss = intra_loss / intra_count
    if inter_loss is not None and inter_count > 0:
        inter_term = inter_loss / inter_count
        total_loss = inter_term if total_loss is None else total_loss + inter_term

    if total_loss is None:
        first_pred = next(iter(outputs["task_predictions"].values()))
        return torch.zeros((), dtype=first_pred.dtype, device=device)
    return config.task_cluster_consistency_loss_weight * total_loss


def compute_unseen_task_alignment_loss(
    outputs: Dict[str, Dict[str, torch.Tensor]],
    config: UrbanOMEConfig,
    device: torch.device,
) -> torch.Tensor:
    if config.unseen_task_alignment_loss_weight <= 0:
        first_pred = next(iter(outputs["task_predictions"].values()))
        return torch.zeros((), dtype=first_pred.dtype, device=device)

    task_features = outputs.get("task_features", {})
    if not task_features:
        first_pred = next(iter(outputs["task_predictions"].values()))
        return torch.zeros((), dtype=first_pred.dtype, device=device)

    seen_tasks = set(config.train_task_keys or [])
    unseen_tasks = set(config.test_task_keys or [])
    if not seen_tasks or not unseen_tasks:
        first_pred = next(iter(outputs["task_predictions"].values()))
        return torch.zeros((), dtype=first_pred.dtype, device=device)

    task_cluster_map = config.task_cluster_map or {}
    cluster_seen_centers = defaultdict(list)
    task_centers = {}

    for task_name, feature in task_features.items():
        center = nn.functional.normalize(feature.mean(dim=0), dim=0)
        task_centers[task_name] = center
        if task_name in seen_tasks:
            cluster_name = task_cluster_map.get(task_name, "general_cluster")
            cluster_seen_centers[cluster_name].append(center)

    if not cluster_seen_centers:
        first_pred = next(iter(outputs["task_predictions"].values()))
        return torch.zeros((), dtype=first_pred.dtype, device=device)

    cluster_prototypes = {}
    for cluster_name, centers in cluster_seen_centers.items():
        stacked = torch.stack(centers, dim=0)
        cluster_prototypes[cluster_name] = nn.functional.normalize(stacked.mean(dim=0), dim=0)

    total_loss = None
    counted = 0.0
    for task_name in unseen_tasks:
        if task_name not in task_centers:
            continue
        cluster_name = task_cluster_map.get(task_name, "general_cluster")
        if cluster_name not in cluster_prototypes:
            continue
        cosine_distance = 1.0 - torch.sum(task_centers[task_name] * cluster_prototypes[cluster_name])
        total_loss = cosine_distance if total_loss is None else total_loss + cosine_distance
        counted += 1.0

    if total_loss is None or counted <= 0:
        first_pred = next(iter(outputs["task_predictions"].values()))
        return torch.zeros((), dtype=first_pred.dtype, device=device)
    return config.unseen_task_alignment_loss_weight * (total_loss / counted)


def compute_unseen_task_router_distill_loss(
    outputs: Dict[str, Dict[str, torch.Tensor]],
    config: UrbanOMEConfig,
    device: torch.device,
) -> torch.Tensor:
    if config.unseen_task_router_distill_loss_weight <= 0:
        first_pred = next(iter(outputs["task_predictions"].values()))
        return torch.zeros((), dtype=first_pred.dtype, device=device)

    routing_by_task = outputs.get("routing_weights_by_task", {})
    if not routing_by_task:
        first_pred = next(iter(outputs["task_predictions"].values()))
        return torch.zeros((), dtype=first_pred.dtype, device=device)

    seen_tasks = set(config.train_task_keys or [])
    unseen_tasks = set(config.test_task_keys or [])
    if not seen_tasks or not unseen_tasks:
        first_pred = next(iter(outputs["task_predictions"].values()))
        return torch.zeros((), dtype=first_pred.dtype, device=device)

    task_cluster_map = config.task_cluster_map or {}
    cluster_seen_routes = {
        "region": defaultdict(list),
        "task_cluster": defaultdict(list),
        "task": defaultdict(list),
    }

    for task_name, routing_parts in routing_by_task.items():
        if task_name not in seen_tasks:
            continue
        cluster_name = task_cluster_map.get(task_name, "general_cluster")
        for key in ["region", "task_cluster", "task"]:
            cluster_seen_routes[key][cluster_name].append(routing_parts[key])

    total_loss = None
    counted = 0.0
    for task_name in unseen_tasks:
        if task_name not in routing_by_task:
            continue
        cluster_name = task_cluster_map.get(task_name, "general_cluster")
        unseen_parts = routing_by_task[task_name]
        for key in ["region", "task_cluster", "task"]:
            seen_list = cluster_seen_routes[key].get(cluster_name, [])
            if not seen_list:
                continue
            target_distribution = torch.stack(seen_list, dim=0).mean(dim=0).detach()
            student_distribution = unseen_parts[key].clamp_min(1e-8)
            teacher_distribution = target_distribution.clamp_min(1e-8)
            kl = (
                teacher_distribution
                * (teacher_distribution.log() - student_distribution.log())
            ).sum(dim=-1).mean()
            total_loss = kl if total_loss is None else total_loss + kl
            counted += 1.0

    if total_loss is None or counted <= 0:
        first_pred = next(iter(outputs["task_predictions"].values()))
        return torch.zeros((), dtype=first_pred.dtype, device=device)
    return config.unseen_task_router_distill_loss_weight * (total_loss / counted)


def sample_masked_task_names(
    batch: Dict[str, torch.Tensor],
    config: UrbanOMEConfig,
) -> List[str]:
    if not config.task_mask_meta_train:
        return []
    candidate_tasks = []
    train_task_set = set(config.train_task_keys or config.task_names)
    for idx, task_name in enumerate(config.task_names):
        if task_name not in train_task_set:
            continue
        if float(batch["target_mask"][:, idx].sum().item()) <= 0:
            continue
        candidate_tasks.append(task_name)
    if len(candidate_tasks) <= 1:
        return []

    raw_count = int(math.ceil(len(candidate_tasks) * config.task_mask_ratio))
    mask_count = max(config.min_masked_tasks, raw_count)
    mask_count = min(mask_count, config.max_masked_tasks, len(candidate_tasks) - 1)
    if mask_count <= 0:
        return []

    perm = torch.randperm(len(candidate_tasks)).tolist()
    return [candidate_tasks[idx] for idx in perm[:mask_count]]


def build_task_subset_masks(
    target_masks: Dict[str, torch.Tensor],
    selected_task_names: List[str],
) -> Dict[str, torch.Tensor]:
    selected = set(selected_task_names)
    subset_masks = {}
    for task_name, mask in target_masks.items():
        if task_name in selected:
            subset_masks[task_name] = mask
        else:
            subset_masks[task_name] = torch.zeros_like(mask)
    return subset_masks


def flatten_task_predictions(
    predictions: Dict[str, torch.Tensor],
    task_names: List[str],
) -> torch.Tensor:
    return torch.cat([predictions[task_name] for task_name in task_names], dim=1)


def compute_group_metrics(rows: List[Dict[str, torch.Tensor]]) -> Dict[str, float]:
    if not rows:
        return {"mse": 0.0, "mae": 0.0, "r2": 0.0}

    pred = torch.stack([row["pred"] for row in rows], dim=0)
    target = torch.stack([row["target"] for row in rows], dim=0)
    mask = torch.stack([row["mask"] for row in rows], dim=0)
    valid = mask > 0
    if valid.sum().item() <= 0:
        return {"mse": 0.0, "mae": 0.0, "r2": 0.0}
    diff = pred - target
    mse = (diff.pow(2) * mask).sum().item() / mask.sum().item()
    mae = (diff.abs() * mask).sum().item() / mask.sum().item()
    target_mean = (target * mask).sum() / mask.sum().clamp_min(1.0)
    denom = (((target - target_mean) ** 2) * mask).sum()
    if denom.item() <= 1e-12:
        r2 = 0.0
    else:
        r2 = 1.0 - float(((diff.pow(2) * mask).sum() / denom).item())
    return {"mse": mse, "mae": mae, "r2": r2}


def compute_masked_r2(
    pred_matrix: torch.Tensor,
    target_matrix: torch.Tensor,
    target_mask: torch.Tensor,
) -> float:
    valid = target_mask > 0
    if valid.sum().item() <= 0:
        return 0.0
    pred_valid = pred_matrix[valid]
    target_valid = target_matrix[valid]
    target_mean = target_valid.mean()
    ss_res = ((pred_valid - target_valid) ** 2).sum()
    ss_tot = ((target_valid - target_mean) ** 2).sum()
    if ss_tot.item() <= 1e-12:
        return 0.0
    return float((1.0 - ss_res / ss_tot).item())


def aggregate_group_rows(rows: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    pred = torch.stack([row["pred"] for row in rows], dim=0)
    target = torch.stack([row["target"] for row in rows], dim=0)
    mask = torch.stack([row["mask"] for row in rows], dim=0)
    denom = mask.sum(dim=0).clamp_min(1.0)
    mean_pred = (pred * mask).sum(dim=0) / denom
    mean_target = (target * mask).sum(dim=0) / denom
    group_mask = (mask.sum(dim=0) > 0).float()
    return {
        "pred": mean_pred,
        "target": mean_target,
        "mask": group_mask,
    }


def format_metrics_table(metrics: Dict[str, float]) -> List[str]:
    lines = [
        (
            f"test_grid_mse={metrics.get('grid_mse', 0.0):.3f} "
            f"test_grid_mae={metrics.get('grid_mae', 0.0):.3f} "
            f"test_grid_R2={metrics.get('grid_r2', 0.0):.3f}"
        )
    ]
    per_target = metrics.get("per_target", {})
    for task_name, task_metrics in per_target.items():
        if int(task_metrics.get("count", 0)) <= 0:
            continue
        lines.append(
            f"{task_name} mse={task_metrics.get('mse', 0.0):.3f} "
            f"mae={task_metrics.get('mae', 0.0):.3f} "
            f"R2={task_metrics.get('r2', 0.0):.3f}"
        )
    return lines


def compute_per_target_metrics(
    pred_matrix: torch.Tensor,
    target_matrix: torch.Tensor,
    target_mask: torch.Tensor,
    task_names: List[str],
) -> Dict[str, Dict[str, float]]:
    metrics = {}
    for idx, task_name in enumerate(task_names):
        pred = pred_matrix[:, idx]
        target = target_matrix[:, idx]
        mask = target_mask[:, idx]
        valid = mask > 0
        if valid.sum().item() <= 0:
            metrics[task_name] = {"mse": 0.0, "mae": 0.0, "r2": 0.0, "count": 0}
            continue
        diff = pred - target
        pred_valid = pred[valid]
        target_valid = target[valid]
        target_mean = target_valid.mean()
        ss_res = ((pred_valid - target_valid) ** 2).sum()
        ss_tot = ((target_valid - target_mean) ** 2).sum()
        r2 = 0.0 if ss_tot.item() <= 1e-12 else float((1.0 - ss_res / ss_tot).item())
        metrics[task_name] = {
            "mse": ((diff.pow(2) * mask).sum() / mask.sum()).item(),
            "mae": ((diff.abs() * mask).sum() / mask.sum()).item(),
            "r2": r2,
            "count": int(valid.sum().item()),
        }
    return metrics


def _wrap_progress(iterable, config: UrbanOMEConfig, desc: str):
    if not config.show_progress or tqdm is None:
        return iterable
    total = len(iterable) if hasattr(iterable, "__len__") else None
    return tqdm(iterable, total=total, desc=desc, leave=False, dynamic_ncols=True)


@torch.no_grad()
def evaluate(model, data_loader, config: UrbanOMEConfig, device: torch.device):
    model.eval()
    total_loss = 0.0
    num_batches = 0
    city_rows = defaultdict(list)
    country_rows = defaultdict(list)
    all_pred_rows = []
    all_target_rows = []
    all_target_masks = []

    progress = _wrap_progress(data_loader, config, desc="eval")
    for batch_idx, batch in enumerate(progress):
        outputs = model(
            satellite=batch["satellite"].to(device),
            satellite_embedding=batch["satellite_embedding"].to(device),
            street_view=batch["street_view"].to(device),
            tabular=batch["tabular"].to(device),
            routing_context=batch["routing_context"].to(device),
            country_label=batch["country_label"].to(device),
            region_label=batch["region_label"].to(device),
            income_label=batch["income_label"].to(device),
            hemisphere_label=batch["hemisphere_label"].to(device),
        )
        if config.debug_forward and batch_idx < config.debug_batches:
            _print_debug_batch(batch, outputs, config, stage="eval", batch_idx=batch_idx)
        targets = split_targets(batch["targets"].to(device), config.task_names)
        target_masks = split_target_masks(batch["target_mask"].to(device), config.task_names)
        total_loss += float(compute_supervised_loss(outputs, targets, target_masks, config))
        num_batches += 1
        if config.show_progress and tqdm is not None and hasattr(progress, "set_postfix"):
            progress.set_postfix(loss=f"{total_loss / max(num_batches, 1):.4f}")

        pred_matrix = flatten_task_predictions(outputs["task_predictions"], config.task_names).cpu()
        target_matrix = batch["targets"].cpu()
        target_mask = batch["target_mask"].cpu()
        all_pred_rows.append(pred_matrix)
        all_target_rows.append(target_matrix)
        all_target_masks.append(target_mask)

        for idx in range(pred_matrix.shape[0]):
            row_payload = {
                "pred": pred_matrix[idx],
                "target": target_matrix[idx],
                "mask": target_mask[idx],
            }
            city_rows[batch["city"][idx]].append(row_payload)
            country_rows[batch["country"][idx]].append(row_payload)

    all_pred = torch.cat(all_pred_rows, dim=0) if all_pred_rows else torch.zeros((0, len(config.task_names)))
    all_target = torch.cat(all_target_rows, dim=0) if all_target_rows else torch.zeros((0, len(config.task_names)))
    all_mask = torch.cat(all_target_masks, dim=0) if all_target_masks else torch.zeros((0, len(config.task_names)))
    per_target_metrics = compute_per_target_metrics(all_pred, all_target, all_mask, config.task_names) if all_pred.numel() > 0 else {}
    city_agg = [aggregate_group_rows(rows) for rows in city_rows.values()]
    country_agg = [aggregate_group_rows(rows) for rows in country_rows.values()]
    city_metrics = [compute_group_metrics([row]) for row in city_agg]
    country_metrics = [compute_group_metrics([row]) for row in country_agg]
    city_pred = torch.stack([row["pred"] for row in city_agg], dim=0) if city_agg else torch.zeros((0, len(config.task_names)))
    city_target = torch.stack([row["target"] for row in city_agg], dim=0) if city_agg else torch.zeros((0, len(config.task_names)))
    city_mask = torch.stack([row["mask"] for row in city_agg], dim=0) if city_agg else torch.zeros((0, len(config.task_names)))
    country_pred = torch.stack([row["pred"] for row in country_agg], dim=0) if country_agg else torch.zeros((0, len(config.task_names)))
    country_target = torch.stack([row["target"] for row in country_agg], dim=0) if country_agg else torch.zeros((0, len(config.task_names)))
    country_mask = torch.stack([row["mask"] for row in country_agg], dim=0) if country_agg else torch.zeros((0, len(config.task_names)))

    return {
        "grid_mse": total_loss / max(num_batches, 1),
        "grid_mae": (
            ((all_pred - all_target).abs() * all_mask).sum().item() / all_mask.sum().item()
            if all_mask.sum().item() > 0
            else 0.0
        ),
        "grid_r2": compute_masked_r2(all_pred, all_target, all_mask),
        "city_mse": sum(m["mse"] for m in city_metrics) / max(len(city_metrics), 1),
        "city_mae": sum(m["mae"] for m in city_metrics) / max(len(city_metrics), 1),
        "city_r2": compute_masked_r2(city_pred, city_target, city_mask),
        "country_mse": sum(m["mse"] for m in country_metrics) / max(len(country_metrics), 1),
        "country_mae": sum(m["mae"] for m in country_metrics) / max(len(country_metrics), 1),
        "country_r2": compute_masked_r2(country_pred, country_target, country_mask),
        "per_target": per_target_metrics,
    }


def train_one_epoch(model, data_loader, optimizer, config: UrbanOMEConfig, device: torch.device):
    model.train()
    total_loss = 0.0
    num_batches = 0

    progress = _wrap_progress(data_loader, config, desc="train")
    for batch_idx, batch in enumerate(progress):
        optimizer.zero_grad()

        outputs = model(
            satellite=batch["satellite"].to(device),
            satellite_embedding=batch["satellite_embedding"].to(device),
            street_view=batch["street_view"].to(device),
            tabular=batch["tabular"].to(device),
            routing_context=batch["routing_context"].to(device),
            country_label=batch["country_label"].to(device),
            region_label=batch["region_label"].to(device),
            income_label=batch["income_label"].to(device),
            hemisphere_label=batch["hemisphere_label"].to(device),
        )
        if config.debug_forward and batch_idx < config.debug_batches:
            _print_debug_batch(batch, outputs, config, stage="train", batch_idx=batch_idx)
        targets = split_targets(batch["targets"].to(device), config.task_names)
        target_masks = split_target_masks(batch["target_mask"].to(device), config.task_names)

        supervised_loss = compute_supervised_loss(outputs, targets, target_masks, config)
        consistency_loss = torch.zeros((), dtype=supervised_loss.dtype, device=device)
        if config.consistency_loss_weight > 0:
            consistency_score = (
                outputs["stability"]["cross_task"].mean()
                + outputs["stability"]["cross_modality"].mean()
            ) / 2.0
            consistency_loss = config.consistency_loss_weight * (1.0 - consistency_score)
        region_aux_loss = (
            compute_region_aux_loss(outputs, batch, config, device)
            if (config.region_aux_loss_weight > 0 or config.income_aux_loss_weight > 0)
            else torch.zeros((), dtype=supervised_loss.dtype, device=device)
        )
        load_balance_loss = (
            compute_load_balance_loss(outputs, config, device)
            if config.load_balance_loss_weight > 0
            else torch.zeros((), dtype=supervised_loss.dtype, device=device)
        )
        expert_diversity_loss = (
            compute_expert_diversity_loss(outputs, config, device)
            if config.expert_diversity_loss_weight > 0
            else torch.zeros((), dtype=supervised_loss.dtype, device=device)
        )
        route_entropy_loss = (
            compute_route_entropy_loss(outputs, config, device)
            if config.route_entropy_loss_weight > 0
            else torch.zeros((), dtype=supervised_loss.dtype, device=device)
        )
        task_cluster_consistency_loss = (
            compute_task_cluster_consistency_loss(outputs, config, device)
            if config.task_cluster_consistency_loss_weight > 0
            else torch.zeros((), dtype=supervised_loss.dtype, device=device)
        )
        unseen_task_alignment_loss = (
            compute_unseen_task_alignment_loss(outputs, config, device)
            if config.unseen_task_alignment_loss_weight > 0
            else torch.zeros((), dtype=supervised_loss.dtype, device=device)
        )
        unseen_task_router_distill_loss = (
            compute_unseen_task_router_distill_loss(outputs, config, device)
            if config.unseen_task_router_distill_loss_weight > 0
            else torch.zeros((), dtype=supervised_loss.dtype, device=device)
        )
        residual_reg_loss = (
            compute_residual_regularization(outputs, targets, target_masks, config)
            if config.residual_reg_loss_weight > 0
            else torch.zeros((), dtype=supervised_loss.dtype, device=device)
        )
        masked_task_meta_loss = torch.zeros((), dtype=supervised_loss.dtype, device=device)
        masked_task_names = sample_masked_task_names(batch, config)
        if masked_task_names:
            masked_outputs = model(
                satellite=batch["satellite"].to(device),
                satellite_embedding=batch["satellite_embedding"].to(device),
                street_view=batch["street_view"].to(device),
                tabular=batch["tabular"].to(device),
                routing_context=batch["routing_context"].to(device),
                country_label=batch["country_label"].to(device),
                region_label=batch["region_label"].to(device),
                income_label=batch["income_label"].to(device),
                hemisphere_label=batch["hemisphere_label"].to(device),
                masked_task_names=masked_task_names,
            )
            masked_target_masks = build_task_subset_masks(target_masks, masked_task_names)
            masked_task_meta_loss = config.task_mask_loss_weight * compute_supervised_loss(
                masked_outputs,
                targets,
                masked_target_masks,
                config,
            )
        total = (
            supervised_loss
            + consistency_loss
            + region_aux_loss
            + load_balance_loss
            + expert_diversity_loss
            + route_entropy_loss
            + task_cluster_consistency_loss
            + unseen_task_alignment_loss
            + unseen_task_router_distill_loss
            + residual_reg_loss
            + masked_task_meta_loss
        )

        total.backward()
        optimizer.step()

        total_loss += float(total.detach())
        num_batches += 1
        if config.show_progress and tqdm is not None and hasattr(progress, "set_postfix"):
            progress.set_postfix(loss=f"{total_loss / max(num_batches, 1):.4f}")

    return {"train_loss": total_loss / max(num_batches, 1)}


def adapt_test_time(model, data_loader, optimizer, config: UrbanOMEConfig, device: torch.device):
    model.train()
    total_loss = 0.0
    num_batches = 0
    total_stable = 0
    total_samples = 0

    progress = _wrap_progress(data_loader, config, desc="tta")
    for batch in progress:
        optimizer.zero_grad()

        outputs = model(
            satellite=batch["satellite"].to(device),
            satellite_embedding=batch["satellite_embedding"].to(device),
            street_view=batch["street_view"].to(device),
            tabular=batch["tabular"].to(device),
            routing_context=batch["routing_context"].to(device),
            country_label=batch["country_label"].to(device),
            region_label=batch["region_label"].to(device),
            income_label=batch["income_label"].to(device),
            hemisphere_label=batch["hemisphere_label"].to(device),
        )

        pseudo_labels, mask = build_pseudo_labels(
            predictions=outputs["task_predictions"],
            stability=outputs["stability"]["stability"],
            threshold=config.pseudo_label_threshold,
        )
        pseudo_loss = masked_regression_loss(outputs["task_predictions"], pseudo_labels, mask)
        consistency_loss = consistency_regularization(
            outputs["task_predictions"],
            outputs["modality_predictions"],
        )
        total = pseudo_loss + 0.1 * consistency_loss

        if total.requires_grad:
            total.backward()
            optimizer.step()

        total_loss += float(total.detach())
        num_batches += 1
        total_stable += int(mask.sum().item())
        total_samples += int(mask.numel())
        if config.show_progress and tqdm is not None and hasattr(progress, "set_postfix"):
            progress.set_postfix(
                loss=f"{total_loss / max(num_batches, 1):.4f}",
                stable=f"{total_stable / max(total_samples, 1):.3f}",
            )

    stable_ratio = total_stable / max(total_samples, 1)
    return {
        "tta_loss": total_loss / max(num_batches, 1),
        "stable_ratio": stable_ratio,
    }


@torch.no_grad()
def export_predictions(model, data_loader, config: UrbanOMEConfig, device: torch.device, output_csv: str):
    model.eval()
    rows = []

    for batch in data_loader:
        outputs = model(
            satellite=batch["satellite"].to(device),
            satellite_embedding=batch["satellite_embedding"].to(device),
            street_view=batch["street_view"].to(device),
            tabular=batch["tabular"].to(device),
            routing_context=batch["routing_context"].to(device),
            country_label=batch["country_label"].to(device),
            region_label=batch["region_label"].to(device),
            income_label=batch["income_label"].to(device),
            hemisphere_label=batch["hemisphere_label"].to(device),
        )
        pred_matrix = flatten_task_predictions(outputs["task_predictions"], config.task_names).cpu()
        target_matrix = batch["targets"].cpu()
        target_mask = batch["target_mask"].cpu()
        stability = outputs["stability"]["stability"].detach().cpu()

        for idx in range(pred_matrix.shape[0]):
            row = {
                "country": batch["country"][idx],
                "city": batch["city"][idx],
                "grid": batch["grid"][idx],
                "stability": float(stability[idx]),
            }
            for task_idx, task_name in enumerate(config.task_names):
                row[f"pred_{task_name}"] = float(pred_matrix[idx, task_idx])
                is_labeled = bool(target_mask[idx, task_idx].item() > 0)
                row[f"has_target_{task_name}"] = int(is_labeled)
                if is_labeled:
                    row[f"target_{task_name}"] = float(target_matrix[idx, task_idx])
                    row[f"error_{task_name}"] = float(pred_matrix[idx, task_idx] - target_matrix[idx, task_idx])
                else:
                    row[f"target_{task_name}"] = None
                    row[f"error_{task_name}"] = None
            rows.append(row)

    if rows:
        output_df = pd.DataFrame(rows)
    else:
        columns = ["country", "city", "grid", "stability"]
        for task_name in config.task_names:
            columns.extend(
                [
                    f"pred_{task_name}",
                    f"has_target_{task_name}",
                    f"target_{task_name}",
                    f"error_{task_name}",
                ]
            )
        output_df = pd.DataFrame(columns=columns)
    round_dataframe_numeric(output_df, decimals=3).to_csv(output_csv, index=False)


def export_group_predictions(prediction_csv: str, city_output_csv: str, country_output_csv: str):
    df = pd.read_csv(prediction_csv)
    target_names = [col[len("pred_"):] for col in df.columns if col.startswith("pred_")]
    if len(df) == 0:
        city_columns = ["city", "country_count", "grid_count", "mean_stability"]
        country_columns = ["country", "city_count", "grid_count", "mean_stability"]
        for target_name in target_names:
            metric_cols = [
                f"pred_{target_name}",
                f"labeled_count_{target_name}",
                f"target_{target_name}",
                f"error_{target_name}",
                f"mae_{target_name}",
            ]
            city_columns.extend(metric_cols)
            country_columns.extend(metric_cols)
        pd.DataFrame(columns=city_columns).to_csv(city_output_csv, index=False)
        pd.DataFrame(columns=country_columns).to_csv(country_output_csv, index=False)
        return

    city_rows = []
    for city_name, city_df in df.groupby("city", dropna=False):
        row = {
            "city": city_name,
            "country_count": int(city_df["country"].nunique()),
            "grid_count": int(len(city_df)),
            "mean_stability": float(city_df["stability"].mean()),
        }
        for target_name in target_names:
            row[f"pred_{target_name}"] = float(city_df[f"pred_{target_name}"].mean())
            labeled_df = city_df[city_df[f"has_target_{target_name}"] > 0]
            row[f"labeled_count_{target_name}"] = int(len(labeled_df))
            row[f"target_{target_name}"] = float(labeled_df[f"target_{target_name}"].mean()) if len(labeled_df) > 0 else None
            row[f"error_{target_name}"] = float(labeled_df[f"error_{target_name}"].mean()) if len(labeled_df) > 0 else None
            row[f"mae_{target_name}"] = float(labeled_df[f"error_{target_name}"].abs().mean()) if len(labeled_df) > 0 else None
        city_rows.append(row)

    country_rows = []
    for country_name, country_df in df.groupby("country", dropna=False):
        row = {
            "country": country_name,
            "city_count": int(country_df["city"].nunique()),
            "grid_count": int(len(country_df)),
            "mean_stability": float(country_df["stability"].mean()),
        }
        for target_name in target_names:
            row[f"pred_{target_name}"] = float(country_df[f"pred_{target_name}"].mean())
            labeled_df = country_df[country_df[f"has_target_{target_name}"] > 0]
            row[f"labeled_count_{target_name}"] = int(len(labeled_df))
            row[f"target_{target_name}"] = float(labeled_df[f"target_{target_name}"].mean()) if len(labeled_df) > 0 else None
            row[f"error_{target_name}"] = float(labeled_df[f"error_{target_name}"].mean()) if len(labeled_df) > 0 else None
            row[f"mae_{target_name}"] = float(labeled_df[f"error_{target_name}"].abs().mean()) if len(labeled_df) > 0 else None
        country_rows.append(row)

    round_dataframe_numeric(pd.DataFrame(city_rows), decimals=3).to_csv(city_output_csv, index=False)
    round_dataframe_numeric(pd.DataFrame(country_rows), decimals=3).to_csv(country_output_csv, index=False)
