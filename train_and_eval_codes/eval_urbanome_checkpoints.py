import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import pandas as pd
import torch

from eval_urbanome import build_config_from_checkpoint, initialize_prompt_encoder_for_loading
from urbanome import UrbanOME
from urbanome.data import build_dataloaders
from urbanome.trainer import (
    evaluate,
    export_group_predictions,
    export_predictions,
    format_metrics_table,
    round_nested_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate multiple UrbanOME checkpoints and summarize MSE/MAE.")
    parser.add_argument("--checkpoint_dir", type=str, default="")
    parser.add_argument("--checkpoint_paths", nargs="+", default=None)
    parser.add_argument("--epoch_numbers", nargs="+", type=int, default=None)
    parser.add_argument("--include_best", action="store_true")
    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--image_root", type=str, default=None)
    parser.add_argument("--satellite_path_template", type=str, default=None)
    parser.add_argument("--street_view_path_template", type=str, default=None)
    parser.add_argument("--satellite_embedding_col", type=str, default=None)
    parser.add_argument("--use_cached_satellite_embeddings", action="store_true")
    parser.add_argument("--prompt_encoder_type", type=str, default=None)
    parser.add_argument("--external_task_prompt_embeddings_path", type=str, default=None)
    parser.add_argument("--task_prompt_embedding_col", type=str, default=None)
    parser.add_argument("--gpt2_model_name", type=str, default=None)
    parser.add_argument("--include_task_keys", nargs="+", default=None)
    parser.add_argument("--train_task_keys", nargs="+", default=None)
    parser.add_argument("--test_task_keys", nargs="+", default=None)
    parser.add_argument("--log1p_task_keys", nargs="+", default=None)
    parser.add_argument("--drop_nan_targets", action="store_true")
    parser.add_argument("--routing_context_cols", nargs="+", default=None)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_test_samples", type=int, default=None)
    parser.add_argument("--max_samples_per_task", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default="eval_outputs_checkpoints")
    parser.add_argument("--use_satellite_only_lite", action="store_true")
    parser.add_argument("--use_semantic_neighbor_router", action="store_true")
    parser.add_argument("--semantic_neighbor_top_k", type=int, default=None)
    parser.add_argument("--semantic_neighbor_temperature", type=float, default=None)
    parser.add_argument("--semantic_neighbor_mix_weight", type=float, default=None)
    parser.add_argument("--use_task_adaptive_semantic_mix", action="store_true")
    parser.add_argument("--use_cluster_prototype_router", action="store_true")
    parser.add_argument("--num_cluster_prototypes", type=int, default=None)
    parser.add_argument("--prototype_temperature", type=float, default=None)
    parser.add_argument("--prototype_mix_weight", type=float, default=None)
    parser.add_argument("--use_backbone_residual_head", action="store_true")
    parser.add_argument("--use_backbone_residual_inference", action="store_true")
    parser.add_argument("--backbone_residual_weight", type=float, default=None)
    parser.add_argument("--residual_reg_loss_weight", type=float, default=None)
    parser.add_argument("--use_shared_opportunity_expert", action="store_true")
    parser.add_argument("--shared_opportunity_init_weight", type=float, default=None)
    parser.add_argument("--debug_forward", action="store_true")
    parser.add_argument("--debug_batches", type=int, default=1)
    parser.add_argument("--disable_export_predictions", action="store_true")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def resolve_checkpoint_paths(args) -> List[str]:
    paths: List[str] = []
    if args.checkpoint_paths:
        paths.extend(args.checkpoint_paths)
    checkpoint_dir = args.checkpoint_dir.strip()
    if checkpoint_dir:
        if args.include_best:
            paths.append(str(Path(checkpoint_dir) / "best_model.pt"))
        for epoch in args.epoch_numbers or []:
            paths.append(str(Path(checkpoint_dir) / f"epoch_{int(epoch):03d}.pt"))
    deduped = []
    seen = set()
    for path in paths:
        if path not in seen:
            deduped.append(path)
            seen.add(path)
    if not deduped:
        raise ValueError("No checkpoints specified. Use --checkpoint_paths or --checkpoint_dir with --include_best/--epoch_numbers.")
    return deduped


def summarize_metrics(checkpoint_name: str, metrics: Dict) -> Dict[str, float]:
    row = {
        "checkpoint": checkpoint_name,
        "grid_mse": float(metrics.get("grid_mse", 0.0)),
        "grid_mae": float(metrics.get("grid_mae", 0.0)),
        "city_mse": float(metrics.get("city_mse", 0.0)),
        "city_mae": float(metrics.get("city_mae", 0.0)),
        "country_mse": float(metrics.get("country_mse", 0.0)),
        "country_mae": float(metrics.get("country_mae", 0.0)),
    }
    per_target = metrics.get("per_target", {})
    for task_name, task_metrics in per_target.items():
        row[f"{task_name}_mse"] = float(task_metrics.get("mse", 0.0))
        row[f"{task_name}_mae"] = float(task_metrics.get("mae", 0.0))
        row[f"{task_name}_count"] = int(task_metrics.get("count", 0))
    return row


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)

    checkpoint_paths = resolve_checkpoint_paths(args)
    summary_rows = []

    for checkpoint_path in checkpoint_paths:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        checkpoint_config = checkpoint["config"]
        config = build_config_from_checkpoint(checkpoint_config, args)
        config.save_dir = os.path.join(args.save_dir, Path(checkpoint_path).stem)
        os.makedirs(config.save_dir, exist_ok=True)

        train_loader, test_loader, schema = build_dataloaders(config)
        model = UrbanOME(config).to(device)
        initialize_prompt_encoder_for_loading(model, config, device)
        model.load_state_dict(checkpoint["model_state_dict"])

        metrics = evaluate(model, test_loader, config, device)
        rounded_metrics = round_nested_metrics(metrics, decimals=3)
        checkpoint_name = Path(checkpoint_path).name
        summary_rows.append(summarize_metrics(checkpoint_name, rounded_metrics))

        print(f"checkpoint={checkpoint_name}")
        for line in format_metrics_table(rounded_metrics):
            print(line)

        if not args.disable_export_predictions:
            prediction_csv = os.path.join(config.save_dir, "test_predictions.csv")
            city_prediction_csv = os.path.join(config.save_dir, "city_predictions.csv")
            country_prediction_csv = os.path.join(config.save_dir, "country_predictions.csv")
            export_predictions(model, test_loader, config, device, prediction_csv)
            export_group_predictions(prediction_csv, city_prediction_csv, country_prediction_csv)
            print("prediction_csv=", prediction_csv)
            print("city_prediction_csv=", city_prediction_csv)
            print("country_prediction_csv=", country_prediction_csv)

    summary_csv = os.path.join(args.save_dir, "checkpoint_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False)
    print("checkpoint_summary_csv=", summary_csv)


if __name__ == "__main__":
    main()
