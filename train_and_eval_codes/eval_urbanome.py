import argparse
import json
import os

import torch

from urbanome import UrbanOME, UrbanOMEConfig
from urbanome.data import build_dataloaders
from urbanome.trainer import (
    adapt_test_time,
    evaluate,
    export_group_predictions,
    export_predictions,
    format_metrics_table,
    round_nested_metrics,
)


def initialize_prompt_encoder_for_loading(model: UrbanOME, config: UrbanOMEConfig, device: torch.device):
    if config.prompt_encoder_type != "external_embedding":
        return
    if not config.task_prompt_embeddings:
        return
    first_vector = next(iter(config.task_prompt_embeddings.values()), None)
    if not first_vector:
        return
    ensure_proj = getattr(model.prompt_encoder, "ensure_proj", None)
    if ensure_proj is not None:
        ensure_proj(len(first_vector), device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--train_csv", type=str, default=None)
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
    parser.add_argument("--save_dir", type=str, default="eval_outputs")
    parser.add_argument("--tta_steps", type=int, default=0)
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
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def build_config_from_checkpoint(checkpoint_config: dict, args) -> UrbanOMEConfig:
    config = UrbanOMEConfig(**checkpoint_config)
    config.fixed_eval_schema = True
    config.test_csv = args.test_csv
    if args.train_csv is not None:
        config.train_csv = args.train_csv
    if args.image_root is not None:
        config.image_root = args.image_root
    if args.satellite_path_template is not None:
        config.satellite_path_template = args.satellite_path_template
    if args.street_view_path_template is not None:
        config.street_view_path_template = args.street_view_path_template
    if args.satellite_embedding_col is not None:
        config.satellite_embedding_col = args.satellite_embedding_col
    if args.use_cached_satellite_embeddings:
        config.use_cached_satellite_embeddings = True
    if args.prompt_encoder_type is not None:
        config.prompt_encoder_type = args.prompt_encoder_type
    if args.external_task_prompt_embeddings_path is not None:
        config.external_task_prompt_embeddings_path = args.external_task_prompt_embeddings_path
    if args.task_prompt_embedding_col is not None:
        config.task_prompt_embedding_col = args.task_prompt_embedding_col
    if args.gpt2_model_name is not None:
        config.gpt2_model_name = args.gpt2_model_name
    if args.include_task_keys is not None:
        config.include_task_keys = args.include_task_keys
    if args.train_task_keys is not None:
        config.train_task_keys = args.train_task_keys
    if args.test_task_keys is not None:
        config.test_task_keys = args.test_task_keys
    if args.log1p_task_keys is not None:
        config.log1p_task_keys = args.log1p_task_keys
    if getattr(args, "drop_nan_targets", False):
        config.drop_nan_targets = True
    if args.routing_context_cols is not None:
        config.routing_context_cols = args.routing_context_cols
    if args.max_train_samples is not None:
        config.max_train_samples = args.max_train_samples
    if args.max_test_samples is not None:
        config.max_test_samples = args.max_test_samples
    if args.max_samples_per_task is not None:
        config.max_samples_per_task = args.max_samples_per_task
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if getattr(args, "use_satellite_only_lite", False):
        config.use_satellite_only_lite = True
        config.expert_diversity_loss_weight = 0.0
    if getattr(args, "use_semantic_neighbor_router", False):
        config.use_semantic_neighbor_router = True
    if getattr(args, "semantic_neighbor_top_k", None) is not None:
        config.semantic_neighbor_top_k = args.semantic_neighbor_top_k
    if getattr(args, "semantic_neighbor_temperature", None) is not None:
        config.semantic_neighbor_temperature = args.semantic_neighbor_temperature
    if getattr(args, "semantic_neighbor_mix_weight", None) is not None:
        config.semantic_neighbor_mix_weight = args.semantic_neighbor_mix_weight
    if getattr(args, "use_task_adaptive_semantic_mix", False):
        config.use_task_adaptive_semantic_mix = True
    if getattr(args, "use_cluster_prototype_router", False):
        config.use_cluster_prototype_router = True
    if getattr(args, "num_cluster_prototypes", None) is not None:
        config.num_cluster_prototypes = args.num_cluster_prototypes
    if getattr(args, "prototype_temperature", None) is not None:
        config.prototype_temperature = args.prototype_temperature
    if getattr(args, "prototype_mix_weight", None) is not None:
        config.prototype_mix_weight = args.prototype_mix_weight
    if getattr(args, "use_backbone_residual_head", False):
        config.use_backbone_residual_head = True
    if getattr(args, "use_backbone_residual_inference", False):
        config.use_backbone_residual_inference = True
    if getattr(args, "backbone_residual_weight", None) is not None:
        config.backbone_residual_weight = args.backbone_residual_weight
    if getattr(args, "residual_reg_loss_weight", None) is not None:
        config.residual_reg_loss_weight = args.residual_reg_loss_weight
    if getattr(args, "use_shared_opportunity_expert", False):
        config.use_shared_opportunity_expert = True
    if getattr(args, "shared_opportunity_init_weight", None) is not None:
        config.shared_opportunity_init_weight = args.shared_opportunity_init_weight
    config.save_dir = args.save_dir
    config.debug_forward = args.debug_forward
    config.debug_batches = args.debug_batches
    return config


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    device = torch.device(args.device)
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    checkpoint_config = checkpoint["config"]
    config = build_config_from_checkpoint(checkpoint_config, args)

    train_loader, test_loader, schema = build_dataloaders(config)
    model = UrbanOME(config).to(device)
    initialize_prompt_encoder_for_loading(model, config, device)
    model.load_state_dict(checkpoint["model_state_dict"])

    print("Loaded checkpoint:", args.checkpoint_path)
    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print("Checkpoint tta_step:", checkpoint.get("tta_step"))
    print("Train CSV:", config.train_csv)
    print("Test CSV:", config.test_csv)
    print("Target columns:", schema.target_cols)
    print("Use cached satellite embeddings:", config.use_cached_satellite_embeddings)
    print("Included task keys:", config.include_task_keys)
    print("Train task keys:", config.train_task_keys)
    print("Test task keys:", config.test_task_keys)
    print("Log1p task keys:", config.log1p_task_keys)
    print("Routing context columns:", schema.routing_context_cols)
    print("Batch size:", config.batch_size)
    print("Debug forward:", config.debug_forward, "Debug batches:", config.debug_batches)

    if args.tta_steps > 0:
        tta_optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.tta_lr,
            weight_decay=config.weight_decay,
        )
        for step in range(args.tta_steps):
            tta_metrics = adapt_test_time(model, test_loader, tta_optimizer, config, device)
            tta_metrics = round_nested_metrics(tta_metrics, decimals=3)
            print(
                f"tta_step={step + 1} "
                f"tta_loss={tta_metrics['tta_loss']:.3f} "
                f"stable_ratio={tta_metrics['stable_ratio']:.3f}"
            )

    final_metrics = evaluate(model, test_loader, config, device)
    rounded_final_metrics = round_nested_metrics(final_metrics, decimals=3)
    prediction_csv = os.path.join(config.save_dir, "test_predictions.csv")
    city_prediction_csv = os.path.join(config.save_dir, "city_predictions.csv")
    country_prediction_csv = os.path.join(config.save_dir, "country_predictions.csv")

    export_predictions(model, test_loader, config, device, prediction_csv)
    export_group_predictions(prediction_csv, city_prediction_csv, country_prediction_csv)
    for line in format_metrics_table(rounded_final_metrics):
        print(line)
    print("prediction_csv=", prediction_csv)
    print("city_prediction_csv=", city_prediction_csv)
    print("country_prediction_csv=", country_prediction_csv)


if __name__ == "__main__":
    main()
