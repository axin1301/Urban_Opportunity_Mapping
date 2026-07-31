import argparse
import json
import os

import torch

from urbanome import UrbanOME, UrbanOMEConfig
from urbanome.data import (
    build_dataloaders,
    build_train_auto_val_test_dataloaders,
    build_train_val_test_dataloaders,
)
from urbanome.trainer import (
    adapt_test_time,
    evaluate,
    export_group_predictions,
    export_predictions,
    train_one_epoch,
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


def print_prompt_audit(config: UrbanOMEConfig, schema) -> None:
    print("Prompt encoder type:", config.prompt_encoder_type)
    print("Has external task embeddings:", schema.task_prompt_embeddings is not None)
    if config.prompt_encoder_type == "external_embedding":
        print("External task prompt embeddings path:", config.external_task_prompt_embeddings_path or "(table column)")
        if schema.task_prompt_embeddings:
            print("External prompt embedding audit:")
            for task_name in schema.target_cols:
                vector = schema.task_prompt_embeddings.get(task_name)
                if vector is None:
                    print(f"{task_name}: missing_external_embedding")
                    continue
                tensor = torch.tensor(vector, dtype=torch.float32)
                print(
                    f"{task_name}: embedding_dim={tensor.numel()} "
                    f"embedding_norm={float(torch.linalg.vector_norm(tensor)):.6f}"
                )
        print("Text prompts shown below are from train/test tables for inspection only.")
        print("When using external_embedding, training uses the external vectors above.")
    print("Task prompt text audit:")
    for task_name in schema.target_cols:
        prompt = schema.task_prompt_templates.get(task_name, "")
        print(f"\n[{task_name}]")
        print(prompt if prompt else "(empty)")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_csv", type=str, default="train.csv")
    parser.add_argument("--val_csv", type=str, default="")
    parser.add_argument("--test_csv", type=str, default="test.csv")
    parser.add_argument("--split_train_as_val", action="store_true")
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--val_split_by", type=str, default="city", choices=["city", "country"])
    parser.add_argument("--image_root", type=str, default="images")
    parser.add_argument("--satellite_path_template", type=str, default="{image_name}")
    parser.add_argument("--street_view_path_template", type=str, default="{image_name}")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--tta_learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--image_encoder_type", type=str, default="resnet18")
    parser.add_argument("--street_view_pooling", type=str, default="attention")
    parser.add_argument("--prompt_encoder_type", type=str, default="lightweight")
    parser.add_argument("--external_task_prompt_embeddings_path", type=str, default="")
    parser.add_argument("--task_prompt_embedding_col", type=str, default="task_prompt_embedding")
    parser.add_argument("--gpt2_model_name", type=str, default="gpt2")
    parser.add_argument("--disable_pretrained_image_encoder", action="store_true")
    parser.add_argument("--unfreeze_image_backbone", action="store_true")
    parser.add_argument("--disable_satellite", action="store_true")
    parser.add_argument("--disable_street_view", action="store_true")
    parser.add_argument("--disable_tabular", action="store_true")
    parser.add_argument("--tta_steps", type=int, default=1)
    parser.add_argument("--resume_checkpoint", type=str, default="")
    parser.add_argument("--pseudo_label_threshold", type=float, default=0.65)
    parser.add_argument("--save_dir", type=str, default="outputs")
    parser.add_argument("--checkpoint_metric", type=str, default="city_mse")
    parser.add_argument("--consistency_loss_weight", type=float, default=0.0)
    parser.add_argument("--load_balance_loss_weight", type=float, default=0.0)
    parser.add_argument("--region_aux_loss_weight", type=float, default=0.0)
    parser.add_argument("--income_aux_loss_weight", type=float, default=0.0)
    parser.add_argument(
        "--task_loss_weights_json",
        type=str,
        default="",
        help='JSON string like {"target_a": 2.0, "target_b": 0.5}',
    )
    parser.add_argument("--disable_loss_normalization", action="store_true")
    parser.add_argument("--include_task_keys", nargs="+", default=None)
    parser.add_argument("--train_task_keys", nargs="+", default=None)
    parser.add_argument("--test_task_keys", nargs="+", default=None)
    parser.add_argument("--meta_val_task_keys", nargs="+", default=None)
    parser.add_argument("--log1p_task_keys", nargs="+", default=None)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    parser.add_argument("--max_samples_per_task", type=int, default=0)
    parser.add_argument("--num_task_experts", type=int, default=6)
    parser.add_argument("--num_task_experts_per_cluster", type=int, default=2)
    parser.add_argument("--top_k_task_cluster", type=int, default=1)
    parser.add_argument("--router_temperature", type=float, default=1.0)
    parser.add_argument("--expert_diversity_loss_weight", type=float, default=0.0)
    parser.add_argument("--route_entropy_loss_weight", type=float, default=0.0)
    parser.add_argument("--task_cluster_consistency_loss_weight", type=float, default=0.0)
    parser.add_argument("--task_cluster_separation_margin", type=float, default=0.25)
    parser.add_argument("--unseen_task_alignment_loss_weight", type=float, default=0.0)
    parser.add_argument("--unseen_task_router_distill_loss_weight", type=float, default=0.0)
    parser.add_argument("--task_mask_meta_train", action="store_true")
    parser.add_argument("--task_mask_ratio", type=float, default=0.3)
    parser.add_argument("--min_masked_tasks", type=int, default=1)
    parser.add_argument("--max_masked_tasks", type=int, default=2)
    parser.add_argument("--task_mask_loss_weight", type=float, default=0.1)
    parser.add_argument("--task_cluster_map_json", type=str, default="")
    parser.add_argument(
        "--task_context_scope",
        type=str,
        default="intra_cluster",
        choices=["global", "intra_cluster", "none"],
    )
    parser.add_argument("--task_context_mix_weight", type=float, default=0.2)
    parser.add_argument(
        "--task_context_top_k",
        type=int,
        default=1,
        help="Keep only top-k most similar task neighbors for task context aggregation; <=0 uses dense weighting.",
    )
    parser.add_argument("--use_satellite_only_lite", action="store_true")
    parser.add_argument("--use_semantic_neighbor_router", action="store_true")
    parser.add_argument("--semantic_neighbor_top_k", type=int, default=2)
    parser.add_argument("--semantic_neighbor_temperature", type=float, default=0.7)
    parser.add_argument("--semantic_neighbor_mix_weight", type=float, default=0.3)
    parser.add_argument("--use_task_adaptive_semantic_mix", action="store_true")
    parser.add_argument("--use_cluster_prototype_router", action="store_true")
    parser.add_argument("--num_cluster_prototypes", type=int, default=4)
    parser.add_argument("--prototype_temperature", type=float, default=0.7)
    parser.add_argument("--prototype_mix_weight", type=float, default=0.3)
    parser.add_argument("--use_backbone_residual_head", action="store_true")
    parser.add_argument("--use_backbone_residual_inference", action="store_true")
    parser.add_argument("--backbone_residual_weight", type=float, default=0.5)
    parser.add_argument("--residual_reg_loss_weight", type=float, default=0.0)
    parser.add_argument("--use_shared_opportunity_expert", action="store_true")
    parser.add_argument("--shared_opportunity_init_weight", type=float, default=0.25)
    parser.add_argument("--debug_forward", action="store_true")
    parser.add_argument("--debug_batches", type=int, default=1)
    parser.add_argument("--disable_tqdm", action="store_true")
    parser.add_argument("--save_every_n_epochs", type=int, default=0)
    parser.add_argument("--early_stop_patience", type=int, default=0)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--satellite_col", type=str, default="sat_image_name")
    parser.add_argument("--satellite_embedding_col", type=str, default="satellite_embedding")
    parser.add_argument("--use_cached_satellite_embeddings", action="store_true")
    parser.add_argument("--task_key_col", type=str, default="indicator_column")
    parser.add_argument("--task_name_col", type=str, default="task_name")
    parser.add_argument("--target_value_col", type=str, default="indicator_value")
    parser.add_argument("--drop_nan_targets", action="store_true")
    parser.add_argument("--text_prompt_col", type=str, default="generated_prompt")
    parser.add_argument("--definition_col", type=str, default="definition")
    parser.add_argument("--routing_context_cols", nargs="+", default=None)
    parser.add_argument("--target_start_col", type=str, default="popu")
    parser.add_argument(
        "--tabular_feature_cols",
        nargs="+",
        default=["popu"],
    )
    parser.add_argument(
        "--street_view_cols",
        nargs="+",
        default=["stv0", "stv1", "stv2", "stv3"],
    )
    parser.add_argument(
        "--exclude_target_cols",
        nargs="+",
        default=[],
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()
    task_loss_weights = json.loads(args.task_loss_weights_json) if args.task_loss_weights_json else None
    task_cluster_map = json.loads(args.task_cluster_map_json) if args.task_cluster_map_json else None

    config = UrbanOMEConfig(
        train_csv=args.train_csv,
        val_csv=args.val_csv,
        test_csv=args.test_csv,
        split_train_as_val=args.split_train_as_val,
        val_ratio=args.val_ratio,
        val_split_by=args.val_split_by,
        image_root=args.image_root,
        satellite_path_template=args.satellite_path_template,
        street_view_path_template=args.street_view_path_template,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        random_seed=args.random_seed,
        learning_rate=args.learning_rate,
        tta_lr=args.tta_learning_rate,
        weight_decay=args.weight_decay,
        image_size=args.image_size,
        image_encoder_type=args.image_encoder_type,
        street_view_pooling=args.street_view_pooling,
        prompt_encoder_type=args.prompt_encoder_type,
        external_task_prompt_embeddings_path=args.external_task_prompt_embeddings_path,
        task_prompt_embedding_col=args.task_prompt_embedding_col,
        gpt2_model_name=args.gpt2_model_name,
        use_pretrained_image_encoder=not args.disable_pretrained_image_encoder,
        freeze_image_backbone=not args.unfreeze_image_backbone,
        use_satellite=not args.disable_satellite,
        use_street_view=not args.disable_street_view,
        use_tabular=not args.disable_tabular,
        pseudo_label_threshold=args.pseudo_label_threshold,
        save_dir=args.save_dir,
        checkpoint_metric=args.checkpoint_metric,
        consistency_loss_weight=args.consistency_loss_weight,
        load_balance_loss_weight=args.load_balance_loss_weight,
        region_aux_loss_weight=args.region_aux_loss_weight,
        income_aux_loss_weight=args.income_aux_loss_weight,
        task_loss_weights=task_loss_weights,
        normalize_task_loss=not args.disable_loss_normalization,
        include_task_keys=args.include_task_keys,
        train_task_keys=args.train_task_keys,
        test_task_keys=args.test_task_keys,
        meta_val_task_keys=args.meta_val_task_keys,
        log1p_task_keys=args.log1p_task_keys,
        max_train_samples=args.max_train_samples,
        max_test_samples=args.max_test_samples,
        max_samples_per_task=args.max_samples_per_task,
        num_task_experts=args.num_task_experts,
        num_task_experts_per_cluster=args.num_task_experts_per_cluster,
        top_k_task_cluster=args.top_k_task_cluster,
        debug_forward=args.debug_forward,
        debug_batches=args.debug_batches,
        show_progress=not args.disable_tqdm,
        save_every_n_epochs=args.save_every_n_epochs,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        router_temperature=args.router_temperature,
        expert_diversity_loss_weight=args.expert_diversity_loss_weight,
        route_entropy_loss_weight=args.route_entropy_loss_weight,
        task_cluster_consistency_loss_weight=args.task_cluster_consistency_loss_weight,
        task_cluster_separation_margin=args.task_cluster_separation_margin,
        unseen_task_alignment_loss_weight=args.unseen_task_alignment_loss_weight,
        unseen_task_router_distill_loss_weight=args.unseen_task_router_distill_loss_weight,
        task_mask_meta_train=args.task_mask_meta_train,
        task_mask_ratio=args.task_mask_ratio,
        min_masked_tasks=args.min_masked_tasks,
        max_masked_tasks=args.max_masked_tasks,
        task_mask_loss_weight=args.task_mask_loss_weight,
        task_context_scope=args.task_context_scope,
        task_context_mix_weight=args.task_context_mix_weight,
        task_context_top_k=args.task_context_top_k,
        use_satellite_only_lite=args.use_satellite_only_lite,
        use_semantic_neighbor_router=args.use_semantic_neighbor_router,
        semantic_neighbor_top_k=args.semantic_neighbor_top_k,
        semantic_neighbor_temperature=args.semantic_neighbor_temperature,
        semantic_neighbor_mix_weight=args.semantic_neighbor_mix_weight,
        use_task_adaptive_semantic_mix=args.use_task_adaptive_semantic_mix,
        use_cluster_prototype_router=args.use_cluster_prototype_router,
        num_cluster_prototypes=args.num_cluster_prototypes,
        prototype_temperature=args.prototype_temperature,
        prototype_mix_weight=args.prototype_mix_weight,
        use_backbone_residual_head=args.use_backbone_residual_head,
        use_backbone_residual_inference=args.use_backbone_residual_inference,
        backbone_residual_weight=args.backbone_residual_weight,
        residual_reg_loss_weight=args.residual_reg_loss_weight,
        use_shared_opportunity_expert=args.use_shared_opportunity_expert,
        shared_opportunity_init_weight=args.shared_opportunity_init_weight,
        task_cluster_map=task_cluster_map,
        satellite_col=args.satellite_col,
        satellite_embedding_col=args.satellite_embedding_col,
        use_cached_satellite_embeddings=args.use_cached_satellite_embeddings,
        task_key_col=args.task_key_col,
        task_name_col=args.task_name_col,
        target_value_col=args.target_value_col,
        drop_nan_targets=args.drop_nan_targets,
        text_prompt_col=args.text_prompt_col,
        definition_col=args.definition_col,
        routing_context_cols=args.routing_context_cols,
        target_start_col=args.target_start_col,
        tabular_feature_cols=args.tabular_feature_cols,
        street_view_cols=args.street_view_cols,
        exclude_target_cols=args.exclude_target_cols,
    )
    if config.use_satellite_only_lite:
        config.expert_diversity_loss_weight = 0.0
    os.makedirs(config.save_dir, exist_ok=True)

    if config.val_csv:
        train_loader, val_loader, test_loader, schema = build_train_val_test_dataloaders(config)
    elif config.split_train_as_val:
        train_loader, val_loader, test_loader, schema = build_train_auto_val_test_dataloaders(config)
    else:
        train_loader, test_loader, schema = build_dataloaders(config)
        val_loader = None
    device = torch.device(args.device)
    model = UrbanOME(config).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    tta_optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.tta_lr,
        weight_decay=config.weight_decay,
    )
    best_metric = float("inf")
    best_ckpt_path = os.path.join(config.save_dir, "best_model.pt")
    best_metrics_path = os.path.join(config.save_dir, "best_metrics.json")
    start_epoch = 0
    start_tta_step = 0
    epochs_without_improvement = 0

    initialize_prompt_encoder_for_loading(model, config, device)

    if args.resume_checkpoint:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

        if checkpoint.get("optimizer_state_dict") is not None:
            if checkpoint.get("tta_step") is not None:
                tta_optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_tta_step = int(checkpoint.get("tta_step", 0))
            else:
                optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                start_epoch = int(checkpoint.get("epoch", 0))

        metrics = checkpoint.get("metrics")
        if metrics is not None and config.checkpoint_metric in metrics:
            best_metric = float(metrics[config.checkpoint_metric])

        print("Resume checkpoint:", args.resume_checkpoint)
        print("Resume epoch:", checkpoint.get("epoch"))
        print("Resume tta_step:", checkpoint.get("tta_step"))
        print("Resume best_metric:", best_metric)

    print("Target columns:", schema.target_cols)
    if config.val_csv:
        print("Validation CSV:", config.val_csv)
    elif config.split_train_as_val:
        print(
            "Validation split:",
            json.dumps(
                {
                    "source": "train_csv",
                    "val_ratio": config.val_ratio,
                    "val_split_by": config.val_split_by,
                }
            ),
        )
    else:
        print("Validation CSV:", "(not provided; using test for model selection)")
    print("Routing context columns:", schema.routing_context_cols)
    print("Long format dataset:", schema.long_format)
    print("Satellite path template:", config.satellite_path_template)
    print("Use cached satellite embeddings:", config.use_cached_satellite_embeddings)
    print("Included task keys:", config.include_task_keys)
    print("Train task keys:", config.train_task_keys)
    print("Test task keys:", config.test_task_keys)
    print("Meta-val task keys:", config.meta_val_task_keys)
    print("Log1p task keys:", config.log1p_task_keys)
    print("Num task experts:", config.num_task_experts)
    print("Num task experts per cluster:", config.num_task_experts_per_cluster)
    print(
        "Loss weights:",
        json.dumps(
            {
                "consistency": config.consistency_loss_weight,
                "load_balance": config.load_balance_loss_weight,
                "region_aux": config.region_aux_loss_weight,
                "income_aux": config.income_aux_loss_weight,
                "expert_diversity": config.expert_diversity_loss_weight,
                "route_entropy": config.route_entropy_loss_weight,
                "task_cluster_consistency": config.task_cluster_consistency_loss_weight,
                "unseen_task_alignment": config.unseen_task_alignment_loss_weight,
                "unseen_task_router_distill": config.unseen_task_router_distill_loss_weight,
                "task_mask_meta": config.task_mask_loss_weight if config.task_mask_meta_train else 0.0,
            }
        ),
    )
    print("Task cluster separation margin:", config.task_cluster_separation_margin)
    print("Task cluster map override:", json.dumps(config.task_cluster_map, ensure_ascii=False))
    print("Task context scope:", config.task_context_scope)
    print("Task context mix weight:", config.task_context_mix_weight)
    print("Task context top-k:", config.task_context_top_k)
    print("Early stop patience:", config.early_stop_patience)
    print("Early stop min delta:", config.early_stop_min_delta)
    print("Max train/test samples:", config.max_train_samples, config.max_test_samples)
    print("Max samples per task:", config.max_samples_per_task)
    print_prompt_audit(config, schema)
    print("Debug forward:", config.debug_forward, "Debug batches:", config.debug_batches)
    print(
        "Enabled modalities:",
        json.dumps(
            {
                "satellite": config.use_satellite,
                "street_view": config.use_street_view,
                "tabular": config.use_tabular,
            }
        ),
    )
    print("Train/Test split assumption: countries and cities are disjoint across CSV files.")

    for epoch in range(start_epoch, config.num_epochs):
        print(f"epoch_start={epoch + 1}/{config.num_epochs}")
        train_metrics = train_one_epoch(model, train_loader, optimizer, config, device)
        selection_loader = val_loader if val_loader is not None else test_loader
        selection_metrics = evaluate(model, selection_loader, config, device)
        current_metric = selection_metrics[config.checkpoint_metric]
        if current_metric < (best_metric - config.early_stop_min_delta):
            best_metric = current_metric
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config.__dict__,
                    "metrics": selection_metrics,
                },
                best_ckpt_path,
            )
            with open(best_metrics_path, "w", encoding="utf-8") as f:
                json.dump(selection_metrics, f, indent=2)
        else:
            epochs_without_improvement += 1
        if config.save_every_n_epochs > 0 and ((epoch + 1) % config.save_every_n_epochs == 0):
            epoch_ckpt_path = os.path.join(config.save_dir, f"epoch_{epoch + 1:03d}.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": config.__dict__,
                    "metrics": selection_metrics,
                },
                epoch_ckpt_path,
            )
        if val_loader is not None:
            print(
                f"epoch={epoch + 1} "
                f"train_loss={train_metrics['train_loss']:.6f} "
                f"val_grid_mse={selection_metrics['grid_mse']:.6f} "
                f"val_city_mse={selection_metrics['city_mse']:.6f} "
                f"val_country_mse={selection_metrics['country_mse']:.6f}"
            )
            print("val_per_target:", json.dumps(selection_metrics["per_target"], ensure_ascii=False))
        else:
            print(
                f"epoch={epoch + 1} "
                f"train_loss={train_metrics['train_loss']:.6f} "
                f"grid_mse={selection_metrics['grid_mse']:.6f} "
                f"city_mse={selection_metrics['city_mse']:.6f} "
                f"country_mse={selection_metrics['country_mse']:.6f}"
            )
            print("per_target:", json.dumps(selection_metrics["per_target"], ensure_ascii=False))
        if config.early_stop_patience > 0 and epochs_without_improvement >= config.early_stop_patience:
            print(
                f"early_stopping_triggered epoch={epoch + 1} "
                f"best_metric={best_metric:.6f} "
                f"epochs_without_improvement={epochs_without_improvement}"
            )
            break

    for step in range(start_tta_step, args.tta_steps):
        print(f"tta_start={step + 1}/{args.tta_steps}")
        tta_metrics = adapt_test_time(model, test_loader, tta_optimizer, config, device)
        selection_loader = val_loader if val_loader is not None else test_loader
        selection_metrics = evaluate(model, selection_loader, config, device)
        current_metric = selection_metrics[config.checkpoint_metric]
        if current_metric < best_metric:
            best_metric = current_metric
            torch.save(
                {
                    "tta_step": step + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": tta_optimizer.state_dict(),
                    "config": config.__dict__,
                    "metrics": selection_metrics,
                },
                best_ckpt_path,
            )
            with open(best_metrics_path, "w", encoding="utf-8") as f:
                json.dump(selection_metrics, f, indent=2)
        if val_loader is not None:
            print(
                f"tta_step={step + 1} "
                f"tta_loss={tta_metrics['tta_loss']:.6f} "
                f"stable_ratio={tta_metrics['stable_ratio']:.6f} "
                f"val_grid_mse={selection_metrics['grid_mse']:.6f} "
                f"val_city_mse={selection_metrics['city_mse']:.6f} "
                f"val_country_mse={selection_metrics['country_mse']:.6f}"
            )
            print("val_per_target:", json.dumps(selection_metrics["per_target"], ensure_ascii=False))
        else:
            print(
                f"tta_step={step + 1} "
                f"tta_loss={tta_metrics['tta_loss']:.6f} "
                f"stable_ratio={tta_metrics['stable_ratio']:.6f} "
                f"grid_mse={selection_metrics['grid_mse']:.6f} "
                f"city_mse={selection_metrics['city_mse']:.6f} "
                f"country_mse={selection_metrics['country_mse']:.6f}"
            )
            print("per_target:", json.dumps(selection_metrics["per_target"], ensure_ascii=False))

    if os.path.exists(best_ckpt_path):
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    final_metrics = evaluate(model, test_loader, config, device)
    export_predictions(
        model,
        test_loader,
        config,
        device,
        os.path.join(config.save_dir, "test_predictions.csv"),
    )
    export_group_predictions(
        os.path.join(config.save_dir, "test_predictions.csv"),
        os.path.join(config.save_dir, "city_predictions.csv"),
        os.path.join(config.save_dir, "country_predictions.csv"),
    )
    with open(os.path.join(config.save_dir, "final_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(final_metrics, f, indent=2)
    print(f"best_checkpoint={best_ckpt_path}")
    print(f"prediction_csv={os.path.join(config.save_dir, 'test_predictions.csv')}")
    print(f"city_prediction_csv={os.path.join(config.save_dir, 'city_predictions.csv')}")
    print(f"country_prediction_csv={os.path.join(config.save_dir, 'country_predictions.csv')}")


if __name__ == "__main__":
    main()
