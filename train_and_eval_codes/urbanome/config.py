from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class UrbanOMEConfig:
    hidden_dim: int = 256
    expert_dim: int = 256
    dropout: float = 0.1

    satellite_in_dim: int = 3
    street_view_in_dim: int = 3
    tabular_in_dim: int = 1
    routing_context_dim: int = 1
    image_size: int = 224
    image_encoder_type: str = "resnet18"
    street_view_pooling: str = "attention"
    use_pretrained_image_encoder: bool = True
    freeze_image_backbone: bool = True
    use_satellite: bool = True
    use_street_view: bool = True
    use_tabular: bool = True
    use_cached_satellite_embeddings: bool = False

    train_csv: str = "train.csv"
    val_csv: str = ""
    test_csv: str = "test.csv"
    split_train_as_val: bool = False
    val_ratio: float = 0.1
    val_split_by: str = "city"
    image_root: str = "images"
    satellite_path_template: str = "{image_name}"
    street_view_path_template: str = "{image_name}"

    country_col: str = "country"
    city_col: str = "city"
    grid_col: str = "grid"
    region_col: str = "region"
    income_level_col: str = "income_level"
    hemisphere_col: str = "hemisphere"
    satellite_col: str = "sat_image_name"
    satellite_embedding_col: str = "satellite_embedding"
    street_view_cols: List[str] = field(
        default_factory=lambda: ["stv0", "stv1", "stv2", "stv3"]
    )
    tabular_feature_cols: List[str] = field(
        default_factory=lambda: ["popu"]
    )
    routing_context_cols: Optional[List[str]] = None
    target_start_col: str = "popu"
    exclude_target_cols: List[str] = field(default_factory=list)
    text_prompt_col: str = "generated_prompt"
    task_key_col: str = "indicator_column"
    task_name_col: str = "task_name"
    target_value_col: str = "indicator_value"
    drop_nan_targets: bool = False
    definition_col: str = "definition"
    default_task_prompt_template: str = (
        "Task: Predict {task}. "
        "Definition: Estimate the urban opportunity indicator for {task}. "
        "Relevant evidence: satellite pattern, street-view scene, population context, and urban structure. "
        "Expected output: normalized opportunity score."
    )

    modality_names: List[str] = field(
        default_factory=lambda: ["satellite", "street_view", "tabular"]
    )
    region_names: List[str] = field(
        default_factory=lambda: [
            "continent",
            "income_level",
            "morphology_cluster",
        ]
    )
    task_names: List[str] = field(
        default_factory=lambda: [
            "accessibility",
            "health",
            "education",
            "climate",
            "mobility",
            "social_welfare",
        ]
    )

    top_k_modality: int = 2
    top_k_region: int = 2
    top_k_task: int = 2
    top_k_task_cluster: int = 1
    num_task_experts: int = 6
    num_task_experts_per_cluster: int = 2

    stability_temperature: float = 0.2
    consistency_margin: float = 0.15
    pseudo_label_threshold: float = 0.65
    tta_lr: float = 1e-4
    batch_size: int = 64
    num_epochs: int = 30
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    train_shuffle: bool = True
    random_seed: int = 42
    save_dir: str = "outputs"
    checkpoint_metric: str = "city_mse"
    task_loss_weights: Optional[Dict[str, float]] = None
    normalize_task_loss: bool = True
    consistency_loss_weight: float = 0.0
    load_balance_loss_weight: float = 0.0
    region_aux_loss_weight: float = 0.0
    income_aux_loss_weight: float = 0.0
    expert_diversity_loss_weight: float = 0.0
    route_entropy_loss_weight: float = 0.0
    task_cluster_consistency_loss_weight: float = 0.0
    task_cluster_separation_margin: float = 0.25
    unseen_task_alignment_loss_weight: float = 0.0
    unseen_task_router_distill_loss_weight: float = 0.0
    task_mask_meta_train: bool = False
    task_mask_ratio: float = 0.3
    min_masked_tasks: int = 1
    max_masked_tasks: int = 2
    task_mask_loss_weight: float = 0.1
    use_task_context: bool = True
    task_context_temperature: float = 0.7
    task_context_mix_weight: float = 0.5
    task_context_scope: str = "intra_cluster"
    task_context_top_k: int = 0
    use_satellite_only_lite: bool = False
    use_prompt_hyper_head: bool = True
    use_semantic_neighbor_router: bool = False
    semantic_neighbor_top_k: int = 2
    semantic_neighbor_temperature: float = 0.7
    semantic_neighbor_mix_weight: float = 0.3
    use_task_adaptive_semantic_mix: bool = False
    use_cluster_prototype_router: bool = False
    num_cluster_prototypes: int = 4
    prototype_temperature: float = 0.7
    prototype_mix_weight: float = 0.3
    use_backbone_residual_head: bool = False
    use_backbone_residual_inference: bool = False
    backbone_residual_weight: float = 0.5
    residual_reg_loss_weight: float = 0.0
    use_shared_opportunity_expert: bool = False
    shared_opportunity_init_weight: float = 0.25
    router_temperature: float = 1.0
    prompt_vocab_size: int = 4096
    prompt_embedding_dim: int = 256
    prompt_encoder_type: str = "lightweight"
    external_task_prompt_embeddings_path: str = ""
    task_prompt_embedding_col: str = "task_prompt_embedding"
    gpt2_model_name: str = "gpt2"
    task_prompt_templates: Optional[Dict[str, str]] = None
    task_prompt_embeddings: Optional[Dict[str, List[float]]] = None
    task_cluster_map: Optional[Dict[str, str]] = None
    fixed_eval_schema: bool = False
    include_task_keys: Optional[List[str]] = None
    train_task_keys: Optional[List[str]] = None
    test_task_keys: Optional[List[str]] = None
    meta_val_task_keys: Optional[List[str]] = None
    log1p_task_keys: Optional[List[str]] = None
    max_train_samples: int = 0
    max_test_samples: int = 0
    max_samples_per_task: int = 0
    debug_forward: bool = False
    debug_batches: int = 1
    show_progress: bool = True
    save_every_n_epochs: int = 0
    early_stop_patience: int = 0
    early_stop_min_delta: float = 0.0

    task_output_dims: Dict[str, int] = field(default_factory=dict)
    country_label_names: List[str] = field(default_factory=list)
    region_label_names: List[str] = field(default_factory=list)
    income_label_names: List[str] = field(default_factory=list)
    hemisphere_label_names: List[str] = field(default_factory=list)
