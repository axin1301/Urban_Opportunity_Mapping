from typing import Dict

import torch
from torch import nn
import math

from .config import UrbanOMEConfig
from .encoders import MultiModalEncoders
from .experts import ExpertPool, FactorizedComposer
from .prompts import build_prompt_encoder
from .router import OpportunityRouter
from .stability import OpportunityStabilityScorer


def infer_task_cluster_map(task_names):
    cluster_map = {}
    for task_name in task_names:
        key = str(task_name).lower()
        if any(token in key for token in ["digital", "lte", "cell"]):
            cluster = "digital_cluster"
        elif any(token in key for token in ["social"]):
            cluster = "social_cluster"
        elif any(token in key for token in ["health", "food", "education", "mobility", "access"]):
            cluster = "access_cluster"
        elif any(token in key for token in ["population", "nighttime", "light"]):
            cluster = "population_cluster"
        else:
            cluster = "general_cluster"
        cluster_map[str(task_name)] = cluster
    return cluster_map


class UrbanOME(nn.Module):
    def __init__(self, config: UrbanOMEConfig):
        super().__init__()
        self.config = config
        self.task_cluster_map = (
            dict(config.task_cluster_map)
            if config.task_cluster_map
            else infer_task_cluster_map(config.task_names)
        )
        self.config.task_cluster_map = dict(self.task_cluster_map)
        self.task_cluster_names = list(dict.fromkeys(self.task_cluster_map[task_name] for task_name in config.task_names))

        self.encoders = MultiModalEncoders(
            satellite_in_dim=config.satellite_in_dim,
            street_view_in_dim=config.street_view_in_dim,
            tabular_in_dim=config.tabular_in_dim,
            hidden_dim=config.hidden_dim,
            image_encoder_type=config.image_encoder_type,
            street_view_pooling=config.street_view_pooling,
            use_pretrained_image_encoder=config.use_pretrained_image_encoder,
            freeze_image_backbone=config.freeze_image_backbone,
            use_satellite=config.use_satellite,
            use_street_view=config.use_street_view,
            use_tabular=config.use_tabular,
            dropout=config.dropout,
        )

        self.shared_proj = nn.Sequential(
            nn.Linear(config.hidden_dim * self.active_modality_count(config), config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        self.prompt_encoder = build_prompt_encoder(
            prompt_encoder_type=config.prompt_encoder_type,
            vocab_size=config.prompt_vocab_size,
            embedding_dim=config.prompt_embedding_dim,
            hidden_dim=config.hidden_dim,
            dropout=config.dropout,
            gpt2_model_name=config.gpt2_model_name,
        )
        self.prompt_to_feature = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        self.router_visual_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        self.router_task_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        self.semantic_neighbor_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        self.semantic_neighbor_gate = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.router_context_proj = nn.Sequential(
            nn.Linear(config.routing_context_dim + config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        self.cluster_name_to_index = {
            cluster_name: idx for idx, cluster_name in enumerate(self.task_cluster_names)
        }
        self.cluster_prototypes = nn.Parameter(
            torch.randn(
                len(self.task_cluster_names),
                config.num_cluster_prototypes,
                config.hidden_dim,
            )
        )
        self.prototype_query_proj = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        self.prototype_value_proj = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        self.prototype_router_fusion = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )

        self.router = OpportunityRouter(
            context_dim=config.hidden_dim,
            hidden_dim=config.hidden_dim,
            modality_names=config.modality_names,
            region_names=config.region_names,
            task_cluster_names=self.task_cluster_names,
            num_task_experts_per_cluster=config.num_task_experts_per_cluster,
            temperature=config.router_temperature,
            use_modality_routing=not config.use_satellite_only_lite,
        )
        self.task_cluster_expert_names = list(self.task_cluster_names)
        self.cluster_task_expert_names = {
            cluster_name: [
                f"{cluster_name}_expert_{idx}"
                for idx in range(config.num_task_experts_per_cluster)
            ]
            for cluster_name in self.task_cluster_names
        }

        self.modality_experts = ExpertPool(
            config.modality_names, config.hidden_dim, config.expert_dim, config.dropout
        )
        self.region_experts = ExpertPool(
            config.region_names, config.hidden_dim, config.expert_dim, config.dropout
        )
        self.task_cluster_experts = ExpertPool(
            self.task_cluster_expert_names, config.hidden_dim, config.expert_dim, config.dropout
        )
        self.task_experts = nn.ModuleDict(
            {
                cluster_name: ExpertPool(
                    expert_names,
                    config.hidden_dim,
                    config.expert_dim,
                    config.dropout,
                )
                for cluster_name, expert_names in self.cluster_task_expert_names.items()
            }
        )

        self.composer = FactorizedComposer(config.hidden_dim, config.dropout)
        self.film_generator = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 2),
            nn.LayerNorm(config.hidden_dim * 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim * 2),
        )
        self.task_context_gate = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.task_context_pred_proj = nn.Sequential(
            nn.Linear(1, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        self.hyper_head_generator = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, config.hidden_dim + 1),
        )
        self.hyper_head_mixer = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )
        self.shared_regressor = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
        self.shared_opportunity_expert = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
        )
        self.shared_opportunity_head = nn.Linear(config.hidden_dim, 1)
        self.shared_opportunity_gate = nn.Sequential(
            nn.Linear(config.hidden_dim * 3, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Linear(config.hidden_dim, 1),
        )
        self.backbone_residual_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )
        self.backbone_residual_gate = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.GELU(),
            nn.Linear(config.hidden_dim // 2, 1),
        )
        self.region_aux_head = (
            nn.Linear(config.hidden_dim, len(config.region_label_names))
            if config.region_label_names
            else None
        )
        self.income_aux_head = (
            nn.Linear(config.hidden_dim, len(config.income_label_names))
            if config.income_label_names
            else None
        )
        self.consistency_head = nn.Linear(config.hidden_dim, 1)
        self.stability = OpportunityStabilityScorer(margin=config.consistency_margin)
        self.router_context_embedding = nn.Embedding(self._num_context_states(config), config.hidden_dim)

    def _use_lite_satellite_path(self) -> bool:
        return (
            self.config.use_satellite_only_lite
            and self.config.use_satellite
            and not self.config.use_street_view
            and not self.config.use_tabular
        )

    def _match_cluster_prototypes(
        self,
        task_cluster_name: str,
        shared: torch.Tensor,
        task_prompt: torch.Tensor,
    ):
        cluster_idx = self.cluster_name_to_index[task_cluster_name]
        prototype_bank = self.cluster_prototypes[cluster_idx]
        prototype_bank = nn.functional.normalize(prototype_bank, dim=-1)
        query = self.prototype_query_proj(torch.cat([shared, task_prompt], dim=-1))
        query = nn.functional.normalize(query, dim=-1)
        attn_logits = query @ prototype_bank.transpose(0, 1)
        attn_weights = torch.softmax(
            attn_logits / max(self.config.prototype_temperature, 1e-6),
            dim=-1,
        )
        prototype_context = attn_weights @ self.prototype_value_proj(prototype_bank)
        return prototype_context, attn_weights

    @staticmethod
    def active_modality_count(config: UrbanOMEConfig) -> int:
        return int(config.use_satellite) + int(config.use_street_view) + int(config.use_tabular)

    @staticmethod
    def _num_context_states(config: UrbanOMEConfig) -> int:
        label_cardinality = (
            len(config.country_label_names)
            + len(config.region_label_names)
            + len(config.income_label_names)
            + len(config.hemisphere_label_names)
        )
        return max(label_cardinality + 1, 1)

    def _build_router_context_embedding(
        self,
        country_label: torch.Tensor,
        region_label: torch.Tensor,
        income_label: torch.Tensor,
        hemisphere_label: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        batch_size = country_label.shape[0]
        context_embed = torch.zeros(batch_size, self.config.hidden_dim, device=device)
        offset = 1
        label_groups = [
            (country_label, len(self.config.country_label_names)),
            (region_label, len(self.config.region_label_names)),
            (income_label, len(self.config.income_label_names)),
            (hemisphere_label, len(self.config.hemisphere_label_names)),
        ]
        for labels, size in label_groups:
            if size <= 0:
                continue
            mapped = labels.to(device).clone()
            valid = mapped >= 0
            mapped[~valid] = 0
            mapped[valid] = mapped[valid] + offset
            context_embed = context_embed + self.router_context_embedding(mapped)
            offset += size
        return context_embed

    def _build_semantic_neighbor_prompts(self, prompt_embeddings: torch.Tensor) -> torch.Tensor:
        task_names = list(self.config.task_names)
        train_task_names = list(self.config.train_task_keys or task_names)
        train_indices = [
            idx
            for idx, task_name in enumerate(task_names)
            if task_name in set(train_task_names)
        ]
        if not train_indices:
            return torch.zeros_like(prompt_embeddings)

        train_index_tensor = torch.tensor(train_indices, device=prompt_embeddings.device, dtype=torch.long)
        normalized_all = nn.functional.normalize(prompt_embeddings, dim=-1)
        normalized_train = normalized_all.index_select(0, train_index_tensor)
        similarity = normalized_all @ normalized_train.transpose(0, 1)

        for task_idx, task_name in enumerate(task_names):
            if task_name in train_task_names:
                same_positions = [pos for pos, idx in enumerate(train_indices) if idx == task_idx]
                if same_positions:
                    similarity[task_idx, same_positions[0]] = float("-inf")

        invalid_rows = torch.isinf(similarity).all(dim=-1)
        similarity_for_topk = similarity.clone()
        if invalid_rows.any():
            similarity_for_topk[invalid_rows] = 0.0

        top_k = min(max(int(self.config.semantic_neighbor_top_k), 1), similarity.shape[-1])
        topk_indices = torch.topk(similarity_for_topk, k=top_k, dim=-1).indices
        topk_mask = torch.zeros_like(similarity, dtype=torch.bool)
        topk_mask.scatter_(1, topk_indices, True)
        topk_mask = topk_mask & ~torch.isinf(similarity)
        similarity = similarity.masked_fill(~topk_mask, float("-inf"))

        weights = torch.softmax(
            similarity / max(self.config.semantic_neighbor_temperature, 1e-6),
            dim=-1,
        )
        if invalid_rows.any():
            weights = weights.clone()
            weights[invalid_rows] = 0.0
        return weights @ prompt_embeddings.index_select(0, train_index_tensor)

    def forward(
        self,
        satellite: torch.Tensor,
        street_view: torch.Tensor,
        tabular: torch.Tensor,
        routing_context: torch.Tensor,
        country_label: torch.Tensor = None,
        region_label: torch.Tensor = None,
        income_label: torch.Tensor = None,
        hemisphere_label: torch.Tensor = None,
        satellite_embedding: torch.Tensor = None,
        masked_task_names=None,
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        masked_task_names = set(masked_task_names or [])
        encoded = self.encoders(
            satellite,
            street_view,
            tabular,
            satellite_embedding=satellite_embedding,
        )
        if not encoded:
            raise ValueError("At least one modality must be enabled.")
        shared = self.shared_proj(torch.cat(list(encoded.values()), dim=-1))

        task_predictions = {}
        base_task_predictions = {}
        residual_task_predictions = {}
        task_features = {}
        modality_predictions = {}
        routing_weight_history = {"modality": [], "region": [], "task": [], "task_cluster": []}
        routing_weights_by_task = {}
        expert_outputs = {"modality": [], "region": [], "task": [], "task_cluster": []}
        prompt_feature_map = {}
        prototype_attention_map = {}

        prompt_lookup = self.config.task_prompt_templates or {
            task_name: self.config.default_task_prompt_template.format(task=task_name)
            for task_name in self.config.task_names
        }
        use_lite_satellite_path = self._use_lite_satellite_path()
        if self.config.prompt_encoder_type == "external_embedding":
            if not self.config.task_prompt_embeddings:
                raise ValueError(
                    "prompt_encoder_type='external_embedding' requires task prompt embeddings "
                    "from task_prompt_embedding column or external_task_prompt_embeddings_path."
                )
            prompt_embeddings = self.prompt_encoder.encode_prompt_embeddings(
                self.config.task_prompt_embeddings,
                self.config.task_names,
                shared.device,
            )
        else:
            prompt_list = [prompt_lookup[task_name] for task_name in self.config.task_names]
            prompt_embeddings = self.prompt_encoder.encode_prompt_list(prompt_list, shared.device)

        if self.config.use_semantic_neighbor_router:
            semantic_neighbor_prompts = self._build_semantic_neighbor_prompts(prompt_embeddings)
        else:
            semantic_neighbor_prompts = torch.zeros_like(prompt_embeddings)

        router_visual = self.router_visual_proj(shared)
        if country_label is None:
            country_label = torch.full((shared.shape[0],), -1, dtype=torch.long, device=shared.device)
        if region_label is None:
            region_label = torch.full((shared.shape[0],), -1, dtype=torch.long, device=shared.device)
        if income_label is None:
            income_label = torch.full((shared.shape[0],), -1, dtype=torch.long, device=shared.device)
        if hemisphere_label is None:
            hemisphere_label = torch.full((shared.shape[0],), -1, dtype=torch.long, device=shared.device)
        router_context_embed = self._build_router_context_embedding(
            country_label=country_label,
            region_label=region_label,
            income_label=income_label,
            hemisphere_label=hemisphere_label,
            device=shared.device,
        )
        router_context = self.router_context_proj(torch.cat([routing_context, router_context_embed], dim=-1))

        for task_idx, task_name in enumerate(self.config.task_names):
            task_prompt = prompt_embeddings[task_idx].unsqueeze(0).expand_as(shared)
            router_task_prompt = self.router_task_proj(task_prompt)
            if self.config.use_semantic_neighbor_router:
                neighbor_prompt = semantic_neighbor_prompts[task_idx].unsqueeze(0).expand_as(shared)
                neighbor_router_prompt = self.router_task_proj(neighbor_prompt)
                if self.config.use_task_adaptive_semantic_mix:
                    semantic_mix_weight = torch.sigmoid(
                        self.semantic_neighbor_gate(
                            torch.cat(
                                [
                                    router_task_prompt,
                                    neighbor_router_prompt,
                                    torch.abs(router_task_prompt - neighbor_router_prompt),
                                ],
                                dim=-1,
                            )
                        )
                    )
                else:
                    semantic_mix_weight = self.config.semantic_neighbor_mix_weight
                router_task_prompt = self.semantic_neighbor_fusion(
                    torch.cat(
                        [
                            router_task_prompt,
                            semantic_mix_weight * neighbor_router_prompt,
                        ],
                        dim=-1,
                    )
                )
            is_masked_task = task_name in masked_task_names
            if is_masked_task:
                router_task_prompt = torch.zeros_like(router_task_prompt)
            task_cluster_name = self.task_cluster_map[task_name]
            if self.config.use_cluster_prototype_router:
                prototype_context, prototype_weights = self._match_cluster_prototypes(
                    task_cluster_name,
                    shared,
                    task_prompt,
                )
                prototype_attention_map[task_name] = prototype_weights
                router_context_for_task = self.prototype_router_fusion(
                    torch.cat(
                        [
                            router_context,
                            self.config.prototype_mix_weight * prototype_context,
                        ],
                        dim=-1,
                    )
                )
            else:
                router_context_for_task = router_context
                prototype_attention_map[task_name] = None
            routing_weights = self.router(
                router_context_for_task,
                router_visual,
                router_task_prompt,
                task_cluster_name,
            )
            if not use_lite_satellite_path:
                routing_weight_history["modality"].append(routing_weights["modality"])
            routing_weight_history["region"].append(routing_weights["region"])
            routing_weight_history["task"].append(routing_weights["task"])
            routing_weight_history["task_cluster"].append(routing_weights["task_cluster"])
            routing_weights_by_task[task_name] = {
                key: value
                for key, value in routing_weights.items()
            }

            if use_lite_satellite_path:
                zero_feature = torch.zeros_like(shared)
                modality_pool = {
                    "mixed": zero_feature,
                    "all": torch.stack(
                        [zero_feature for _ in self.config.modality_names],
                        dim=1,
                    ),
                }
            else:
                modality_pool = self.modality_experts(
                    shared + task_prompt, routing_weights["modality"], self.config.top_k_modality
                )
            region_pool = self.region_experts(
                shared + task_prompt, routing_weights["region"], self.config.top_k_region
            )

            prompt_feature = self.prompt_to_feature(task_prompt)
            prompt_feature_map[task_name] = prompt_feature
            task_conditioned = shared + prompt_feature
            if is_masked_task:
                zero_feature = torch.zeros_like(shared)
                task_cluster_pool = {
                    "mixed": zero_feature,
                    "all": [zero_feature for _ in self.task_cluster_expert_names],
                }
                task_pool = {
                    "mixed": zero_feature,
                    "all": [zero_feature for _ in self.cluster_task_expert_names[task_cluster_name]],
                }
            else:
                task_cluster_pool = self.task_cluster_experts(
                    task_conditioned,
                    routing_weights["task_cluster"],
                    self.config.top_k_task_cluster,
                )
                task_pool = self.task_experts[task_cluster_name](
                    task_conditioned, routing_weights["task"], self.config.top_k_task
                )
            modality_mix = modality_pool["mixed"]
            region_mix = region_pool["mixed"]
            task_mix = task_cluster_pool["mixed"] + task_pool["mixed"]
            fused = self.composer(
                shared,
                modality_mix,
                region_mix,
                task_mix,
                include_modality=not use_lite_satellite_path,
            )
            fused = fused + prompt_feature
            film_params = self.film_generator(prompt_feature)
            gamma, beta = torch.chunk(film_params, 2, dim=-1)
            modulated_fused = (1.0 + gamma) * fused + beta
            task_features[task_name] = modulated_fused
            base_task_predictions[task_name] = self.shared_regressor(
                torch.cat([modulated_fused, prompt_feature], dim=-1)
            )
            if not use_lite_satellite_path:
                expert_outputs["modality"].append(modality_pool["all"])
            expert_outputs["region"].append(region_pool["all"])
            expert_outputs["task"].append(task_pool["all"])
            expert_outputs["task_cluster"].append(task_cluster_pool["all"])

        task_name_order = list(self.config.task_names)
        if task_name_order:
            stacked_features = torch.stack([task_features[task_name] for task_name in task_name_order], dim=0)
            stacked_base_predictions = torch.stack(
                [base_task_predictions[task_name] for task_name in task_name_order],
                dim=0,
            )
            if self.config.use_task_context and self.config.task_context_scope != "none" and len(task_name_order) > 1:
                normalized_prompt_embeddings = nn.functional.normalize(prompt_embeddings, dim=-1)
                semantic_similarity = normalized_prompt_embeddings @ normalized_prompt_embeddings.transpose(0, 1)
                diagonal_mask = torch.eye(
                    semantic_similarity.shape[0],
                    device=semantic_similarity.device,
                    dtype=torch.bool,
                )
                semantic_similarity = semantic_similarity.masked_fill(diagonal_mask, float("-inf"))
                if self.config.task_context_scope == "intra_cluster":
                    cluster_mask = torch.zeros_like(semantic_similarity, dtype=torch.bool)
                    for i, task_i in enumerate(task_name_order):
                        cluster_i = self.task_cluster_map[task_i]
                        for j, task_j in enumerate(task_name_order):
                            if i == j:
                                continue
                            if self.task_cluster_map[task_j] == cluster_i:
                                cluster_mask[i, j] = True
                    semantic_similarity = semantic_similarity.masked_fill(~cluster_mask, float("-inf"))
                    no_neighbor_rows = ~cluster_mask.any(dim=-1)
                    if no_neighbor_rows.any():
                        semantic_similarity = semantic_similarity.clone()
                        semantic_similarity[no_neighbor_rows] = float("-inf")
                if self.config.task_context_top_k and self.config.task_context_top_k > 0:
                    top_k = min(int(self.config.task_context_top_k), semantic_similarity.shape[-1])
                    if top_k > 0:
                        similarity_for_topk = semantic_similarity.clone()
                        invalid_rows = torch.isinf(similarity_for_topk).all(dim=-1)
                        if invalid_rows.any():
                            similarity_for_topk[invalid_rows] = 0.0
                        topk_indices = torch.topk(similarity_for_topk, k=top_k, dim=-1).indices
                        topk_mask = torch.zeros_like(semantic_similarity, dtype=torch.bool)
                        topk_mask.scatter_(1, topk_indices, True)
                        topk_mask = topk_mask & ~torch.isinf(semantic_similarity)
                        semantic_similarity = semantic_similarity.masked_fill(~topk_mask, float("-inf"))
                task_context_weights = torch.softmax(
                    semantic_similarity / max(self.config.task_context_temperature, 1e-6),
                    dim=-1,
                )
                invalid_rows = torch.isinf(semantic_similarity).all(dim=-1)
                if invalid_rows.any():
                    task_context_weights = task_context_weights.clone()
                    task_context_weights[invalid_rows] = 0.0
                context_features = torch.einsum("ij,jbh->ibh", task_context_weights, stacked_features)
                context_predictions = torch.einsum("ij,jbo->ibo", task_context_weights, stacked_base_predictions)
            else:
                task_context_weights = None
                context_features = torch.zeros_like(stacked_features)
                context_predictions = torch.zeros_like(stacked_base_predictions)

            for task_idx, task_name in enumerate(task_name_order):
                own_feature = stacked_features[task_idx]
                prompt_feature = prompt_feature_map[task_name]
                base_prediction = base_task_predictions[task_name]
                if self.config.use_task_context and len(task_name_order) > 1:
                    context_feature = context_features[task_idx]
                    context_prediction = context_predictions[task_idx]
                    context_signal = context_feature + self.task_context_pred_proj(context_prediction)
                    context_gate = torch.sigmoid(
                        self.task_context_gate(torch.cat([own_feature, context_signal], dim=-1))
                    )
                    enhanced_feature = own_feature + self.config.task_context_mix_weight * context_gate * context_signal
                else:
                    enhanced_feature = own_feature

                if self.config.use_prompt_hyper_head:
                    hyper_params = self.hyper_head_generator(prompt_feature)
                    hyper_weight = hyper_params[:, : self.config.hidden_dim]
                    hyper_bias = hyper_params[:, self.config.hidden_dim :]
                    hyper_prediction = (
                        (hyper_weight * enhanced_feature).sum(dim=-1, keepdim=True) / math.sqrt(self.config.hidden_dim)
                    ) + hyper_bias
                    mix_alpha = torch.sigmoid(self.hyper_head_mixer(prompt_feature))
                    final_prediction = mix_alpha * base_prediction + (1.0 - mix_alpha) * hyper_prediction
                else:
                    final_prediction = base_prediction

                if self.config.use_shared_opportunity_expert:
                    shared_opportunity_feature = self.shared_opportunity_expert(
                        torch.cat([shared, prompt_feature], dim=-1)
                    )
                    shared_opportunity_prediction = self.shared_opportunity_head(shared_opportunity_feature)
                    shared_gate = torch.sigmoid(
                        self.shared_opportunity_gate(
                            torch.cat(
                                [
                                    enhanced_feature,
                                    shared_opportunity_feature,
                                    prompt_feature,
                                ],
                                dim=-1,
                            )
                        )
                    )
                    shared_alpha = self.config.shared_opportunity_init_weight * shared_gate
                    final_prediction = (1.0 - shared_alpha) * final_prediction + shared_alpha * shared_opportunity_prediction

                if self.config.use_backbone_residual_head:
                    residual_prediction = self.backbone_residual_head(shared)
                    residual_task_predictions[task_name] = residual_prediction
                    if self.config.use_backbone_residual_inference:
                        residual_gate = torch.sigmoid(self.backbone_residual_gate(prompt_feature))
                        residual_alpha = self.config.backbone_residual_weight * residual_gate
                        final_prediction = final_prediction + residual_alpha * residual_prediction

                task_features[task_name] = enhanced_feature
                task_predictions[task_name] = final_prediction

        for modality_name, feature in encoded.items():
            modality_predictions[modality_name] = self.consistency_head(feature)

        mean_routing_weights = {
            key: torch.stack(weight_list, dim=0).mean(dim=0)
            for key, weight_list in routing_weight_history.items()
            if weight_list
        }

        stability = self.stability(
            task_predictions=task_predictions,
            routing_weights=mean_routing_weights,
            modality_predictions=modality_predictions,
        )

        aux_outputs = {}
        if self.region_aux_head is not None:
            aux_outputs["region_logits"] = self.region_aux_head(shared)
        if self.income_aux_head is not None:
            aux_outputs["income_logits"] = self.income_aux_head(shared)

        return {
            "encoded": encoded,
            "routing_weights": mean_routing_weights,
            "routing_weights_by_task": routing_weights_by_task,
            "task_features": task_features,
            "base_task_predictions": base_task_predictions,
            "residual_task_predictions": residual_task_predictions,
            "task_predictions": task_predictions,
            "expert_outputs": expert_outputs,
            "modality_predictions": modality_predictions,
            "stability": stability,
            "aux_outputs": aux_outputs,
            "debug": {
                "prompt_embeddings": prompt_embeddings.detach(),
                "shared": shared.detach(),
                "task_context_weights": (
                    task_context_weights.detach()
                    if (task_name_order and task_context_weights is not None)
                    else None
                ),
                "prototype_attention_map": prototype_attention_map,
            },
        }
