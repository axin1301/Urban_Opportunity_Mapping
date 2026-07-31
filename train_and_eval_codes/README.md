# OppMoE: Prompt-Conditioned Mixture-of-Experts for Urban Opportunity Mapping

This repository contains the training, evaluation, baseline, and analysis code for OppMoE, a prompt-conditioned region-task Mixture-of-Experts model for transferable urban opportunity mapping.

## Structure

- `urbanome/`: core model, data loading, prompt encoding, expert routing, and training utilities.
- `train_urbanome.py`: main training entry point.
- `eval_urbanome.py`: evaluate one checkpoint.
- `eval_urbanome_checkpoints.py`: evaluate multiple checkpoints in one run.
- `baselines/run_embedding_baseline.py`: prompt-MLP baseline on cached satellite embeddings.

Large files such as satellite images, cached embeddings, checkpoints, and prediction outputs are intentionally not included.

## Main Training Command

```bash
python -u train_urbanome.py \
  --train_csv /path/to/train_cached_remoteclip.parquet \
  --test_csv /path/to/unseen_city_test_cached_remoteclip.parquet \
  --use_cached_satellite_embeddings \
  --satellite_embedding_col satellite_embedding_remoteclip \
  --split_train_as_val \
  --val_ratio 0.2 \
  --val_split_by city \
  --image_root /path/to/output_files \
  --satellite_path_template "{city}/arcgis-imagery-FUA/{image_name}" \
  --disable_street_view \
  --disable_tabular \
  --prompt_encoder_type external_embedding \
  --external_task_prompt_embeddings_path /path/to/task_prompt_qwen3_4b_embeddings.json \
  --train_task_keys population_count nighttime_light_intensity education_accessibility food_accessibility digital_total_cell_intensity \
  --test_task_keys healthcare_accessibility mobility_opportunity social_opportunity \
  --target_value_col indicator_value_log1p \
  --checkpoint_metric grid_mse \
  --num_task_experts 4 \
  --num_task_experts_per_cluster 1 \
  --top_k_task_cluster 1 \
  --router_temperature 1.5 \
  --routing_context_cols tile_x tile_y patch_lon patch_lat min_lon min_lat max_lon max_lat patch_area_km2 \
  --batch_size 1024 \
  --num_epochs 100 \
  --save_every_n_epochs 10 \
  --tta_steps 0 \
  --save_dir outputs_remoteclip_qwen_final_no_semantic_neighbor \
  --task_context_top_k 2 \
  --early_stop_patience 5 \
  --early_stop_min_delta 1e-4 \
  --use_satellite_only_lite \
  --task_mask_meta_train \
  --task_mask_ratio 0.3 \
  --min_masked_tasks 1 \
  --max_masked_tasks 2 \
  --task_mask_loss_weight 0.05 \
  --use_shared_opportunity_expert \
  --shared_opportunity_init_weight 0.05 \
  --drop_nan_targets \
  --unseen_task_alignment_loss_weight 0 \
  --device cuda
```

## Evaluation Example

```bash
python -u eval_urbanome_checkpoints.py \
  --checkpoint_dir outputs_remoteclip_qwen_final_no_semantic_neighbor \
  --include_best \
  --train_csv /path/to/train_cached_remoteclip.parquet \
  --test_csv /path/to/unseen_city_test_cached_remoteclip.parquet \
  --use_cached_satellite_embeddings \
  --satellite_embedding_col satellite_embedding_remoteclip \
  --prompt_encoder_type external_embedding \
  --external_task_prompt_embeddings_path /path/to/task_prompt_qwen3_4b_embeddings.json \
  --image_root /path/to/output_files \
  --satellite_path_template "{city}/arcgis-imagery-FUA/{image_name}" \
  --train_task_keys population_count nighttime_light_intensity education_accessibility food_accessibility digital_total_cell_intensity \
  --test_task_keys healthcare_accessibility mobility_opportunity social_opportunity \
  --routing_context_cols tile_x tile_y patch_lon patch_lat min_lon min_lat max_lon max_lat patch_area_km2 \
  --use_satellite_only_lite \
  --use_shared_opportunity_expert \
  --shared_opportunity_init_weight 0.05 \
  --drop_nan_targets \
  --save_dir eval_outputs
```

## Notes

- The expected data format is a long table with `indicator_column` and `indicator_value_log1p`.
- Cached satellite embeddings should be stored in a column such as `satellite_embedding_remoteclip`.
- Task prompt embeddings are loaded externally through `--external_task_prompt_embeddings_path`.
- Validation is split from training cities when `--split_train_as_val --val_split_by city` is enabled.

## Baseline Example

```bash
python run_embedding_baseline.py \
  --train_csv /path/to/train.csv \
  --test_csv /path/to/unseen_city_test.csv \
  --feature_pt /path/to/all_split_features_remoteclip.pt \
  --model_type prompt_mlp \
  --setting unseen_indicator \
  --train_task_keys population_count nighttime_light_intensity education_accessibility food_accessibility digital_total_cell_intensity \
  --test_task_keys healthcare_accessibility mobility_opportunity social_opportunity \
  --prompt_embeddings_json /path/to/task_prompt_qwen3_4b_embeddings.json \
  --target_value_col indicator_value_log1p \
  --batch_size 1024 \
  --num_epochs 100 \
  --learning_rate 1e-3 \
  --weight_decay 1e-4 \
  --hidden_dim 256 \
  --dropout 0.1 \
  --val_ratio 0.2 \
  --val_split_by city \
  --checkpoint_metric grid_mse \
  --device cuda \
  --save_dir baseline_outputs
```