from dataclasses import dataclass
import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .config import UrbanOMEConfig
from .prompts import resolve_task_prompt_embeddings, resolve_task_prompt_templates


ID_COLUMNS = {"country", "city", "grid"}


@dataclass
class CSVSchema:
    target_cols: List[str]
    routing_context_cols: List[str]
    task_prompt_templates: Dict[str, str]
    task_prompt_embeddings: Optional[Dict[str, List[float]]]
    long_format: bool
    country_label_names: List[str]
    region_label_names: List[str]
    income_label_names: List[str]
    hemisphere_label_names: List[str]


def _safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def read_table(path: str) -> pd.DataFrame:
    if path.lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _numeric_value_and_mask(value) -> Tuple[float, float]:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return 0.0, 0.0
    return float(numeric), 1.0


def _transform_target_value(task_key: str, value: float, config: UrbanOMEConfig) -> float:
    if config.log1p_task_keys and task_key in set(config.log1p_task_keys):
        return float(np.log1p(max(value, 0.0)))
    return value


def _build_label_names(df: pd.DataFrame, col: str) -> List[str]:
    if col not in df.columns:
        return []
    values = df[col].dropna().astype(str)
    values = [value for value in values.tolist() if value.strip()]
    return list(dict.fromkeys(values))


def _encode_label(value, label_names: List[str]) -> int:
    if not label_names:
        return -1
    if pd.isna(value):
        return -1
    value = str(value)
    if not value.strip():
        return -1
    try:
        return label_names.index(value)
    except ValueError:
        return -1


def _resolve_image_path(
    image_root: str,
    image_name: str,
    image_path_template: str,
    row: pd.Series,
) -> str:
    image_name = str(image_name)
    if os.path.isabs(image_name):
        return image_name

    format_kwargs = {col: str(row[col]) for col in row.index}
    format_kwargs["image_name"] = image_name
    relative_path = image_path_template.format(**format_kwargs)
    if os.path.isabs(relative_path):
        return relative_path
    return os.path.join(image_root, relative_path)


def _load_image(image_path: str, image_size: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((image_size, image_size))
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = np.transpose(image_array, (2, 0, 1))
    return torch.from_numpy(image_array)


def _parse_embedding_value(value) -> Optional[List[float]]:
    if isinstance(value, list):
        return [float(x) for x in value]
    if isinstance(value, tuple):
        return [float(x) for x in value]
    if isinstance(value, np.ndarray):
        return [float(x) for x in value.reshape(-1).tolist()]
    if value is None:
        return None
    if np.isscalar(value) and pd.isna(value):
        return None
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return None
        if isinstance(parsed, list):
            return [float(x) for x in parsed]
    return None


def infer_cached_embedding_dim(df: pd.DataFrame, embedding_col: str) -> Optional[int]:
    if embedding_col not in df.columns:
        return None
    for value in df[embedding_col].tolist():
        parsed = _parse_embedding_value(value)
        if parsed is not None and len(parsed) > 0:
            return int(len(parsed))
    return None


def infer_target_columns(df: pd.DataFrame, config: UrbanOMEConfig) -> List[str]:
    if config.task_key_col in df.columns and config.target_value_col in df.columns:
        return list(dict.fromkeys(df[config.task_key_col].astype(str).tolist()))

    if config.target_start_col not in df.columns:
        raise ValueError(f"Missing target_start_col: {config.target_start_col}")

    start_idx = df.columns.get_loc(config.target_start_col) + 1
    candidate_cols = list(df.columns[start_idx:])
    blocked = set(config.exclude_target_cols) | {config.text_prompt_col}
    target_cols = [c for c in candidate_cols if c not in blocked]
    if not target_cols:
        raise ValueError("No target columns inferred from CSV. Check target_start_col/exclude_target_cols.")
    return target_cols


def merge_task_names(train_target_cols: List[str], test_target_cols: List[str]) -> List[str]:
    merged = list(train_target_cols)
    for task_name in test_target_cols:
        if task_name not in merged:
            merged.append(task_name)
    return merged


def merge_explicit_task_keys(target_cols: List[str], config: UrbanOMEConfig) -> List[str]:
    merged = list(target_cols)
    for task_list in (config.include_task_keys, config.train_task_keys, config.test_task_keys, config.meta_val_task_keys):
        for task_name in task_list or []:
            task_name = str(task_name)
            if task_name not in merged:
                merged.append(task_name)
    return merged


def infer_routing_context_columns(df: pd.DataFrame, config: UrbanOMEConfig, target_cols: List[str]) -> List[str]:
    if config.routing_context_cols is not None:
        missing = [c for c in config.routing_context_cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing routing context columns: {missing}")
        return config.routing_context_cols

    if config.use_tabular and config.tabular_feature_cols:
        available_tabular_cols = [c for c in config.tabular_feature_cols if c in df.columns]
        if available_tabular_cols:
            return list(available_tabular_cols)

    blocked = (
        {
            config.country_col,
            config.city_col,
            config.grid_col,
            config.satellite_col,
            config.text_prompt_col,
            config.task_key_col,
            config.task_name_col,
            config.target_value_col,
            config.definition_col,
        }
        | set(config.street_view_cols)
        | set(target_cols)
    )
    context_cols = []
    for col in df.columns:
        if col in blocked:
            continue
        if col in ID_COLUMNS:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if numeric.notna().any():
            context_cols.append(col)
    if not context_cols:
        raise ValueError(
            "No routing context columns were inferred. "
            "Please provide --routing_context_cols explicitly or enable valid tabular feature columns."
        )
    return context_cols


def _build_long_format_prompt_templates(df: pd.DataFrame, config: UrbanOMEConfig) -> Dict[str, str]:
    task_prompt_templates = {}
    grouped = df.groupby(config.task_key_col, dropna=False)
    for task_key, task_df in grouped:
        task_key = str(task_key)
        prompt = None
        if config.text_prompt_col in task_df.columns:
            non_null = task_df[config.text_prompt_col].dropna()
            if len(non_null) > 0:
                prompt = str(non_null.iloc[0])
        if not prompt and config.definition_col in task_df.columns:
            non_null = task_df[config.definition_col].dropna()
            if len(non_null) > 0:
                prompt = f"Task: {task_key}. Definition: {str(non_null.iloc[0])}"
        if not prompt and config.task_name_col in task_df.columns:
            non_null = task_df[config.task_name_col].dropna()
            if len(non_null) > 0:
                prompt = config.default_task_prompt_template.format(task=str(non_null.iloc[0]))
        if not prompt:
            prompt = config.default_task_prompt_template.format(task=task_key)
        task_prompt_templates[task_key] = prompt
    return task_prompt_templates


def load_schema(train_df: pd.DataFrame, test_df: pd.DataFrame, config: UrbanOMEConfig) -> CSVSchema:
    if config.fixed_eval_schema and config.task_names:
        target_cols = list(config.task_names)
        train_target_cols = target_cols
        test_target_cols = target_cols
    else:
        train_target_cols = infer_target_columns(train_df, config)
        test_target_cols = infer_target_columns(test_df, config)
        target_cols = merge_task_names(train_target_cols, test_target_cols)
        target_cols = merge_explicit_task_keys(target_cols, config)
    routing_context_cols = infer_routing_context_columns(train_df, config, train_target_cols)
    long_format = config.task_key_col in train_df.columns and config.target_value_col in train_df.columns
    raw_prompt_value = ""
    raw_embedding_value = ""
    if not long_format and config.text_prompt_col in train_df.columns:
        non_null_prompts = train_df[config.text_prompt_col].dropna()
        if len(non_null_prompts) > 0:
            raw_prompt_value = str(non_null_prompts.iloc[0])
    if not long_format and config.task_prompt_embedding_col in train_df.columns:
        non_null_embeddings = train_df[config.task_prompt_embedding_col].dropna()
        if len(non_null_embeddings) > 0:
            raw_embedding_value = str(non_null_embeddings.iloc[0])

    if long_format:
        prompt_df = pd.concat([train_df, test_df], ignore_index=True)
        task_prompt_templates = _build_long_format_prompt_templates(prompt_df, config)
        if config.fixed_eval_schema and config.task_prompt_templates:
            merged_templates = dict(config.task_prompt_templates)
            merged_templates.update(task_prompt_templates)
            for task_name in target_cols:
                merged_templates.setdefault(
                    task_name,
                    config.default_task_prompt_template.format(task=task_name),
                )
            task_prompt_templates = merged_templates
    else:
        task_prompt_templates = resolve_task_prompt_templates(
            raw_prompt_value=raw_prompt_value,
            task_names=target_cols,
            default_template=config.default_task_prompt_template,
        )

    task_prompt_embeddings = resolve_task_prompt_embeddings(
        raw_embedding_value=raw_embedding_value,
        task_names=target_cols,
        external_path=config.external_task_prompt_embeddings_path,
    )
    if config.fixed_eval_schema:
        country_label_names = list(config.country_label_names)
        region_label_names = list(config.region_label_names)
        income_label_names = list(config.income_label_names)
        hemisphere_label_names = list(config.hemisphere_label_names)
    else:
        concat_df = pd.concat([train_df, test_df], ignore_index=True)
        country_label_names = _build_label_names(concat_df, config.country_col)
        region_label_names = _build_label_names(concat_df, config.region_col)
        income_label_names = _build_label_names(concat_df, config.income_level_col)
        hemisphere_label_names = _build_label_names(concat_df, config.hemisphere_col)

    return CSVSchema(
        target_cols=target_cols,
        routing_context_cols=routing_context_cols,
        task_prompt_templates=task_prompt_templates,
        task_prompt_embeddings=task_prompt_embeddings,
        long_format=long_format,
        country_label_names=country_label_names,
        region_label_names=region_label_names,
        income_label_names=income_label_names,
        hemisphere_label_names=hemisphere_label_names,
    )


def filter_dataframe_for_experiment(
    df: pd.DataFrame,
    config: UrbanOMEConfig,
    is_train: bool,
    override_task_keys: Optional[List[str]] = None,
    exclude_task_keys: Optional[List[str]] = None,
) -> pd.DataFrame:
    filtered = df.copy()

    task_keys = override_task_keys
    if task_keys is None:
        task_keys = config.train_task_keys if is_train else config.test_task_keys
    if task_keys is None:
        task_keys = config.include_task_keys
    if task_keys and config.task_key_col in filtered.columns:
        keep = set(task_keys)
        filtered = filtered[filtered[config.task_key_col].astype(str).isin(keep)].copy()
    if exclude_task_keys and config.task_key_col in filtered.columns:
        drop = set(exclude_task_keys)
        filtered = filtered[~filtered[config.task_key_col].astype(str).isin(drop)].copy()

    if (
        config.drop_nan_targets
        and config.task_key_col in filtered.columns
        and config.target_value_col in filtered.columns
    ):
        numeric_targets = pd.to_numeric(filtered[config.target_value_col], errors="coerce")
        filtered = filtered[numeric_targets.notna()].copy()

    if config.max_samples_per_task > 0 and config.task_key_col in filtered.columns:
        sampled_groups = []
        for _, group_df in filtered.groupby(config.task_key_col, dropna=False):
            sampled_groups.append(
                group_df.sample(
                    n=min(config.max_samples_per_task, len(group_df)),
                    random_state=config.random_seed,
                )
            )
        if sampled_groups:
            filtered = pd.concat(sampled_groups, ignore_index=True)

    max_samples = config.max_train_samples if is_train else config.max_test_samples
    if max_samples > 0 and len(filtered) > max_samples:
        filtered = filtered.sample(n=max_samples, random_state=config.random_seed).reset_index(drop=True)
    else:
        filtered = filtered.reset_index(drop=True)

    return filtered


class UrbanCSVData(Dataset):
    def __init__(self, df: pd.DataFrame, config: UrbanOMEConfig, schema: CSVSchema):
        self.df = df.reset_index(drop=True).copy()
        self.config = config
        self.schema = schema
        self.task_to_index = {task_name: idx for idx, task_name in enumerate(schema.target_cols)}

        required = [
            config.country_col,
            config.city_col,
        ]
        if config.grid_col in self.df.columns:
            required.append(config.grid_col)
        if schema.long_format:
            required.extend([config.task_key_col, config.target_value_col])
        else:
            required.extend(schema.target_cols)
        satellite_missing = []
        if config.use_satellite:
            if config.use_cached_satellite_embeddings:
                has_cached = config.satellite_embedding_col in self.df.columns
                has_raw = config.satellite_col in self.df.columns
                if not has_cached and not has_raw:
                    satellite_missing.extend([config.satellite_embedding_col, config.satellite_col])
            else:
                required.append(config.satellite_col)
        if config.use_street_view:
            required.extend(config.street_view_cols)
        if config.use_tabular:
            required.extend(config.tabular_feature_cols)
        missing = [c for c in required if c not in self.df.columns]
        missing.extend([c for c in satellite_missing if c not in missing])
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]

        satellite_embedding = torch.zeros(0, dtype=torch.float32)
        if self.config.use_satellite:
            parsed_embedding = None
            if self.config.use_cached_satellite_embeddings and self.config.satellite_embedding_col in self.df.columns:
                parsed_embedding = _parse_embedding_value(row[self.config.satellite_embedding_col])
            if parsed_embedding is not None:
                satellite_embedding = torch.tensor(parsed_embedding, dtype=torch.float32)
                satellite = torch.zeros(0, dtype=torch.float32)
                satellite_path = ""
            else:
                if self.config.use_cached_satellite_embeddings and self.config.satellite_col not in self.df.columns:
                    raise ValueError(
                        "Cached satellite embedding is missing for this row and no raw satellite image column "
                        f"is available: {self.config.satellite_col}"
                    )
                satellite_path = _resolve_image_path(
                    self.config.image_root,
                    row[self.config.satellite_col],
                    self.config.satellite_path_template,
                    row,
                )
                satellite = _load_image(
                    satellite_path,
                    self.config.image_size,
                )
        else:
            satellite = torch.zeros(0, dtype=torch.float32)
            satellite_path = ""

        if self.config.use_street_view:
            street_view = torch.stack(
                [
                    _load_image(
                        _resolve_image_path(
                            self.config.image_root,
                            row[c],
                            self.config.street_view_path_template,
                            row,
                        ),
                        self.config.image_size,
                    )
                    for c in self.config.street_view_cols
                ],
                dim=0,
            )
        else:
            street_view = torch.zeros(0, dtype=torch.float32)

        if self.config.use_tabular:
            tabular = torch.tensor(
                [_safe_numeric(pd.Series([row[c]])).iloc[0] for c in self.config.tabular_feature_cols],
                dtype=torch.float32,
            )
        else:
            tabular = torch.zeros(0, dtype=torch.float32)
        routing_context = torch.tensor(
            [_safe_numeric(pd.Series([row[c]])).iloc[0] for c in self.schema.routing_context_cols],
            dtype=torch.float32,
        )
        if self.schema.long_format:
            targets_list = [0.0] * len(self.schema.target_cols)
            mask_list = [0.0] * len(self.schema.target_cols)
            task_key = str(row[self.config.task_key_col])
            task_idx = self.task_to_index[task_key]
            target_value, target_mask_value = _numeric_value_and_mask(row[self.config.target_value_col])
            target_value = _transform_target_value(task_key, target_value, self.config)
            targets_list[task_idx] = target_value
            mask_list[task_idx] = target_mask_value
            targets = torch.tensor(targets_list, dtype=torch.float32)
            target_mask = torch.tensor(mask_list, dtype=torch.float32)
        else:
            target_pairs = []
            for c in self.schema.target_cols:
                target_value, target_mask_value = _numeric_value_and_mask(row[c])
                target_value = _transform_target_value(c, target_value, self.config)
                target_pairs.append((target_value, target_mask_value))
            targets = torch.tensor(
                [pair[0] for pair in target_pairs],
                dtype=torch.float32,
            )
            target_mask = torch.tensor(
                [pair[1] for pair in target_pairs],
                dtype=torch.float32,
            )

        grid_value = str(row[self.config.grid_col]) if self.config.grid_col in self.df.columns else f"{row[self.config.city_col]}_{idx}"
        task_key_value = str(row[self.config.task_key_col]) if self.config.task_key_col in self.df.columns else ""
        task_prompt_value = self.schema.task_prompt_templates.get(task_key_value, "")
        satellite_path_value = satellite_path if self.config.use_satellite else ""
        if self.config.use_street_view:
            street_view_paths = [
                _resolve_image_path(
                    self.config.image_root,
                    row[c],
                    self.config.street_view_path_template,
                    row,
                )
                for c in self.config.street_view_cols
            ]
            street_view_paths_value = " | ".join(street_view_paths)
        else:
            street_view_paths_value = ""

        return {
            "satellite": satellite,
            "satellite_embedding": satellite_embedding,
            "street_view": street_view,
            "tabular": tabular,
            "routing_context": routing_context,
            "targets": targets,
            "target_mask": target_mask,
            "country_label": torch.tensor(
                _encode_label(row[self.config.country_col], self.schema.country_label_names)
                if self.config.country_col in self.df.columns
                else -1,
                dtype=torch.long,
            ),
            "region_label": torch.tensor(
                _encode_label(row[self.config.region_col], self.schema.region_label_names)
                if self.config.region_col in self.df.columns
                else -1,
                dtype=torch.long,
            ),
            "income_label": torch.tensor(
                _encode_label(row[self.config.income_level_col], self.schema.income_label_names)
                if self.config.income_level_col in self.df.columns
                else -1,
                dtype=torch.long,
            ),
            "hemisphere_label": torch.tensor(
                _encode_label(row[self.config.hemisphere_col], self.schema.hemisphere_label_names)
                if self.config.hemisphere_col in self.df.columns
                else -1,
                dtype=torch.long,
            ),
            "country": str(row[self.config.country_col]),
            "city": str(row[self.config.city_col]),
            "grid": grid_value,
            "task_key": task_key_value,
            "task_prompt": task_prompt_value,
            "satellite_path": satellite_path_value,
            "street_view_paths": street_view_paths_value,
        }


def _prepare_dataframe(
    df: pd.DataFrame,
    config: UrbanOMEConfig,
    is_train: bool,
    override_task_keys: Optional[List[str]] = None,
    exclude_task_keys: Optional[List[str]] = None,
) -> pd.DataFrame:
    filtered_df = filter_dataframe_for_experiment(
        df,
        config,
        is_train=is_train,
        override_task_keys=override_task_keys,
        exclude_task_keys=exclude_task_keys,
    )
    return filtered_df


def split_train_dataframe_as_val(
    train_df: pd.DataFrame,
    config: UrbanOMEConfig,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_col = config.city_col if config.val_split_by == "city" else config.country_col
    if split_col not in train_df.columns:
        raise ValueError(f"Cannot split train as val because column is missing: {split_col}")
    if not (0.0 < config.val_ratio < 1.0):
        raise ValueError(f"val_ratio must be in (0, 1), got {config.val_ratio}")

    groups = [str(x) for x in train_df[split_col].dropna().astype(str).unique().tolist() if str(x).strip()]
    if len(groups) < 2:
        raise ValueError(f"Need at least 2 unique {split_col} values to split train into train/val.")

    rng = np.random.default_rng(config.random_seed)
    permuted = list(rng.permutation(groups))
    num_val_groups = max(1, int(round(len(permuted) * config.val_ratio)))
    num_val_groups = min(num_val_groups, len(permuted) - 1)
    val_groups = set(permuted[:num_val_groups])

    val_df = train_df[train_df[split_col].astype(str).isin(val_groups)].copy().reset_index(drop=True)
    inner_train_df = train_df[~train_df[split_col].astype(str).isin(val_groups)].copy().reset_index(drop=True)

    if len(inner_train_df) == 0 or len(val_df) == 0:
        raise ValueError("Automatic train/val split produced an empty split. Adjust val_ratio or data filtering.")
    return inner_train_df, val_df


def build_dataloaders(config: UrbanOMEConfig):
    train_df = read_table(config.train_csv)
    test_df = read_table(config.test_csv)
    train_df = _prepare_dataframe(train_df, config, is_train=True)
    test_df = _prepare_dataframe(test_df, config, is_train=False)

    if config.use_street_view and not all(col in train_df.columns for col in config.street_view_cols):
        config.use_street_view = False
    if config.use_tabular:
        available_tabular_cols = [col for col in config.tabular_feature_cols if col in train_df.columns]
        config.tabular_feature_cols = available_tabular_cols
        if len(config.tabular_feature_cols) == 0:
            config.use_tabular = False
    if config.use_satellite:
        if config.use_cached_satellite_embeddings:
            has_cached = config.satellite_embedding_col in train_df.columns
            has_raw = config.satellite_col in train_df.columns
            if not has_cached and not has_raw:
                raise ValueError(
                    "Missing satellite inputs: need either cached embedding column "
                    f"{config.satellite_embedding_col} or raw image column {config.satellite_col}"
                )
        elif config.satellite_col not in train_df.columns:
            raise ValueError(f"Missing satellite column: {config.satellite_col}")

    schema = load_schema(train_df, test_df, config)
    config.routing_context_dim = len(schema.routing_context_cols)
    if config.use_satellite and config.use_cached_satellite_embeddings:
        inferred_sat_dim = infer_cached_embedding_dim(train_df, config.satellite_embedding_col)
        config.satellite_in_dim = inferred_sat_dim if inferred_sat_dim is not None else 3
    else:
        config.satellite_in_dim = 3 if config.use_satellite else 0
    config.street_view_in_dim = 3 if config.use_street_view else 0
    config.tabular_in_dim = len(config.tabular_feature_cols) if config.use_tabular else 0
    config.task_names = list(schema.target_cols)
    config.task_prompt_templates = dict(schema.task_prompt_templates)
    config.task_prompt_embeddings = schema.task_prompt_embeddings
    config.country_label_names = list(schema.country_label_names)
    config.region_label_names = list(schema.region_label_names)
    config.income_label_names = list(schema.income_label_names)
    config.hemisphere_label_names = list(schema.hemisphere_label_names)

    train_ds = UrbanCSVData(train_df, config, schema)
    test_ds = UrbanCSVData(test_df, config, schema)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=config.train_shuffle)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)
    return train_loader, test_loader, schema


def build_train_auto_val_test_dataloaders(config: UrbanOMEConfig):
    raw_train_df = read_table(config.train_csv)
    raw_test_df = read_table(config.test_csv)
    meta_val_task_keys = list(config.meta_val_task_keys or [])

    if meta_val_task_keys:
        raw_train_df, raw_val_df = split_train_dataframe_as_val(raw_train_df, config)
        train_df = _prepare_dataframe(
            raw_train_df,
            config,
            is_train=True,
            exclude_task_keys=meta_val_task_keys,
        )
        val_df = _prepare_dataframe(
            raw_val_df,
            config,
            is_train=False,
            override_task_keys=meta_val_task_keys,
        )
        test_df = _prepare_dataframe(raw_test_df, config, is_train=False)
    else:
        train_df = _prepare_dataframe(raw_train_df, config, is_train=True)
        test_df = _prepare_dataframe(raw_test_df, config, is_train=False)
        train_df, val_df = split_train_dataframe_as_val(train_df, config)

    if config.use_street_view and not all(col in train_df.columns for col in config.street_view_cols):
        config.use_street_view = False
    if config.use_tabular:
        available_tabular_cols = [col for col in config.tabular_feature_cols if col in train_df.columns]
        config.tabular_feature_cols = available_tabular_cols
        if len(config.tabular_feature_cols) == 0:
            config.use_tabular = False
    if config.use_satellite:
        if config.use_cached_satellite_embeddings:
            has_cached = config.satellite_embedding_col in train_df.columns
            has_raw = config.satellite_col in train_df.columns
            if not has_cached and not has_raw:
                raise ValueError(
                    "Missing satellite inputs: need either cached embedding column "
                    f"{config.satellite_embedding_col} or raw image column {config.satellite_col}"
                )
        elif config.satellite_col not in train_df.columns:
            raise ValueError(f"Missing satellite column: {config.satellite_col}")

    schema = load_schema(train_df, pd.concat([val_df, test_df], ignore_index=True), config)
    config.routing_context_dim = len(schema.routing_context_cols)
    if config.use_satellite and config.use_cached_satellite_embeddings:
        inferred_sat_dim = infer_cached_embedding_dim(train_df, config.satellite_embedding_col)
        config.satellite_in_dim = inferred_sat_dim if inferred_sat_dim is not None else 3
    else:
        config.satellite_in_dim = 3 if config.use_satellite else 0
    config.street_view_in_dim = 3 if config.use_street_view else 0
    config.tabular_in_dim = len(config.tabular_feature_cols) if config.use_tabular else 0
    config.task_names = list(schema.target_cols)
    config.task_prompt_templates = dict(schema.task_prompt_templates)
    config.task_prompt_embeddings = schema.task_prompt_embeddings
    config.country_label_names = list(schema.country_label_names)
    config.region_label_names = list(schema.region_label_names)
    config.income_label_names = list(schema.income_label_names)
    config.hemisphere_label_names = list(schema.hemisphere_label_names)

    train_ds = UrbanCSVData(train_df, config, schema)
    val_ds = UrbanCSVData(val_df, config, schema)
    test_ds = UrbanCSVData(test_df, config, schema)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=config.train_shuffle)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, schema


def build_train_val_test_dataloaders(config: UrbanOMEConfig):
    raw_train_df = read_table(config.train_csv)
    raw_val_df = read_table(config.val_csv)
    raw_test_df = read_table(config.test_csv)
    meta_val_task_keys = list(config.meta_val_task_keys or [])

    if meta_val_task_keys:
        train_df = _prepare_dataframe(
            raw_train_df,
            config,
            is_train=True,
            exclude_task_keys=meta_val_task_keys,
        )
        val_df = _prepare_dataframe(
            raw_val_df,
            config,
            is_train=False,
            override_task_keys=meta_val_task_keys,
        )
        test_df = _prepare_dataframe(raw_test_df, config, is_train=False)
    else:
        train_df = _prepare_dataframe(raw_train_df, config, is_train=True)
        val_df = _prepare_dataframe(raw_val_df, config, is_train=False)
        test_df = _prepare_dataframe(raw_test_df, config, is_train=False)

    if config.use_street_view and not all(col in train_df.columns for col in config.street_view_cols):
        config.use_street_view = False
    if config.use_tabular:
        available_tabular_cols = [col for col in config.tabular_feature_cols if col in train_df.columns]
        config.tabular_feature_cols = available_tabular_cols
        if len(config.tabular_feature_cols) == 0:
            config.use_tabular = False
    if config.use_satellite:
        if config.use_cached_satellite_embeddings:
            has_cached = config.satellite_embedding_col in train_df.columns
            has_raw = config.satellite_col in train_df.columns
            if not has_cached and not has_raw:
                raise ValueError(
                    "Missing satellite inputs: need either cached embedding column "
                    f"{config.satellite_embedding_col} or raw image column {config.satellite_col}"
                )
        elif config.satellite_col not in train_df.columns:
            raise ValueError(f"Missing satellite column: {config.satellite_col}")

    schema = load_schema(train_df, pd.concat([val_df, test_df], ignore_index=True), config)
    config.routing_context_dim = len(schema.routing_context_cols)
    if config.use_satellite and config.use_cached_satellite_embeddings:
        inferred_sat_dim = infer_cached_embedding_dim(train_df, config.satellite_embedding_col)
        config.satellite_in_dim = inferred_sat_dim if inferred_sat_dim is not None else 3
    else:
        config.satellite_in_dim = 3 if config.use_satellite else 0
    config.street_view_in_dim = 3 if config.use_street_view else 0
    config.tabular_in_dim = len(config.tabular_feature_cols) if config.use_tabular else 0
    config.task_names = list(schema.target_cols)
    config.task_prompt_templates = dict(schema.task_prompt_templates)
    config.task_prompt_embeddings = schema.task_prompt_embeddings
    config.country_label_names = list(schema.country_label_names)
    config.region_label_names = list(schema.region_label_names)
    config.income_label_names = list(schema.income_label_names)
    config.hemisphere_label_names = list(schema.hemisphere_label_names)

    train_ds = UrbanCSVData(train_df, config, schema)
    val_ds = UrbanCSVData(val_df, config, schema)
    test_ds = UrbanCSVData(test_df, config, schema)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=config.train_shuffle)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)
    return train_loader, val_loader, test_loader, schema
