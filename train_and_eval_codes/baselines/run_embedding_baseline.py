import argparse
import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from pandas.api.types import is_numeric_dtype


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_table(path: str) -> pd.DataFrame:
    if str(path).lower().endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def load_feature_store(path: str) -> Dict[str, Dict]:
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Feature store must be a dict: {path}")
    return {k: v for k, v in payload.items() if not str(k).startswith("__")}


def load_feature_stores(paths: List[str]) -> Dict[str, Dict]:
    combined: Dict[str, Dict] = {}
    for path in paths:
        payload = load_feature_store(path)
        overlap = set(combined.keys()) & set(payload.keys())
        if overlap:
            preview = ", ".join(sorted(list(overlap))[:5])
            raise ValueError(
                f"Duplicate sample_ids across feature stores. file={path}, first_duplicates={preview}"
            )
        combined.update(payload)
    return combined


def parse_embedding_value(value: Any) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if torch.is_tensor(value):
        return value.detach().cpu().float().view(-1)
    if isinstance(value, np.ndarray):
        return torch.tensor(value, dtype=torch.float32).view(-1)
    if isinstance(value, list):
        return torch.tensor(value, dtype=torch.float32).view(-1)
    if isinstance(value, tuple):
        return torch.tensor(list(value), dtype=torch.float32).view(-1)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
            if isinstance(parsed, list):
                return torch.tensor(parsed, dtype=torch.float32).view(-1)
        except json.JSONDecodeError:
            pass
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return None


def build_sample_id(row: pd.Series, row_idx: int) -> str:
    parts = []
    for col in ("city", "indicator_column", "tile_x", "tile_y", "image_name", "sat_image_name"):
        if col in row.index and pd.notna(row[col]):
            parts.append(str(row[col]))
    if not parts:
        return f"sample_{row_idx:08d}"
    return "__".join(parts) + f"__row{row_idx:08d}"


def split_train_as_val(train_df: pd.DataFrame, split_by: str, val_ratio: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    split_col = "city" if split_by == "city" else "country"
    if split_col not in train_df.columns:
        raise ValueError(f"Missing split column for val split: {split_col}")
    groups = [str(x) for x in train_df[split_col].dropna().astype(str).unique().tolist() if str(x).strip()]
    if len(groups) < 2:
        raise ValueError(f"Need at least 2 unique {split_col} values for train/val split.")
    rng = np.random.default_rng(seed)
    permuted = list(rng.permutation(groups))
    num_val_groups = max(1, int(round(len(permuted) * val_ratio)))
    num_val_groups = min(num_val_groups, len(permuted) - 1)
    val_groups = set(permuted[:num_val_groups])
    val_df = train_df[train_df[split_col].astype(str).isin(val_groups)].copy().reset_index(drop=True)
    inner_train_df = train_df[~train_df[split_col].astype(str).isin(val_groups)].copy().reset_index(drop=True)
    return inner_train_df, val_df


def compute_group_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {"mse": 0.0, "mae": 0.0}
    pred = np.asarray([row["pred"] for row in rows], dtype=np.float64)
    target = np.asarray([row["target"] for row in rows], dtype=np.float64)
    diff = pred - target
    return {
        "mse": float(np.mean(diff ** 2)),
        "mae": float(np.mean(np.abs(diff))),
    }


def compute_scalar_metrics(pred: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    diff = pred - target
    mse = float(np.mean(diff ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(diff)))
    bias = float(np.mean(diff))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    if target.size >= 1 and ss_tot > 1e-12:
        r2 = float(1.0 - np.sum(diff ** 2) / ss_tot)
    else:
        r2 = 0.0
    if pred.size >= 2 and np.std(pred) > 1e-12 and np.std(target) > 1e-12:
        pearson = float(np.corrcoef(pred, target)[0, 1])
    else:
        pearson = 0.0
    rank_pred = pd.Series(pred).rank(method="average").to_numpy(dtype=np.float64)
    rank_target = pd.Series(target).rank(method="average").to_numpy(dtype=np.float64)
    if pred.size >= 2 and np.std(rank_pred) > 1e-12 and np.std(rank_target) > 1e-12:
        spearman = float(np.corrcoef(rank_pred, rank_target)[0, 1])
    else:
        spearman = 0.0
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "bias": bias,
        "pearson": pearson,
        "spearman": spearman,
        "pred_mean": float(np.mean(pred)),
        "target_mean": float(np.mean(target)),
    }


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


def format_metrics_table(metrics: Dict[str, float]) -> List[str]:
    lines = [
        (
            f"test_grid_mse={metrics.get('grid_mse', 0.0):.3f} "
            f"test_grid_mae={metrics.get('grid_mae', 0.0):.3f} "
            f"test_grid_R2={metrics.get('grid_r2', 0.0):.3f}"
        )
    ]
    for task_key, task_metrics in metrics.get("per_target", {}).items():
        if int(task_metrics.get("count", 0)) <= 0:
            continue
        lines.append(
            f"{task_key} mse={task_metrics.get('mse', 0.0):.3f} "
            f"mae={task_metrics.get('mae', 0.0):.3f} "
            f"R2={task_metrics.get('r2', 0.0):.3f}"
        )
    return lines


@dataclass
class BaselineConfig:
    train_csv: str
    test_csv: str
    feature_pt: Optional[str]
    feature_pts: Optional[List[str]]
    save_dir: str
    model_type: str
    setting: str
    target_value_col: str
    cached_embedding_col: str
    train_task_keys: Optional[List[str]]
    test_task_keys: Optional[List[str]]
    include_task_keys: Optional[List[str]]
    prompt_embeddings_json: str
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float
    hidden_dim: int
    dropout: float
    random_seed: int
    val_ratio: float
    val_split_by: str
    checkpoint_metric: str
    early_stop_patience: int
    early_stop_min_delta: float
    save_history: bool


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feature_store: Optional[Dict[str, Dict]],
        prompt_embeddings: Optional[Dict[str, List[float]]],
        model_type: str,
        target_value_col: str,
        cached_embedding_col: str,
    ):
        self.rows = []
        self.model_type = model_type
        self.prompt_embeddings = prompt_embeddings or {}
        missing_features = 0
        missing_prompts = set()

        work_df = df.copy()
        if "__row_id__" not in work_df.columns:
            work_df["__row_id__"] = np.arange(len(work_df), dtype=np.int64)

        for _, row in work_df.iterrows():
            sample_row_id = int(row["__row_id__"])
            sample_id = build_sample_id(row, sample_row_id)
            sat = None
            if feature_store is not None:
                feat_payload = feature_store.get(sample_id)
                if feat_payload is None:
                    missing_features += 1
                    continue
                sat = feat_payload.get("sat")
                if sat is None:
                    missing_features += 1
                    continue
                if not torch.is_tensor(sat):
                    sat = torch.tensor(sat, dtype=torch.float32)
                sat = sat.float().view(-1)
            else:
                if cached_embedding_col not in row.index:
                    raise ValueError(
                        f"Missing cached embedding column '{cached_embedding_col}' in input table."
                    )
                sat = parse_embedding_value(row[cached_embedding_col])
                if sat is None:
                    missing_features += 1
                    continue

            task_key = str(row["indicator_column"])
            prompt_vec = None
            if model_type == "prompt_mlp":
                if task_key not in self.prompt_embeddings:
                    missing_prompts.add(task_key)
                    continue
                prompt_vec = torch.tensor(self.prompt_embeddings[task_key], dtype=torch.float32).view(-1)

            target = pd.to_numeric(pd.Series([row[target_value_col]]), errors="coerce").iloc[0]
            if pd.isna(target):
                continue

            self.rows.append(
                {
                    "sample_id": sample_id,
                    "x": sat,
                    "prompt_x": prompt_vec,
                    "y": float(target),
                    "task_key": task_key,
                    "city": str(row["city"]) if "city" in row.index else "",
                    "country": str(row["country"]) if "country" in row.index else "",
                }
            )

        if missing_prompts:
            raise ValueError(f"Missing prompt embeddings for tasks: {sorted(missing_prompts)}")
        if len(self.rows) == 0:
            raise ValueError(
                f"No usable rows after matching features/targets. missing_features={missing_features}, "
                f"input_rows={len(df)}"
            )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        prompt_x = row["prompt_x"] if row["prompt_x"] is not None else torch.zeros(0, dtype=torch.float32)
        return {
            "x": row["x"],
            "prompt_x": prompt_x,
            "y": torch.tensor(row["y"], dtype=torch.float32),
            "sample_id": row["sample_id"],
            "task_key": row["task_key"],
            "city": row["city"],
            "country": row["country"],
        }


class MLPRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class PromptMLPRegressor(nn.Module):
    def __init__(self, feature_dim: int, prompt_dim: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.prompt_proj = nn.Sequential(
            nn.Linear(prompt_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.net = nn.Sequential(
            nn.Linear(feature_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor, prompt_x: torch.Tensor) -> torch.Tensor:
        prompt_h = self.prompt_proj(prompt_x)
        return self.net(torch.cat([x, prompt_h], dim=-1)).squeeze(-1)


def build_model(model_type: str, dataset: EmbeddingDataset, hidden_dim: int, dropout: float) -> nn.Module:
    feature_dim = int(dataset.rows[0]["x"].numel())
    if model_type == "mlp":
        return MLPRegressor(feature_dim, hidden_dim, dropout)
    prompt_dim = int(dataset.rows[0]["prompt_x"].numel())
    return PromptMLPRegressor(feature_dim, prompt_dim, hidden_dim, dropout)


def filter_df_by_tasks(df: pd.DataFrame, task_keys: Optional[List[str]]) -> pd.DataFrame:
    work_df = df.copy()
    if "__row_id__" not in work_df.columns:
        work_df["__row_id__"] = np.arange(len(work_df), dtype=np.int64)
    if not task_keys:
        return work_df
    keep = set(task_keys)
    return work_df[work_df["indicator_column"].astype(str).isin(keep)].copy()


def build_task_lists(config: BaselineConfig) -> Tuple[Optional[List[str]], Optional[List[str]]]:
    if config.setting == "seen_indicator":
        train_task_keys = config.train_task_keys or config.include_task_keys
        test_task_keys = train_task_keys
    else:
        train_task_keys = config.train_task_keys or config.include_task_keys
        test_task_keys = config.test_task_keys
    return train_task_keys, test_task_keys


def validate_task_protocol(
    config: BaselineConfig,
    train_task_keys: Optional[List[str]],
    test_task_keys: Optional[List[str]],
) -> None:
    if config.setting != "unseen_indicator":
        return
    if not train_task_keys:
        raise ValueError("unseen_indicator setting requires non-empty --train_task_keys or --include_task_keys")
    if not test_task_keys:
        raise ValueError("unseen_indicator setting requires non-empty --test_task_keys")
    overlap = sorted(set(train_task_keys) & set(test_task_keys))
    if overlap:
        raise ValueError(
            "unseen_indicator setting requires disjoint train/test task keys. "
            f"Overlapping tasks: {overlap}"
        )


def run_epoch(model: nn.Module, data_loader: DataLoader, optimizer, device: torch.device, model_type: str) -> float:
    model.train()
    losses = []
    for batch in data_loader:
        optimizer.zero_grad()
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        if model_type == "prompt_mlp":
            pred = model(x, batch["prompt_x"].to(device))
        else:
            pred = model(x)
        loss = nn.functional.mse_loss(pred, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device, model_type: str) -> Tuple[Dict[str, float], List[Dict[str, object]]]:
    model.eval()
    rows = []
    for batch in data_loader:
        x = batch["x"].to(device)
        if model_type == "prompt_mlp":
            pred = model(x, batch["prompt_x"].to(device))
        else:
            pred = model(x)
        pred = pred.detach().cpu().numpy()
        target = batch["y"].cpu().numpy()
        for idx in range(len(pred)):
            rows.append(
                {
                    "sample_id": batch["sample_id"][idx],
                    "task_key": batch["task_key"][idx],
                    "city": batch["city"][idx],
                    "country": batch["country"][idx],
                    "pred": float(pred[idx]),
                    "target": float(target[idx]),
                }
            )

    pred_arr = np.asarray([row["pred"] for row in rows], dtype=np.float64)
    target_arr = np.asarray([row["target"] for row in rows], dtype=np.float64)
    scalar_metrics = compute_scalar_metrics(pred_arr, target_arr) if len(rows) > 0 else {"mse": 0.0, "rmse": 0.0, "mae": 0.0, "bias": 0.0, "pearson": 0.0, "spearman": 0.0, "pred_mean": 0.0, "target_mean": 0.0}

    city_rows = defaultdict(list)
    country_rows = defaultdict(list)
    per_target = defaultdict(list)
    for row in rows:
        city_rows[row["city"]].append(row)
        country_rows[row["country"]].append(row)
        per_target[row["task_key"]].append(row)

    city_pred = np.asarray([np.mean([r["pred"] for r in group]) for group in city_rows.values()], dtype=np.float64) if city_rows else np.asarray([], dtype=np.float64)
    city_target = np.asarray([np.mean([r["target"] for r in group]) for group in city_rows.values()], dtype=np.float64) if city_rows else np.asarray([], dtype=np.float64)
    country_pred = np.asarray([np.mean([r["pred"] for r in group]) for group in country_rows.values()], dtype=np.float64) if country_rows else np.asarray([], dtype=np.float64)
    country_target = np.asarray([np.mean([r["target"] for r in group]) for group in country_rows.values()], dtype=np.float64) if country_rows else np.asarray([], dtype=np.float64)
    city_scalar_metrics = compute_scalar_metrics(city_pred, city_target) if city_pred.size > 0 else {"mse": 0.0, "mae": 0.0, "r2": 0.0}
    country_scalar_metrics = compute_scalar_metrics(country_pred, country_target) if country_pred.size > 0 else {"mse": 0.0, "mae": 0.0, "r2": 0.0}
    per_target_metrics = {}
    for task_key, group in per_target.items():
        per_target_metrics[task_key] = {"count": len(group), **compute_scalar_metrics(np.asarray([r["pred"] for r in group]), np.asarray([r["target"] for r in group]))}

    metrics = {
        "grid_mse": scalar_metrics["mse"],
        "grid_rmse": scalar_metrics["rmse"],
        "grid_mae": scalar_metrics["mae"],
        "grid_r2": scalar_metrics["r2"],
        "grid_pearson": scalar_metrics["pearson"],
        "grid_spearman": scalar_metrics["spearman"],
        "city_mse": city_scalar_metrics["mse"],
        "city_mae": city_scalar_metrics["mae"],
        "city_r2": city_scalar_metrics["r2"],
        "country_mse": country_scalar_metrics["mse"],
        "country_mae": country_scalar_metrics["mae"],
        "country_r2": country_scalar_metrics["r2"],
        "per_target": per_target_metrics,
    }
    return metrics, rows


def save_predictions(rows: List[Dict[str, object]], output_csv: str) -> None:
    df = pd.DataFrame(rows)
    if len(df) == 0:
        df = pd.DataFrame(columns=["sample_id", "task_key", "city", "country", "pred", "target"])
    df["error"] = df["pred"] - df["target"] if len(df) > 0 else []
    round_dataframe_numeric(df, decimals=3).to_csv(output_csv, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embedding baseline runner for seen/unseen indicator experiments.")
    parser.add_argument("--train_csv", required=True)
    parser.add_argument("--test_csv", required=True)
    parser.add_argument("--feature_pt", default="")
    parser.add_argument("--feature_pts", nargs="+", default=None)
    parser.add_argument("--cached_embedding_col", default="satellite_embedding")
    parser.add_argument("--save_dir", default="baseline_outputs")
    parser.add_argument("--model_type", choices=["mlp", "prompt_mlp"], default="mlp")
    parser.add_argument("--setting", choices=["seen_indicator", "unseen_indicator"], default="seen_indicator")
    parser.add_argument("--train_task_keys", nargs="+", default=None)
    parser.add_argument("--test_task_keys", nargs="+", default=None)
    parser.add_argument("--include_task_keys", nargs="+", default=None)
    parser.add_argument("--prompt_embeddings_json", default="")
    parser.add_argument("--target_value_col", default="indicator_value_log1p")
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--val_split_by", choices=["city", "country"], default="city")
    parser.add_argument("--checkpoint_metric", choices=["grid_mse", "city_mse", "country_mse"], default="grid_mse")
    parser.add_argument("--early_stop_patience", type=int, default=10)
    parser.add_argument("--early_stop_min_delta", type=float, default=0.0)
    parser.add_argument("--disable_history", action="store_true")
    parser.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.random_seed))
    os.makedirs(args.save_dir, exist_ok=True)

    config = BaselineConfig(
        train_csv=args.train_csv,
        test_csv=args.test_csv,
        feature_pt=(args.feature_pt or None),
        feature_pts=args.feature_pts,
        save_dir=args.save_dir,
        model_type=args.model_type,
        setting=args.setting,
        target_value_col=args.target_value_col,
        cached_embedding_col=args.cached_embedding_col,
        train_task_keys=args.train_task_keys,
        test_task_keys=args.test_task_keys,
        include_task_keys=args.include_task_keys,
        prompt_embeddings_json=args.prompt_embeddings_json,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        random_seed=args.random_seed,
        val_ratio=args.val_ratio,
        val_split_by=args.val_split_by,
        checkpoint_metric=args.checkpoint_metric,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        save_history=not args.disable_history,
    )

    prompt_embeddings = None
    if config.model_type == "prompt_mlp":
        if not config.prompt_embeddings_json:
            raise ValueError("prompt_mlp requires --prompt_embeddings_json")
        with open(config.prompt_embeddings_json, "r", encoding="utf-8") as f:
            prompt_embeddings = json.load(f)

    train_df = load_table(config.train_csv)
    test_df = load_table(config.test_csv)

    feature_store: Optional[Dict[str, Dict]] = None
    if config.feature_pts:
        feature_store = load_feature_stores(config.feature_pts)
    elif config.feature_pt:
        feature_store = load_feature_store(config.feature_pt)
    else:
        feature_store = None

    train_task_keys, test_task_keys = build_task_lists(config)
    validate_task_protocol(config, train_task_keys, test_task_keys)
    train_df = filter_df_by_tasks(train_df, train_task_keys)
    test_df = filter_df_by_tasks(test_df, test_task_keys)
    train_df, val_df = split_train_as_val(train_df, config.val_split_by, config.val_ratio, config.random_seed)

    train_ds = EmbeddingDataset(
        train_df,
        feature_store,
        prompt_embeddings,
        config.model_type,
        config.target_value_col,
        config.cached_embedding_col,
    )
    val_ds = EmbeddingDataset(
        val_df,
        feature_store,
        prompt_embeddings,
        config.model_type,
        config.target_value_col,
        config.cached_embedding_col,
    )
    test_ds = EmbeddingDataset(
        test_df,
        feature_store,
        prompt_embeddings,
        config.model_type,
        config.target_value_col,
        config.cached_embedding_col,
    )

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size, shuffle=False)

    device = torch.device(args.device)
    model = build_model(config.model_type, train_ds, config.hidden_dim, config.dropout).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    best_metric = float("inf")
    best_ckpt_path = os.path.join(config.save_dir, "best_model.pt")
    epochs_without_improvement = 0
    history_jsonl_path = os.path.join(config.save_dir, "train_history.jsonl")
    if config.save_history and os.path.exists(history_jsonl_path):
        os.remove(history_jsonl_path)

    print("setting=", config.setting)
    print("model_type=", config.model_type)
    print("feature_mode=", "pt_store" if feature_store is not None else "cached_embedding_column")
    print("feature_pt=", config.feature_pt)
    print("feature_pts=", json.dumps(config.feature_pts, ensure_ascii=False) if config.feature_pts else "[]")
    print("cached_embedding_col=", config.cached_embedding_col)
    print("target_value_col=", config.target_value_col)
    print("train_rows=", len(train_ds))
    print("val_rows=", len(val_ds))
    print("test_rows=", len(test_ds))
    print("train_task_keys=", json.dumps(train_task_keys, ensure_ascii=False))
    print("test_task_keys=", json.dumps(test_task_keys, ensure_ascii=False))
    print("val_ratio=", config.val_ratio)
    print("val_split_by=", config.val_split_by)
    print("checkpoint_metric=", config.checkpoint_metric)
    print("early_stop_patience=", config.early_stop_patience)
    print("early_stop_min_delta=", config.early_stop_min_delta)
    print("protocol_note=", "test task labels are used only for final evaluation, never for training or validation selection")

    for epoch in range(config.num_epochs):
        train_loss = run_epoch(model, train_loader, optimizer, device, config.model_type)
        val_metrics, _ = evaluate(model, val_loader, device, config.model_type)
        current_metric = float(val_metrics[config.checkpoint_metric])
        is_best = current_metric < (best_metric - config.early_stop_min_delta)
        if is_best:
            best_metric = current_metric
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "metrics": val_metrics,
                    "config": vars(args),
                },
                best_ckpt_path,
            )
        else:
            epochs_without_improvement += 1
        if config.save_history:
            history_row = {
                "epoch": epoch + 1,
                "train_loss": float(train_loss),
                "val_grid_mse": float(val_metrics["grid_mse"]),
                "val_city_mse": float(val_metrics["city_mse"]),
                "val_country_mse": float(val_metrics["country_mse"]),
                "checkpoint_metric": config.checkpoint_metric,
                "checkpoint_metric_value": float(current_metric),
                "epochs_without_improvement": int(epochs_without_improvement),
                "is_best": bool(is_best),
            }
            with open(history_jsonl_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(round_nested_metrics(history_row, decimals=6), ensure_ascii=False) + "\n")
        print(
            f"epoch={epoch + 1} "
            f"train_loss={train_loss:.3f} "
            f"val_grid_mse={val_metrics['grid_mse']:.3f} "
            f"val_city_mse={val_metrics['city_mse']:.3f} "
            f"val_country_mse={val_metrics['country_mse']:.3f}"
        )
        if config.early_stop_patience > 0 and epochs_without_improvement >= config.early_stop_patience:
            print(
                f"early_stopping_triggered epoch={epoch + 1} "
                f"best_metric={best_metric:.6f} "
                f"epochs_without_improvement={epochs_without_improvement}"
            )
            break

    if os.path.exists(best_ckpt_path):
        checkpoint = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])

    test_metrics, test_rows = evaluate(model, test_loader, device, config.model_type)
    prediction_csv = os.path.join(config.save_dir, "test_predictions.csv")
    save_predictions(test_rows, prediction_csv)
    rounded_test_metrics = round_nested_metrics(test_metrics, decimals=3)
    for line in format_metrics_table(rounded_test_metrics):
        print(line)
    print("best_checkpoint=", best_ckpt_path)
    print("prediction_csv=", prediction_csv)


if __name__ == "__main__":
    main()
