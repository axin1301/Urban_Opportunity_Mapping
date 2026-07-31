import json
import os
import re
from typing import Dict, List, Optional

import torch
from torch import nn


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def tokenize_prompt(text: str) -> List[str]:
    return TOKEN_PATTERN.findall(text.lower())


def hash_token(token: str, vocab_size: int) -> int:
    return abs(hash(token)) % vocab_size


def resolve_task_prompt_templates(
    raw_prompt_value: str,
    task_names: List[str],
    default_template: str,
) -> Dict[str, str]:
    if raw_prompt_value:
        try:
            parsed = json.loads(raw_prompt_value)
            if isinstance(parsed, dict):
                return {
                    task_name: str(parsed.get(task_name, default_template.format(task=task_name)))
                    for task_name in task_names
                }
        except json.JSONDecodeError:
            pass

        if "{task}" in raw_prompt_value:
            return {
                task_name: raw_prompt_value.format(task=task_name)
                for task_name in task_names
            }

    return {
        task_name: default_template.format(task=task_name)
        for task_name in task_names
    }


def resolve_task_prompt_embeddings(
    raw_embedding_value: str,
    task_names: List[str],
    external_path: str = "",
) -> Optional[Dict[str, List[float]]]:
    payload = None

    if external_path:
        with open(external_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
    elif raw_embedding_value:
        try:
            payload = json.loads(raw_embedding_value)
        except json.JSONDecodeError:
            payload = None

    if not isinstance(payload, dict):
        return None

    resolved = {}
    for task_name in task_names:
        if task_name not in payload:
            continue
        vector = payload[task_name]
        if isinstance(vector, list):
            resolved[task_name] = [float(x) for x in vector]
    return resolved or None


class LightweightPromptEncoder(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.EmbeddingBag(vocab_size, embedding_dim, mode="mean")
        self.proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def encode_prompt_list(self, prompts: List[str], device: torch.device) -> torch.Tensor:
        token_ids = []
        offsets = [0]
        for prompt in prompts:
            prompt_tokens = tokenize_prompt(prompt)
            hashed = [hash_token(token, self.vocab_size) for token in prompt_tokens]
            if not hashed:
                hashed = [0]
            token_ids.extend(hashed)
            offsets.append(offsets[-1] + len(hashed))

        token_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
        offset_tensor = torch.tensor(offsets[:-1], dtype=torch.long, device=device)
        return self.proj(self.embedding(token_tensor, offset_tensor))


class ExternalEmbeddingPromptEncoder(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.proj = None

    def ensure_proj(self, input_dim: int, device: torch.device):
        if self.proj is None:
            self.proj = nn.Sequential(
                nn.Linear(input_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.GELU(),
                self.dropout,
            ).to(device)
        return self.proj

    def encode_prompt_embeddings(
        self,
        embeddings: Dict[str, List[float]],
        task_names: List[str],
        device: torch.device,
    ) -> torch.Tensor:
        vectors = []
        for task_name in task_names:
            if task_name not in embeddings:
                raise ValueError(f"Missing external task embedding for task: {task_name}")
            vectors.append(embeddings[task_name])

        tensor = torch.tensor(vectors, dtype=torch.float32, device=device)
        self.ensure_proj(tensor.shape[-1], device)
        return self.proj(tensor)


class GPT2EmbeddingPromptEncoder(nn.Module):
    def __init__(self, model_name: str, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        try:
            from transformers import GPT2Model, GPT2TokenizerFast
        except ImportError as exc:
            raise ImportError(
                "prompt_encoder_type='gpt2_embedding' requires transformers to be installed."
            ) from exc

        self.tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.gpt2 = GPT2Model.from_pretrained(model_name)
        for param in self.gpt2.parameters():
            param.requires_grad = False
        self.proj = nn.Sequential(
            nn.Linear(self.gpt2.config.hidden_size, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def encode_prompt_list(self, prompts: List[str], device: torch.device) -> torch.Tensor:
        encoded = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        encoded = {k: v.to(device) for k, v in encoded.items()}
        outputs = self.gpt2(**encoded)
        hidden = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1)
        pooled = (hidden * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp_min(1)
        return self.proj(pooled)


def build_prompt_encoder(
    prompt_encoder_type: str,
    vocab_size: int,
    embedding_dim: int,
    hidden_dim: int,
    dropout: float = 0.1,
    gpt2_model_name: str = "gpt2",
) -> nn.Module:
    if prompt_encoder_type == "lightweight":
        return LightweightPromptEncoder(vocab_size, embedding_dim, hidden_dim, dropout)
    if prompt_encoder_type == "external_embedding":
        return ExternalEmbeddingPromptEncoder(hidden_dim, dropout)
    if prompt_encoder_type == "gpt2_embedding":
        return GPT2EmbeddingPromptEncoder(gpt2_model_name, hidden_dim, dropout)
    raise ValueError(f"Unsupported prompt_encoder_type: {prompt_encoder_type}")
