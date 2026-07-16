"""Dataset-backed evaluation for an aggregated LoRA adapter."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from utils.logger import get_logger

logger = get_logger("evaluation")


@dataclass
class AdapterEvaluationResult:
    round_id: int
    adapter_version: str
    evaluation_loss: Optional[float]
    num_eval_samples: int
    passed: bool
    evaluated: bool
    previous_loss: Optional[float] = None
    improvement: Optional[float] = None
    reason: Optional[str] = None


def _load_eval_texts(path: Path, text_column: str) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"LoRA evaluation dataset not found: {path}")
    if path.is_dir():
        texts = [
            item.read_text(encoding="utf-8")
            for item in sorted(path.rglob("*.txt"))
        ]
    elif path.suffix.lower() == ".jsonl":
        texts = []
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                if not isinstance(row, dict) or text_column not in row:
                    raise ValueError(
                        f"JSONL evaluation row {line_number} lacks {text_column!r}"
                    )
                texts.append(str(row[text_column]))
    elif path.suffix.lower() == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(value, dict) and "data" in value:
            value = value["data"]
        if not isinstance(value, list):
            raise ValueError("Evaluation JSON must be a list or {'data': [...]}")
        texts = [
            str(row[text_column])
            for row in value
            if isinstance(row, dict) and text_column in row
        ]
    else:
        texts = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    max_samples = max(1, int(os.getenv("LORA_EVAL_MAX_SAMPLES", "64")))
    texts = texts[:max_samples]
    if not texts:
        raise ValueError("LoRA evaluation dataset contains no text samples")
    return texts


def evaluate_adapter(
    round_id: int,
    adapter_version: str,
    aggregated_adapter: Dict[str, Any],
    *,
    base_model_name: str,
    lora_r: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: List[str],
    max_seq_length: int,
    previous_adapter_loss: Optional[float] = None,
    dataset_path: Optional[str] = None,
) -> AdapterEvaluationResult:
    """Load the base model + aggregate and calculate causal-LM holdout loss.

    If no evaluation dataset is configured, evaluation is explicitly skipped;
    no parameter-norm proxy or fabricated loss is returned.
    """
    configured_path = (
        dataset_path or os.getenv("LORA_EVAL_DATASET_PATH", "")
    ).strip()
    if not configured_path:
        reason = "LORA_EVAL_DATASET_PATH is not configured"
        if os.getenv("LORA_REQUIRE_EVALUATION", "false").lower() in {
            "1",
            "true",
            "yes",
        }:
            raise ValueError(reason)
        return AdapterEvaluationResult(
            round_id=round_id,
            adapter_version=adapter_version,
            evaluation_loss=None,
            num_eval_samples=0,
            passed=False,
            evaluated=False,
            previous_loss=previous_adapter_loss,
            reason=reason,
        )
    if not aggregated_adapter:
        raise ValueError("Cannot evaluate an empty adapter")

    try:
        import torch
        from peft import (
            LoraConfig,
            TaskType,
            get_peft_model,
            set_peft_model_state_dict,
        )
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Real LoRA evaluation requires torch, transformers, and peft"
        ) from exc

    texts = _load_eval_texts(
        Path(configured_path).expanduser().resolve(),
        os.getenv("LORA_EVAL_TEXT_COLUMN", "text"),
    )
    device_name = os.getenv(
        "LORA_EVAL_DEVICE",
        "cuda" if torch.cuda.is_available() else "cpu",
    )
    device = torch.device(device_name)
    dtype = torch.float16 if device.type == "cuda" else torch.float32

    logger.info(
        f"Evaluating adapter {adapter_version} on {len(texts)} samples",
        extra={
            "component": "evaluation",
            "event": "evaluation_started",
            "round_id": round_id,
            "base_model": base_model_name,
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_name,
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        trust_remote_code=False,
        torch_dtype=dtype,
    )
    peft_model = get_peft_model(
        model,
        LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        ),
    )
    state = {
        key: torch.tensor(value)
        for key, value in aggregated_adapter.items()
    }
    load_result = set_peft_model_state_dict(peft_model, state)
    unexpected = getattr(load_result, "unexpected_keys", [])
    if unexpected:
        raise ValueError(f"Aggregated adapter has unexpected keys: {unexpected[:5]}")
    peft_model.to(device)
    peft_model.eval()

    batch_size = max(1, int(os.getenv("LORA_EVAL_BATCH_SIZE", "2")))
    weighted_loss = 0.0
    evaluated = 0
    with torch.no_grad():
        for offset in range(0, len(texts), batch_size):
            batch = texts[offset : offset + batch_size]
            tokens = tokenizer(
                batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_seq_length,
            )
            tokens = {key: value.to(device) for key, value in tokens.items()}
            labels = tokens["input_ids"].clone()
            labels[tokens["attention_mask"] == 0] = -100
            output = peft_model(**tokens, labels=labels)
            weighted_loss += float(output.loss.item()) * len(batch)
            evaluated += len(batch)

    loss = weighted_loss / evaluated
    improvement = (
        loss - previous_adapter_loss
        if previous_adapter_loss is not None
        else None
    )
    tolerance = float(os.getenv("LORA_EVAL_REGRESSION_TOLERANCE", "0.1"))
    passed = improvement is None or improvement <= tolerance
    return AdapterEvaluationResult(
        round_id=round_id,
        adapter_version=adapter_version,
        evaluation_loss=loss,
        num_eval_samples=evaluated,
        passed=passed,
        evaluated=True,
        previous_loss=previous_adapter_loss,
        improvement=improvement,
    )
