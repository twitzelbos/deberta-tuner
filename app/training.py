"""DeBERTa fine-tuning for the four supported task types.

A hand-rolled loop rather than transformers.Trainer: the per-task differences
(loss shape, label alignment, metrics) stay visible, and there is no dependency
on Trainer's evolving API surface.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)

from . import config
from .data import Dataset
from .schemas import JobConfig, TaskType

# Subword positions that should not contribute to the token-classification loss.
IGNORE_INDEX = -100


class Cancelled(Exception):
    """Raised when a cancel request is observed mid-training."""


def pick_device() -> torch.device:
    if config.FORCE_CPU or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


# --------------------------------------------------------------------------- #
# Datasets
# --------------------------------------------------------------------------- #


class _TextDataset(TorchDataset):
    """Sequence classification, multi-label and regression."""

    def __init__(self, ds: Dataset, tok, cfg: JobConfig, label2id: dict[str, int]):
        self.rows = ds.rows
        self.task = ds.task
        self.tok = tok
        self.cfg = cfg
        self.label2id = label2id

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, Any]:
        row = self.rows[i]
        enc = self.tok(
            row["text"],
            row.get("text_pair"),
            truncation=True,
            max_length=self.cfg.max_length,
        )
        item = {k: enc[k] for k in ("input_ids", "attention_mask") if k in enc}

        if self.task is TaskType.REGRESSION:
            item["labels"] = float(row["label"])
        elif self.task is TaskType.MULTI_LABEL_CLASSIFICATION:
            vec = [0.0] * len(self.label2id)
            for name in row["labels"]:
                vec[self.label2id[name]] = 1.0
            item["labels"] = vec
        else:
            item["labels"] = self.label2id[row["label"]]
        return item


class _TokenDataset(TorchDataset):
    """Token classification, with word -> subword label alignment."""

    def __init__(self, ds: Dataset, tok, cfg: JobConfig, label2id: dict[str, int]):
        self.rows = ds.rows
        self.tok = tok
        self.cfg = cfg
        self.label2id = label2id

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict[str, Any]:
        row = self.rows[i]
        enc = self.tok(
            row["tokens"],
            is_split_into_words=True,
            truncation=True,
            max_length=self.cfg.max_length,
        )
        word_ids = enc.word_ids()
        labels: list[int] = []
        prev = None
        for wid in word_ids:
            if wid is None:
                labels.append(IGNORE_INDEX)          # special token
            elif wid != prev:
                labels.append(self.label2id[row["tags"][wid]])   # first subword
            else:
                labels.append(IGNORE_INDEX)          # continuation subword
            prev = wid
        return {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": labels,
        }


def _make_collate(tok, task: TaskType) -> Callable:
    pad_id = tok.pad_token_id or 0

    def collate(batch: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        width = max(len(b["input_ids"]) for b in batch)
        ids, mask, labels = [], [], []
        for b in batch:
            gap = width - len(b["input_ids"])
            ids.append(b["input_ids"] + [pad_id] * gap)
            mask.append(b["attention_mask"] + [0] * gap)
            if task is TaskType.TOKEN_CLASSIFICATION:
                labels.append(b["labels"] + [IGNORE_INDEX] * gap)
            else:
                labels.append(b["labels"])

        out = {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(mask, dtype=torch.long),
        }
        if task is TaskType.REGRESSION:
            out["labels"] = torch.tensor(labels, dtype=torch.float)
        elif task is TaskType.MULTI_LABEL_CLASSIFICATION:
            out["labels"] = torch.tensor(labels, dtype=torch.float)
        else:
            out["labels"] = torch.tensor(labels, dtype=torch.long)
        return out

    return collate


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def _metrics(task: TaskType, logits: np.ndarray, gold: np.ndarray,
             labels: list[str], threshold: float) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    if task is TaskType.REGRESSION:
        pred = logits.reshape(-1)
        err = pred - gold
        out = {
            "mse": float(np.mean(err ** 2)),
            "mae": float(np.mean(np.abs(err))),
        }
        # Pearson is undefined when either side is constant.
        if pred.std() > 1e-12 and gold.std() > 1e-12:
            out["pearson"] = float(np.corrcoef(pred, gold)[0, 1])
        return out

    if task is TaskType.MULTI_LABEL_CLASSIFICATION:
        prob = 1.0 / (1.0 + np.exp(-logits))
        pred = (prob >= threshold).astype(int)
        return {
            "f1_micro": float(f1_score(gold, pred, average="micro", zero_division=0)),
            "f1_macro": float(f1_score(gold, pred, average="macro", zero_division=0)),
            "subset_accuracy": float((pred == gold).all(axis=1).mean()),
        }

    if task is TaskType.TOKEN_CLASSIFICATION:
        from seqeval.metrics import f1_score as seq_f1
        from seqeval.metrics import precision_score, recall_score

        pred_ids = logits.argmax(-1)
        true_seqs, pred_seqs = [], []
        for p_row, g_row in zip(pred_ids, gold):
            keep = g_row != IGNORE_INDEX
            true_seqs.append([labels[i] for i in g_row[keep]])
            pred_seqs.append([labels[i] for i in p_row[keep]])
        return {
            "precision": float(precision_score(true_seqs, pred_seqs, zero_division=0)),
            "recall": float(recall_score(true_seqs, pred_seqs, zero_division=0)),
            "f1": float(seq_f1(true_seqs, pred_seqs, zero_division=0)),
        }

    pred = logits.argmax(-1)
    return {
        "accuracy": float(accuracy_score(gold, pred)),
        "f1_macro": float(f1_score(gold, pred, average="macro", zero_division=0)),
    }


# --------------------------------------------------------------------------- #
# Train
# --------------------------------------------------------------------------- #


def run(
    train_ds: Dataset,
    eval_ds: Dataset | None,
    labels: list[str],
    cfg: JobConfig,
    out_dir: Path,
    log: Callable[[str], None],
    on_progress: Callable[[float], None],
    should_cancel: Callable[[], bool],
) -> dict[str, Any]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = pick_device()
    log(f"device={device} task={cfg.task.value} base={cfg.base_model}")
    log(f"train={len(train_ds)} eval={len(eval_ds) if eval_ds else 0} labels={labels}")

    tok = AutoTokenizer.from_pretrained(cfg.base_model)
    label2id = {name: i for i, name in enumerate(labels)}

    is_token = cfg.task is TaskType.TOKEN_CLASSIFICATION
    if cfg.task is TaskType.REGRESSION:
        num_labels, problem = 1, "regression"
    elif cfg.task is TaskType.MULTI_LABEL_CLASSIFICATION:
        num_labels, problem = len(labels), "multi_label_classification"
    else:
        num_labels, problem = len(labels), "single_label_classification"

    kwargs: dict[str, Any] = {"num_labels": num_labels}
    if labels:
        kwargs["id2label"] = {i: n for i, n in enumerate(labels)}
        kwargs["label2id"] = label2id
    if not is_token:
        kwargs["problem_type"] = problem

    factory = AutoModelForTokenClassification if is_token else AutoModelForSequenceClassification
    # Force fp32 master weights. transformers>=5 loads a checkpoint in its native
    # dtype, and several DeBERTa checkpoints are fp16; the regression head then
    # does logits.to(labels.dtype), so backward pushes a Float grad into a Half
    # tensor and dies with "Found dtype Float but expected Half". fp32 params
    # with bf16 autocast is also the correct AMP setup on GPU.
    model = factory.from_pretrained(cfg.base_model, **kwargs).to(
        device=device, dtype=torch.float32
    )

    make = _TokenDataset if is_token else _TextDataset
    collate = _make_collate(tok, cfg.task)
    train_loader = DataLoader(
        make(train_ds, tok, cfg, label2id),
        batch_size=cfg.batch_size, shuffle=True, collate_fn=collate,
    )
    eval_loader = (
        DataLoader(
            make(eval_ds, tok, cfg, label2id),
            batch_size=cfg.eval_batch_size, shuffle=False, collate_fn=collate,
        )
        if eval_ds is not None
        else None
    )

    total_steps = max(1, math.ceil(len(train_loader) * cfg.epochs))
    warmup = int(total_steps * cfg.warmup_ratio)
    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay
    )

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / max(1, warmup)
        span = max(1, total_steps - warmup)
        return max(0.0, (total_steps - step) / span)

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    # bf16 on Ada needs no GradScaler, unlike fp16.
    use_amp = device.type == "cuda"

    log(f"total_steps={total_steps} warmup={warmup}")
    model.train()
    step = 0
    running = 0.0
    done = False
    for epoch in range(math.ceil(cfg.epochs)):
        if done:
            break
        for batch in train_loader:
            if should_cancel():
                raise Cancelled()
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            optim.zero_grad(set_to_none=True)

            running += loss.item()
            step += 1
            if step % 10 == 0 or step == total_steps:
                log(f"step {step}/{total_steps} loss={running / min(step, 10):.4f}")
                running = 0.0
            on_progress(step / total_steps)
            if step >= total_steps:
                done = True
                break
        log(f"epoch {epoch + 1} complete")

    metrics: dict[str, Any] = {"train_steps": step}
    if eval_loader is not None:
        model.eval()
        all_logits, all_gold = [], []
        with torch.no_grad():
            for batch in eval_loader:
                gold = batch.pop("labels")
                batch = {k: v.to(device) for k, v in batch.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    logits = model(**batch).logits
                all_logits.append(logits.float().cpu().numpy())
                all_gold.append(gold.numpy())

        if is_token:
            width = max(a.shape[1] for a in all_logits)
            logits = np.concatenate(
                [np.pad(a, ((0, 0), (0, width - a.shape[1]), (0, 0))) for a in all_logits]
            )
            gold = np.concatenate(
                [
                    np.pad(g, ((0, 0), (0, width - g.shape[1])), constant_values=IGNORE_INDEX)
                    for g in all_gold
                ]
            )
        else:
            logits = np.concatenate(all_logits)
            gold = np.concatenate(all_gold)

        metrics.update(_metrics(cfg.task, logits, gold, labels, cfg.threshold))
        log(f"eval metrics: {json.dumps(metrics)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    (out_dir / "training_metrics.json").write_text(json.dumps(metrics, indent=2))
    log(f"saved model to {out_dir}")
    return metrics
