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


CHECKPOINT_FILE = "checkpoint.pt"


# Tensors that may legitimately be rebuilt when adapting a checkpoint to a new
# label set. Anything else mismatching means the base is not compatible -- a
# different hidden size or architecture -- and silently randomising it would
# train a subtly broken model rather than fail.
HEAD_PREFIXES = ("classifier", "pooler", "score")


def _check_load_report(info: dict | None, base_model: str, log) -> None:
    """Log what `from_pretrained` rebuilt, and refuse anything beyond the head.

    `reinit_head` exists to rebuild a classification head, not to paper over a
    wrong base model. Scoping the relaxation here keeps the loud failure for
    every case it was protecting.
    """
    if not info:
        return

    missing = sorted(info.get("missing_keys") or ())
    mismatched = sorted(k[0] for k in (info.get("mismatched_keys") or ()))

    if missing:
        log(f"newly initialised (absent from the checkpoint): {missing}")
    if not mismatched:
        return
    log(f"reinitialised due to shape mismatch: {mismatched}")

    beyond_head = [k for k in mismatched if not k.startswith(HEAD_PREFIXES)]
    if beyond_head:
        raise ValueError(
            "reinit_head permits rebuilding the classification head only, but "
            f"these tensors also mismatched: {beyond_head}. That usually means "
            f"{base_model} is not a compatible base for this task."
        )


class Cancelled(Exception):
    """Raised when a cancel request is observed mid-training."""


class Interrupted(Exception):
    """Raised when the service is shutting down and the job should resume later."""


class Preempted(Exception):
    """Raised at an epoch boundary when higher-priority work is waiting."""


def _save_checkpoint(ckpt_dir: Path, model, optim, sched, epochs_completed: int,
                     step: int, labels: list[str], log) -> None:
    """Write a resumable checkpoint, replacing any previous one atomically."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tmp = ckpt_dir / (CHECKPOINT_FILE + ".tmp")
    final = ckpt_dir / CHECKPOINT_FILE
    torch.save(
        {
            "epochs_completed": epochs_completed,
            "step": step,
            "labels": labels,
            "model": model.state_dict(),
            "optimizer": optim.state_dict(),
            "scheduler": sched.state_dict(),
            "torch_rng": torch.get_rng_state(),
        },
        tmp,
    )
    # Rename is atomic within a filesystem, so a crash mid-write can never
    # leave a truncated file where a good checkpoint used to be.
    tmp.replace(final)
    log(f"checkpoint saved at epoch {epochs_completed} "
        f"({final.stat().st_size // (1 << 20)} MiB)")


def _load_checkpoint(ckpt_dir: Path, model, optim, sched, labels: list[str], log):
    """Restore state written by _save_checkpoint. Returns None if unusable."""
    path = ckpt_dir / CHECKPOINT_FILE
    if not path.exists():
        return None
    try:
        # map_location='cpu' keeps the RNG state loadable; load_state_dict then
        # copies into the already-placed model and optimizer tensors.
        # weights_only=False is safe here: this file is written by the service
        # into its own state directory, never supplied by a caller.
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        log(f"checkpoint unreadable ({exc}); starting from scratch")
        return None

    if ckpt.get("labels") != labels:
        log("checkpoint label set differs from the current data; ignoring it")
        return None

    try:
        model.load_state_dict(ckpt["model"])
        optim.load_state_dict(ckpt["optimizer"])
        sched.load_state_dict(ckpt["scheduler"])
        if ckpt.get("torch_rng") is not None:
            torch.set_rng_state(ckpt["torch_rng"].to(torch.uint8).cpu())
    except Exception as exc:
        log(f"checkpoint incompatible ({exc}); starting from scratch")
        return None
    return ckpt


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
    ckpt_dir: Path,
    log: Callable[[str], None],
    on_progress: Callable[[float], None],
    on_checkpoint: Callable[[int], None],
    should_cancel: Callable[[], bool],
    should_stop: Callable[[], bool],
    should_yield: Callable[[], bool],
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

    if cfg.reinit_head:
        # Transfer learning onto a different label set: the encoder loads
        # normally and only shape-mismatched tensors are rebuilt. Scoped by
        # _check_load_report below, so this cannot quietly randomise the encoder.
        kwargs["ignore_mismatched_sizes"] = True

    try:
        model, load_info = factory.from_pretrained(
            cfg.base_model, output_loading_info=True, **kwargs
        )
    except (RuntimeError, ValueError) as exc:
        if "mismatch" in str(exc).lower():
            raise RuntimeError(
                f"{cfg.base_model} carries a classification head that does not "
                f"match num_labels={num_labels}. If you mean to fine-tune an "
                'existing task checkpoint onto a new label set, set '
                '"reinit_head": true to rebuild just the head.'
            ) from exc
        raise

    _check_load_report(load_info, cfg.base_model, log)

    # Force fp32 master weights. transformers>=5 loads a checkpoint in its native
    # dtype, and several DeBERTa checkpoints are fp16; the regression head then
    # does logits.to(labels.dtype), so backward pushes a Float grad into a Half
    # tensor and dies with "Found dtype Float but expected Half". fp32 params
    # with bf16 autocast is also the correct AMP setup on GPU.
    model = model.to(device=device, dtype=torch.float32)

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

    # Resume if a checkpoint from an earlier attempt is present. Granularity is
    # a whole epoch: the interrupted epoch is redone from its start, which
    # avoids replaying the dataloader to an exact batch.
    start_epoch, step = 0, 0
    resumed = _load_checkpoint(ckpt_dir, model, optim, sched, labels, log)
    if resumed is not None:
        start_epoch = int(resumed["epochs_completed"])
        step = int(resumed["step"])
        log(f"resumed from checkpoint: {start_epoch} epochs done, step {step}/{total_steps}")
        on_progress(min(1.0, step / total_steps))

    model.train()
    running = 0.0
    done = False
    for epoch in range(start_epoch, math.ceil(cfg.epochs)):
        if done:
            break
        for batch in train_loader:
            if should_cancel():
                raise Cancelled()
            if should_stop():
                # Service shutting down. Unwind now; the last checkpoint is
                # already durable and the job will be re-queued.
                raise Interrupted()
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

        epochs_done = epoch + 1
        # The epoch boundary is the only preemption point: it is where a
        # checkpoint is coherent. A job with checkpointing disabled cannot be
        # preempted, because there would be nothing to resume from.
        can_checkpoint = bool(cfg.checkpoint_every_epochs) and not done
        due = can_checkpoint and epochs_done % cfg.checkpoint_every_epochs == 0
        yielding = can_checkpoint and should_yield()

        # When `done` is set the step budget ran out mid-epoch and training is
        # over anyway, so a checkpoint would be written only to be deleted.
        # Yielding forces a write even off-interval, so no work is lost.
        if due or yielding:
            _save_checkpoint(ckpt_dir, model, optim, sched, epochs_done, step, labels, log)
            on_checkpoint(epochs_done)

        if yielding:
            log(f"higher-priority work is waiting; yielding after epoch {epochs_done}")
            raise Preempted()

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
