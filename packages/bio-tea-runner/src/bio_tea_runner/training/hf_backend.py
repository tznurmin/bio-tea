# Copyright 2026 tznurmin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from bio_tea.datasets.io import read_jsonl


def _flatten_token_predictions(
    predictions: Any,
    labels: Any,
) -> tuple[list[int], list[int]]:
    pred_rows = predictions.tolist() if hasattr(predictions, "tolist") else predictions
    label_rows = labels.tolist() if hasattr(labels, "tolist") else labels

    y_true: list[int] = []
    y_pred: list[int] = []
    for prow, lrow in zip(pred_rows, label_rows):
        for pvec, lid in zip(prow, lrow):
            li = int(lid)
            if li == -100:
                continue
            if isinstance(pvec, (list, tuple)):
                if not pvec:
                    continue
                pi = int(max(range(len(pvec)), key=lambda i: float(pvec[i])))
            else:
                pi = int(pvec)
            y_true.append(li)
            y_pred.append(pi)
    return y_true, y_pred


def _prf_excluding_o(*, y_true: Sequence[int], y_pred: Sequence[int], id2label: Mapping[int, str]) -> tuple[float, float, float]:
    tp = 0
    fp = 0
    fn = 0
    for t, p in zip(y_true, y_pred):
        tl = str(id2label.get(int(t), "O"))
        pl = str(id2label.get(int(p), "O"))
        t_pos = tl != "O"
        p_pos = pl != "O"
        if p_pos and t_pos:
            tp += 1
        elif p_pos and not t_pos:
            fp += 1
        elif (not p_pos) and t_pos:
            fn += 1
    precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(f1)


def _coerce_loss(metrics: Mapping[str, Any]) -> float:
    for k in ["test_loss", "eval_loss", "loss"]:
        if k in metrics:
            try:
                return float(metrics[k])
            except Exception:
                pass
    return float("nan")


def _collate_tokenized_batch(features: list[dict]):
    """Pad pre-tokenized features to the longest sequence in batch.

    Prepared datasets already provide `input_ids`, `attention_mask`, and `labels`.
    Padding is model-agnostic because prepared datasets already provide tokenized features.
    """

    try:
        import torch
    except Exception as e:  # pragma: no cover
        raise RuntimeError("torch is required for HF backend collation") from e

    if not features:
        raise ValueError("Cannot collate empty feature batch")

    max_len = max(len(list(f.get("input_ids") or [])) for f in features)
    if max_len <= 0:
        raise ValueError("Invalid empty input_ids in feature batch")

    def _pad(key: str, pad_value: int) -> Any:
        rows: list[list[int]] = []
        for f in features:
            vals = [int(x) for x in (f.get(key) or [])]
            if len(vals) > max_len:
                vals = vals[:max_len]
            if len(vals) < max_len:
                vals = vals + [int(pad_value)] * (max_len - len(vals))
            rows.append(vals)
        return torch.tensor(rows)

    return {
        "input_ids": _pad("input_ids", 0),
        "attention_mask": _pad("attention_mask", 0),
        "labels": _pad("labels", -100),
    }


def _predict_metrics_for_eval(*, trainer: Any, eval_ds: Any, id2label: Mapping[int, str]) -> dict[str, float]:
    pred = trainer.predict(eval_ds)
    y_true, y_pred = _flatten_token_predictions(pred.predictions, pred.label_ids)
    precision, recall, f1 = _prf_excluding_o(y_true=y_true, y_pred=y_pred, id2label=id2label)
    loss = _coerce_loss(pred.metrics or {})
    if math.isnan(loss):
        loss = 0.0
    return {
        "loss": float(loss),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def run_hf_backend(
    *,
    prepared_dir: Path,
    prepared_manifest: Mapping[str, Any],
    job: Mapping[str, Any],
    cfg: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Train/evaluate one Exp1 job using Hugging Face Trainer."""

    try:
        from datasets import Dataset
        from transformers import AutoModelForTokenClassification, Trainer, TrainerCallback, TrainingArguments
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Hugging Face training backend requires datasets + transformers + torch runtime dependencies"
        ) from e

    prepared_dir = Path(prepared_dir)
    train_rows = read_jsonl(prepared_dir / "train.tokenized.jsonl")
    eval_sets = list(prepared_manifest.get("eval_sets") or [])
    eval_rows = {
        name: read_jsonl(prepared_dir / f"eval.{name}.tokenized.jsonl")
        for name in eval_sets
    }

    # Stored as {"0": "O", ...} in JSON.
    raw_id2label = dict(prepared_manifest.get("id2label") or {})
    id2label = {int(k): str(v) for k, v in raw_id2label.items()}
    label2id = {str(k): int(v) for k, v in dict(prepared_manifest.get("label2id") or {}).items()}

    ds_train = Dataset.from_list(train_rows)
    ds_eval = {name: Dataset.from_list(rows) for name, rows in eval_rows.items()}

    exp_id = str(job.get("experiment_id") or "exp1")
    global_hparams = (((cfg.get("training") or {}).get(exp_id) or {}).get("hparams") or {})
    hparams = dict(global_hparams or {})
    hparams.update(dict(job.get("hparams") or {}))
    hf_cfg = ((cfg.get("training") or {}).get("hf") or {})
    report_each_epoch = bool(hparams.get("report_each_epoch", False))
    dataloader_pin_memory = bool(hf_cfg.get("dataloader_pin_memory", False))
    model_local_files_only = bool(job.get("model_local_files_only", hf_cfg.get("local_files_only", False)))
    model_revision_raw = job.get("model_revision", hf_cfg.get("revision"))
    model_revision = str(model_revision_raw).strip() if model_revision_raw is not None else ""
    model_revision = model_revision or None
    output_dir = str(prepared_dir / "_trainer")
    train_args = TrainingArguments(
        output_dir=output_dir,
        do_train=True,
        do_eval=False,
        num_train_epochs=float(hparams.get("epochs", 1)),
        learning_rate=float(hparams.get("learning_rate", 5e-5)),
        per_device_train_batch_size=int(hparams.get("per_device_train_batch_size", 8)),
        per_device_eval_batch_size=int(hparams.get("per_device_eval_batch_size", 8)),
        seed=int(job.get("seed", 0)),
        data_seed=int(job.get("seed", 0)),
        logging_strategy="no",
        save_strategy="no",
        report_to=[],
        dataloader_pin_memory=dataloader_pin_memory,
    )

    model_load_kwargs: dict[str, Any] = {
        "num_labels": len(label2id),
        "id2label": id2label,
        "label2id": label2id,
    }
    if model_local_files_only:
        model_load_kwargs["local_files_only"] = True
    if model_revision is not None:
        model_load_kwargs["revision"] = model_revision
    model = AutoModelForTokenClassification.from_pretrained(
        str(job["model_name_or_path"]),
        **model_load_kwargs,
    )
    epoch_history: dict[str, list[dict]] = {name: [] for name in eval_sets}

    class _EpochEvalCallback(TrainerCallback):
        def __init__(self):
            self._trainer = None

        def bind_trainer(self, trainer_obj):
            self._trainer = trainer_obj

        def on_epoch_end(self, args, state, control, **kwargs):
            if not report_each_epoch:
                return
            if self._trainer is None:
                return
            ep = float(state.epoch or 0.0)
            for name, eval_ds in ds_eval.items():
                m = _predict_metrics_for_eval(trainer=self._trainer, eval_ds=eval_ds, id2label=id2label)
                epoch_history[name].append(
                    {
                        "epoch": float(ep),
                        **m,
                    }
                )

    cb = _EpochEvalCallback()

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds_train,
        data_collator=_collate_tokenized_batch,
        callbacks=[cb],
    )
    cb.bind_trainer(trainer)
    trainer.train()

    save_model = bool(job.get("save_model", False))
    save_model_dir_raw = str(job.get("save_model_dir") or "").strip()
    if save_model and save_model_dir_raw:
        save_model_dir = Path(save_model_dir_raw)
        save_model_dir.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(save_model_dir))

    out: dict[str, dict[str, Any]] = {}
    for eval_name, eval_ds in ds_eval.items():
        out[eval_name] = _predict_metrics_for_eval(trainer=trainer, eval_ds=eval_ds, id2label=id2label)
        if report_each_epoch:
            out[eval_name]["per_epoch"] = list(epoch_history.get(eval_name) or [])
    return out
