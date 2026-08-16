"""Backward-compatible early stopping for the downloaded official trainer."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import torch


class _StopTraining(RuntimeError):
    pass


@dataclass
class EarlyStoppingResult:
    stopped_early: bool
    epochs_completed: int
    best_epoch: int
    best_validation_loss: float
    history: list[dict[str, float]] = field(default_factory=list)


def train_with_early_stopping(
    train_module: Any,
    *,
    checkpoint: Path,
    min_epochs: int,
    patience: int,
    min_delta: float,
    train_kwargs: dict[str, Any],
) -> EarlyStoppingResult:
    """Call the official trainer and stop before a clearly unproductive epoch.

    The official module is patched only for the duration of this call.  Its
    public API and files remain untouched, so the legacy runners keep working.
    The stopping monitor is validation loss; model-method ranking remains the
    exact Kaggle competition score after full held-out graph inference.
    """
    if min_epochs < 1 or patience < 1 or min_delta < 0:
        raise ValueError("invalid early-stopping settings")

    original_evaluate: Callable = train_module.evaluate
    original_train_epoch: Callable = train_module.train_epoch
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_path = checkpoint.with_name("edge_predictor_earlystop_best.pth")
    state: dict[str, Any] = {
        "history": [], "best_loss": float("inf"), "best_epoch": -1,
        "bad_epochs": 0, "stop_requested": False,
    }

    def evaluate(model: torch.nn.Module, *args: Any, **kwargs: Any):
        values = original_evaluate(model, *args, **kwargs)
        validation_loss, accuracy, node_recall = map(float, values)
        epoch = len(state["history"])
        improved = validation_loss < state["best_loss"] - min_delta
        if improved:
            state["best_loss"] = validation_loss
            state["best_epoch"] = epoch
            state["bad_epochs"] = 0
            weights = {
                key.replace("unet.module.", "unet.", 1): value.detach().cpu()
                for key, value in model.state_dict().items()
            }
            torch.save(weights, best_path)
        else:
            state["bad_epochs"] += 1
        state["history"].append({
            "epoch": epoch,
            "validation_loss": validation_loss,
            "edge_accuracy_proxy": accuracy,
            "node_recall_proxy": node_recall,
            "improved": bool(improved),
        })
        state["stop_requested"] = (
            len(state["history"]) >= min_epochs and state["bad_epochs"] >= patience
        )
        return values

    def train_epoch(*args: Any, **kwargs: Any):
        if state["stop_requested"]:
            raise _StopTraining("early stopping patience exhausted")
        return original_train_epoch(*args, **kwargs)

    stopped = False
    train_module.evaluate = evaluate
    train_module.train_epoch = train_epoch
    try:
        train_module.train(**train_kwargs)
    except _StopTraining:
        stopped = True
        print(
            "Early stopping: validation loss did not improve for "
            f"{patience} epochs; best epoch={state['best_epoch'] + 1}.",
            flush=True,
        )
    finally:
        train_module.evaluate = original_evaluate
        train_module.train_epoch = original_train_epoch

    if not best_path.is_file():
        raise FileNotFoundError(f"early-stopping checkpoint was not created: {best_path}")
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best_path, checkpoint)
    return EarlyStoppingResult(
        stopped_early=stopped,
        epochs_completed=len(state["history"]),
        best_epoch=int(state["best_epoch"]),
        best_validation_loss=float(state["best_loss"]),
        history=list(state["history"]),
    )
