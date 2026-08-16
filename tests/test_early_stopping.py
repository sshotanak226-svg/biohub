from pathlib import Path

import torch

from biohub_demo.early_stopping import train_with_early_stopping


class _FakeTrainer:
    def __init__(self) -> None:
        self.losses = iter([1.0, 0.8, 0.81, 0.82, 0.83])

    def train_epoch(self, *args, **kwargs):
        return 0.0, 0.0

    def evaluate(self, *args, **kwargs):
        return next(self.losses), 0.9, 0.8

    def train(self, n_epochs: int) -> None:
        model = torch.nn.Linear(1, 1)
        for _ in range(n_epochs):
            self.train_epoch(model, None, None, None)
            self.evaluate(model, None, None)


def test_early_stopping_restores_module_and_copies_best_checkpoint(tmp_path: Path) -> None:
    trainer = _FakeTrainer()
    original_train_epoch = trainer.train_epoch
    original_evaluate = trainer.evaluate
    checkpoint = tmp_path / "weights" / "edge_predictor_best.pth"
    result = train_with_early_stopping(
        trainer,
        checkpoint=checkpoint,
        min_epochs=2,
        patience=2,
        min_delta=0.0,
        train_kwargs={"n_epochs": 10},
    )
    assert result.stopped_early is True
    assert result.epochs_completed == 4
    assert result.best_epoch == 1
    assert result.best_validation_loss == 0.8
    assert checkpoint.is_file()
    assert trainer.train_epoch == original_train_epoch
    assert trainer.evaluate == original_evaluate


def test_early_stopping_can_limit_monitoring_batches(tmp_path: Path) -> None:
    class LoaderAwareTrainer(_FakeTrainer):
        def __init__(self) -> None:
            super().__init__()
            self.seen_validation_batches: list[int] = []

        def evaluate(self, model, loader, *args, **kwargs):
            self.seen_validation_batches.append(len(list(loader)))
            return next(self.losses), 0.9, 0.8

        def train(self, n_epochs: int) -> None:
            model = torch.nn.Linear(1, 1)
            for _ in range(n_epochs):
                self.train_epoch(model, None, None, None)
                self.evaluate(model, range(10), None)

    trainer = LoaderAwareTrainer()
    train_with_early_stopping(
        trainer,
        checkpoint=tmp_path / "weights" / "edge_predictor_best.pth",
        min_epochs=1,
        patience=1,
        min_delta=0.0,
        train_kwargs={"n_epochs": 1},
        max_validation_iters=3,
    )
    assert trainer.seen_validation_batches == [3]
