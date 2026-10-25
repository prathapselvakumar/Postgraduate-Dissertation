"""Plateau-based stopping rule for RL training runs.

A run has converged when the moving average of the per-iteration training return has not
improved by more than `min_delta` for `patience` iterations, after at least `min_iterations`.
The return history is persisted next to the checkpoints, so the rule carries over when a run
is resumed in a new Slurm job, and a `CONVERGED` marker is written when it fires so the job
scripts stop resubmitting.

The rule is identical for every credit-assignment method, so iterations-to-convergence can be
compared between arms. It reads the training batch's return, which is noisy (48 rollouts per
iteration); the moving-average window exists to smooth that.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path

MARKER_NAME = "CONVERGED"
HISTORY_NAME = "convergence.json"


@dataclass
class ConvergenceTracker:
    window: int = 20
    patience: int = 40
    min_delta: float = 0.01
    min_iterations: int = 100
    returns: list[float] = field(default_factory=list)
    best_moving_average: float = float("-inf")
    best_iteration: int = 0

    def update(self, avg_return: float) -> bool:
        """Record one iteration's return; True once the run has converged."""
        self.returns.append(float(avg_return))
        iteration = len(self.returns)
        if iteration >= self.window:
            moving_average = sum(self.returns[-self.window :]) / self.window
            if moving_average > self.best_moving_average + self.min_delta:
                self.best_moving_average = moving_average
                self.best_iteration = iteration
        return (
            iteration >= self.min_iterations
            and iteration >= self.window
            and iteration - self.best_iteration >= self.patience
        )

    def summary(self) -> dict[str, float]:
        latest = (
            sum(self.returns[-self.window :]) / self.window
            if len(self.returns) >= self.window
            else float("nan")
        )
        return {
            "convergence/moving_avg_return": latest,
            "convergence/best_moving_avg_return": self.best_moving_average,
            "convergence/iterations_since_best": float(len(self.returns) - self.best_iteration),
        }

    # Persistence: plain files in the (local, shared-filesystem) checkpoint directory.
    @staticmethod
    def _dir(cloud_path: str) -> Path | None:
        return None if "://" in cloud_path else Path(cloud_path)

    def save(self, cloud_path: str) -> None:
        directory = self._dir(cloud_path)
        if directory is None:
            return
        directory.mkdir(parents=True, exist_ok=True)
        state = dict(returns=self.returns, best_moving_average=self.best_moving_average,
                     best_iteration=self.best_iteration)
        tmp = directory / (HISTORY_NAME + ".tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(directory / HISTORY_NAME)

    def load(self, cloud_path: str, iterations_completed: int) -> None:
        """Restore history, truncated to the checkpoint actually resumed from."""
        directory = self._dir(cloud_path)
        if directory is None or not (directory / HISTORY_NAME).exists():
            return
        state = json.loads((directory / HISTORY_NAME).read_text())
        self.returns = list(state["returns"])[:iterations_completed]
        # Recompute the best point from the truncated history rather than trusting the file.
        self.best_moving_average, self.best_iteration = float("-inf"), 0
        history, self.returns = self.returns, []
        for value in history:
            self.update(value)

    def mark_converged(self, cloud_path: str) -> None:
        directory = self._dir(cloud_path)
        if directory is None:
            return
        (directory / MARKER_NAME).write_text(
            json.dumps(dict(iteration=len(self.returns),
                            best_iteration=self.best_iteration,
                            best_moving_average=self.best_moving_average))
        )
