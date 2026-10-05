"""Multidimensional Brownian and fractional Brownian motion paths."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np
from fbm import fgn
from matplotlib.animation import FuncAnimation, PillowWriter

log = logging.getLogger(__name__)


class Path:
    """A sampled trajectory of shape (n_samples, dimension)."""

    def __init__(self, x: np.ndarray, dimension: int, increment: float) -> None:
        self.x = np.asarray(x, dtype=float)
        self.dimension = dimension
        self.increment = increment

    @property
    def time(self) -> np.ndarray:
        """Time stamps of the samples."""
        return np.arange(len(self.x)) * self.increment

    def mean(self) -> np.ndarray:
        """Mean of each coordinate."""
        return self.x.mean(axis=0)

    def std(self) -> np.ndarray:
        """Standard deviation of each coordinate."""
        return self.x.std(axis=0)

    def project_on_axis(self, axis: int) -> Path:
        """Return the one-dimensional path made of a single coordinate."""
        return Path(self.x[:, [axis]], 1, self.increment)

    def plot(self, ax: plt.Axes | None = None) -> plt.Axes:
        """Plot the path: x(t) in 1D, the (x, y) trajectory otherwise."""
        if ax is None:
            _, ax = plt.subplots(figsize=(6, 5))
        if self.dimension == 1:
            ax.plot(self.time, self.x[:, 0])
            ax.set_xlabel("$t$")
            ax.set_ylabel("$x(t)$")
        else:
            ax.plot(self.x[:, 0], self.x[:, 1])
            ax.set_xlabel("$x$")
            ax.set_ylabel("$y$")
            ax.set_aspect("equal", adjustable="datalim")
        ax.grid(True, alpha=0.3)
        return ax

    def plot_animation(
        self, interval: int = 20, step: int = 1, save_path: str | None = None
    ) -> FuncAnimation:
        """Animate the first two coordinates of the path.

        Parameters
        ----------
        interval:
            Delay between frames, in milliseconds.
        step:
            Number of samples revealed per frame.
        save_path:
            If given, the animation is written to this GIF file.
        """
        if self.dimension < 2:
            raise ValueError("Animation requires a path of dimension >= 2.")
        fig, ax = plt.subplots(figsize=(9, 5.2))
        ax.grid(True, alpha=0.3)
        # Same plot area as scripts/make_gif.py, so that the two animations line up.
        fig.subplots_adjust(left=0.08, right=0.84, bottom=0.12, top=0.96)
        # Equal scale on both axes, with limits widened to fill the wide frame.
        box = ax.get_position()
        ratio = (box.width * fig.get_figwidth()) / (box.height * fig.get_figheight())
        lo, hi = self.x[:, :2].min(axis=0), self.x[:, :2].max(axis=0)
        center, span = (lo + hi) / 2, 1.1 * (hi - lo)
        half_y = max(span[1], span[0] / ratio) / 2
        half_x = half_y * ratio
        ax.set_xlim(center[0] - half_x, center[0] + half_x)
        ax.set_ylim(center[1] - half_y, center[1] + half_y)
        ax.set(xlabel="$x$", ylabel="$y$")
        (line,) = ax.plot([], [], lw=1.4, label="$H = 1/2$")
        (head,) = ax.plot([], [], "o", ms=6)
        ax.legend(handles=[line], loc="upper left", bbox_to_anchor=(1.01, 1.0), frameon=False)
        frames = range(1, len(self.x), step)

        def update(i: int):
            line.set_data(self.x[:i, 0], self.x[:i, 1])
            head.set_data([self.x[i - 1, 0]], [self.x[i - 1, 1]])
            return line, head

        anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=True)
        if save_path is not None:
            anim.save(save_path, writer=PillowWriter(fps=max(1, 1000 // interval)), dpi=130)
            log.info("Saved animation to %s", save_path)
        return anim


class StochasticProcess(ABC):
    """Base class of the continuous-time processes sampled on a regular grid."""

    def __init__(self, increment: float, dimension: int) -> None:
        self.increment = increment
        self.dimension = dimension

    @abstractmethod
    def generate(self, samples_number: int) -> Path:
        """Generate a path with the given number of samples."""


class FractionalBrownianMotion(StochasticProcess):
    """Multidimensional fBm with independent coordinates.

    The increments of each coordinate are fractional Gaussian noise with
    standard deviation ``delta * increment**hurst`` (exact Davies-Harte sampling).
    For ``hurst = 0.5`` this is the standard Brownian motion.
    """

    def __init__(
        self,
        increment: float = 0.01,
        delta: float = 0.1,
        hurst: float = 0.5,
        dim: int = 2,
        initial_condition: np.ndarray | None = None,
    ) -> None:
        super().__init__(increment, dim)
        if not 0 < hurst < 1:
            raise ValueError("The Hurst parameter must lie in (0, 1).")
        self.hurst = hurst
        self.delta = delta
        self.initial_condition = (
            np.zeros(dim) if initial_condition is None else np.asarray(initial_condition, float)
        )

    def _increments(self, n: int) -> np.ndarray:
        """Return n increments per coordinate, shape (n, dimension)."""
        if self.hurst == 0.5:
            scale = self.delta * np.sqrt(self.increment)
            return np.random.normal(0.0, scale, size=(n, self.dimension))
        length = n * self.increment
        cols = [fgn(n, self.hurst, length, "daviesharte") for _ in range(self.dimension)]
        return self.delta * np.stack(cols, axis=1)

    def generate(self, samples_number: int) -> Path:
        """Generate a path starting at the initial condition."""
        steps = self._increments(samples_number - 1)
        x = np.vstack([np.zeros((1, self.dimension)), np.cumsum(steps, axis=0)])
        log.debug("Generated fBm path: H=%.2f, n=%d", self.hurst, samples_number)
        return Path(self.initial_condition + x, self.dimension, self.increment)


class BrownianMotion(FractionalBrownianMotion):
    """Standard multidimensional Brownian motion (Hurst parameter 1/2)."""

    def __init__(
        self,
        increment: float = 0.01,
        delta: float = 0.1,
        dim: int = 2,
        initial_condition: np.ndarray | None = None,
    ) -> None:
        super().__init__(increment, delta, 0.5, dim, initial_condition)
