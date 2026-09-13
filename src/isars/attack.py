"""Projected-gradient attacks for video-text retrieval.

The module is intentionally model-agnostic. A model wrapper only needs a
``device`` attribute and a differentiable ``get_video_embedding`` method.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch


class VideoEmbeddingModel(Protocol):
    """Minimal interface required by the attacks."""

    device: torch.device | str

    def get_video_embedding(self, video: torch.Tensor) -> torch.Tensor:
        """Return one embedding per video while preserving gradients."""


@dataclass(frozen=True)
class AttackConfig:
    """Configuration shared by framewise and temporally regularized PGD."""

    epsilon: float = 8 / 255
    alpha: float = 2 / 255
    steps: int = 15
    lambda1: float = 0.5
    lambda2: float = 0.1
    random_start: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        if not 0 < self.epsilon <= 1:
            raise ValueError("epsilon must be in (0, 1]")
        if not 0 < self.alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        if self.steps < 1:
            raise ValueError("steps must be at least 1")
        if self.lambda1 < 0 or self.lambda2 < 0:
            raise ValueError("temporal weights must be non-negative")


class SpatioTemporalPGDAttack:
    """Untargeted query-conditioned PGD with multi-order smoothness.

    The optimization minimizes cosine similarity between the attacked video and
    its ground-truth query while penalizing first- and second-order changes in
    the perturbation. Temporal losses use a full tensor mean, so their scale is
    invariant to frame resolution.
    """

    def __init__(self, model: VideoEmbeddingModel, config: AttackConfig | None = None):
        self.model = model
        self.config = config or AttackConfig()

    @staticmethod
    def temporal_losses(delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return first- and second-order temporal mean-squared differences."""
        if delta.ndim != 5:
            raise ValueError("delta must have shape (B, T, C, H, W)")

        zero = delta.new_zeros(())
        first = zero
        second = zero
        if delta.shape[1] > 1:
            first = (delta[:, 1:] - delta[:, :-1]).square().mean()
        if delta.shape[1] > 2:
            curvature = delta[:, 2:] - 2 * delta[:, 1:-1] + delta[:, :-2]
            second = curvature.square().mean()
        return first, second

    def _initial_delta(self, clean: torch.Tensor) -> torch.Tensor:
        if not self.config.random_start:
            return torch.zeros_like(clean)

        generator = None
        if self.config.seed is not None:
            generator = torch.Generator(device=clean.device)
            generator.manual_seed(self.config.seed)

        delta = torch.empty_like(clean).uniform_(
            -self.config.epsilon,
            self.config.epsilon,
            generator=generator,
        )
        return torch.clamp(clean + delta, 0, 1) - clean

    def attack(
        self,
        video: torch.Tensor,
        target_text_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """Return an adversarial video with the same shape as ``video``.

        Args:
            video: Float tensor shaped ``(B, T, C, H, W)`` in ``[0, 1]``.
            target_text_embedding: One target text embedding per batch item.
        """
        if video.ndim != 5:
            raise ValueError("video must have shape (B, T, C, H, W)")
        if not video.is_floating_point():
            raise TypeError("video must be a floating-point tensor")
        if video.numel() == 0:
            raise ValueError("video must not be empty")
        if video.detach().amin().item() < 0 or video.detach().amax().item() > 1:
            raise ValueError("video values must be in [0, 1]")

        device = torch.device(self.model.device)
        clean = video.detach().clone().to(device)
        target = target_text_embedding.detach().clone().to(device)
        if target.ndim != 2 or target.shape[0] != clean.shape[0]:
            raise ValueError("target_text_embedding must have shape (B, D)")

        delta = self._initial_delta(clean)
        for _ in range(self.config.steps):
            delta = delta.detach().requires_grad_(True)
            adversarial = torch.clamp(clean + delta, 0, 1)
            video_embedding = self.model.get_video_embedding(adversarial)
            if video_embedding.shape != target.shape:
                raise ValueError("video and text embedding shapes must match")

            cosine_similarity = torch.nn.functional.cosine_similarity(
                video_embedding,
                target,
                dim=-1,
            ).mean()
            first, second = self.temporal_losses(delta)
            loss = cosine_similarity + self.config.lambda1 * first + self.config.lambda2 * second
            gradient = torch.autograd.grad(loss, delta, only_inputs=True)[0]

            delta = delta - self.config.alpha * gradient.sign()
            delta = delta.clamp(-self.config.epsilon, self.config.epsilon)
            delta = torch.clamp(clean + delta, 0, 1) - clean

        result = torch.clamp(clean + delta.detach(), 0, 1)
        observed_linf = (result - clean).abs().flatten(1).amax(dim=1)
        if torch.any(observed_linf > self.config.epsilon + 1e-6):
            raise RuntimeError("generated perturbation violates the L-infinity bound")
        return result


class FramewisePGDAttack(SpatioTemporalPGDAttack):
    """White-box framewise PGD baseline with no temporal regularization."""

    def __init__(self, model: VideoEmbeddingModel, config: AttackConfig | None = None):
        base = config or AttackConfig()
        framewise = AttackConfig(
            epsilon=base.epsilon,
            alpha=base.alpha,
            steps=base.steps,
            lambda1=0,
            lambda2=0,
            random_start=base.random_start,
            seed=base.seed,
        )
        super().__init__(model=model, config=framewise)
