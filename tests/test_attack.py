import pytest

torch = pytest.importorskip("torch")

from isars.attack import AttackConfig, SpatioTemporalPGDAttack  # noqa: E402


class MeanEmbeddingModel:
    device = "cpu"

    def get_video_embedding(self, video: torch.Tensor) -> torch.Tensor:
        return video.mean(dim=(1, 3, 4))


def test_attack_respects_pixel_and_linf_bounds() -> None:
    clean = torch.full((1, 4, 3, 2, 2), 0.5)
    target = torch.tensor([[1.0, 0.0, 0.0]])
    config = AttackConfig(
        epsilon=0.1,
        alpha=0.05,
        steps=2,
        random_start=False,
    )
    result = SpatioTemporalPGDAttack(MeanEmbeddingModel(), config).attack(clean, target)
    assert result.shape == clean.shape
    assert float(result.min()) >= 0.0
    assert float(result.max()) <= 1.0
    assert float((result - clean).abs().max()) <= config.epsilon + 1e-6


def test_temporal_losses_are_zero_for_constant_perturbation() -> None:
    delta = torch.ones((1, 4, 3, 2, 2))
    first, second = SpatioTemporalPGDAttack.temporal_losses(delta)
    assert first.item() == 0.0
    assert second.item() == 0.0
