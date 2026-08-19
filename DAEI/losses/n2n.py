"""Noise2Noise loss for DAE training.

Given two independent noisy observations of the same clean embedding,
trains the DAE to denoise one observation to match the other.
"""

from typing import Tuple

import torch
import torch.nn as nn


def n2n_loss(
    dae: nn.Module,
    noisy_e_a: torch.Tensor,
    noisy_e_b: torch.Tensor,
    detach_denoised: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute Noise2Noise MSE loss.

    loss = ||DAE(ẽ_a) - ẽ_b||²

    Args:
        dae: ResidualDAE (or any nn.Module with forward(x) -> denoised)
        noisy_e_a: (B, D) first noisy observation
        noisy_e_b: (B, D) second noisy observation (independent noise, same clean signal)
        detach_denoised: if False, e_denoised retains grad (for joint stage-2)

    Returns:
        (loss, e_denoised): scalar MSE loss, (B, D) denoised embeddings from ẽ_a
    """
    e_denoised = dae(noisy_e_a)
    loss = ((e_denoised - noisy_e_b) ** 2).sum(dim=-1).mean()
    return loss, e_denoised if not detach_denoised else e_denoised.detach()
