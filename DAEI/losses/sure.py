"""MC-SURE (Monte Carlo Stein's Unbiased Risk Estimator) loss for DAE training.

Uses three variance-reduction techniques:
1. Rademacher probes (z ∈ {-1, +1}) instead of Gaussian
2. Residual trace trick: compute Tr(J_g) instead of Tr(J_f) since f = id + g
3. Multiple probes (K=5-10) averaged per step
"""

from typing import Tuple

import torch
import torch.nn as nn


def rademacher_like(x: torch.Tensor) -> torch.Tensor:
    """Sample Rademacher random vector (±1) with same shape/device/dtype as x."""
    return torch.randint(0, 2, x.shape, device=x.device, dtype=x.dtype) * 2.0 - 1.0


def mc_sure_loss(
    dae: nn.Module,
    noisy_e: torch.Tensor,
    sigma: float,
    n_probes: int = 5,
    detach_denoised: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute MC-SURE loss for a residual DAE.

    SURE = ||g(y)||² + 2σ² · Tr(J_g(y))  (constant term -dσ² omitted)

    Because the -d·σ² term is omitted, the reported scalar can be **negative**
    even when training is healthy; compare runs using eval ``sure_delta_*`` metrics.

    where y = noisy embedding, g = DAE residual, J_g = Jacobian of g w.r.t. y.

    Args:
        dae: ResidualDAE with get_residual(x, sigma) method returning g(x)
        noisy_e: (B, D) noisy embeddings
        sigma: known noise standard deviation (scalar)
        n_probes: number of Rademacher probes for Hutchinson trace estimation
        detach_denoised: if False, e_denoised retains grad (needed for joint stage-2
            where CE loss must backprop through DAE via the denoised embedding)

    Returns:
        (loss, e_denoised): scalar SURE loss, (B, D) denoised embeddings
    """
    B, D = noisy_e.shape

    y = noisy_e.detach().requires_grad_(True)

    # Pass sigma as tensor for sigma-conditioned DAE
    sigma_t = torch.tensor(sigma, device=y.device, dtype=y.dtype).expand(B)
    g_out = dae.get_residual(y, sigma=sigma_t)

    # ||g(y)||² per sample
    recons = (g_out ** 2).sum(dim=-1)  # (B,)

    # Hutchinson trace estimation of Tr(J_g) using Rademacher probes
    trace_estimates = []
    for _ in range(n_probes):
        z = rademacher_like(y)
        # VJP: z^T J_g  (create_graph=True for second-order grad through DAE)
        vjp = torch.autograd.grad(
            outputs=g_out,
            inputs=y,
            grad_outputs=z,
            create_graph=True,
            retain_graph=True,
        )[0]
        # Tr(J_g) ≈ z^T J_g z = (z * vjp).sum(dim=-1)
        trace_est = (z * vjp).sum(dim=-1)  # (B,)
        trace_estimates.append(trace_est)

    mean_trace = torch.stack(trace_estimates).mean(dim=0)  # (B,)

    sigma_sq = sigma ** 2
    sure_per_sample = recons + 2.0 * sigma_sq * mean_trace  # (B,)
    loss = sure_per_sample.mean()

    e_denoised = y + g_out
    if detach_denoised:
        e_denoised = e_denoised.detach()

    return loss, e_denoised
