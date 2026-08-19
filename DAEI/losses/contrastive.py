"""InfoNCE contrastive loss for DAE training.

Ensures that denoised embeddings stay close to their corresponding clean
embeddings while remaining distinguishable from other samples in the batch.
"""

import torch
import torch.nn.functional as F


def info_nce_loss(
    denoised: torch.Tensor,
    clean: torch.Tensor,
    tau: float = 0.07,
) -> torch.Tensor:
    """Symmetric InfoNCE loss between denoised and clean embeddings.

    Args:
        denoised: (B, D) denoised embeddings from DAE.
        clean: (B, D) corresponding clean (noise-free) embeddings.
        tau: temperature scaling factor.

    Returns:
        Scalar loss.
    """
    denoised_norm = F.normalize(denoised, dim=-1)
    clean_norm = F.normalize(clean, dim=-1)

    # (B, B) similarity matrix
    logits = denoised_norm @ clean_norm.t() / tau
    labels = torch.arange(logits.size(0), device=logits.device)

    loss_d2c = F.cross_entropy(logits, labels)
    loss_c2d = F.cross_entropy(logits.t(), labels)
    return (loss_d2c + loss_c2d) * 0.5
