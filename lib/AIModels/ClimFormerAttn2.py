import math
from typing import Optional

import torch
import torch.nn as nn
import transformers as tr


class FeatureSelfAttention(nn.Module):
    """Feature‑wise self‑attention **with residual gating**.

    The layer treats *features* as tokens and performs scaled dot‑product
    attention **independently at every time step**.  A *residual* connection
    with a learnable scalar gate `gamma` (initialised to *0*) preserves the
    original scale of each feature series while letting the model gradually
    incorporate cross‑feature information:

    .. code-block::

        out = x + gamma * A(x)          # A(x) = attention‑mixed features

    Because `gamma` starts at *0*, the network behaves as an **identity mapping**
    at initialisation, eliminating the risk of early‑training scale collapse
    observed with a plain weighted average.
    """

    def __init__(self, feature_dim: int, projection_dim: Optional[int] = None):
        super().__init__()
        projection_dim = projection_dim or feature_dim
        # Linear projections for Q and K (values are the raw features)
        self.W_q = nn.Linear(feature_dim, projection_dim, bias=False)
        self.W_k = nn.Linear(feature_dim, projection_dim, bias=False)
        self.scale = math.sqrt(projection_dim)

        # Per‑feature residual gate γ ∈ ℝ^F (broadcast over batch/time)
        self.gamma = nn.Parameter(torch.zeros(1, 1, feature_dim))  # shape (1,1,F)

    def forward(self, x: torch.Tensor):
        """Return rescaled output and attention weights.

        Parameters
        ----------
        x : torch.Tensor, shape *(B, T, F)*.
        """
        Q = self.W_q(x)  # (B, T, F_q)
        K = self.W_k(x)  # (B, T, F_q)

        # Compute attention across feature dimension
        # scores shape: (B, T, F, F)
        scores = torch.matmul(Q.unsqueeze(-1), K.unsqueeze(-2)) / self.scale
        attn_weights = torch.softmax(scores, dim=-1)

        # Weighted sum of original values (B, T, F)
        mixed = torch.matmul(attn_weights, x.unsqueeze(-1)).squeeze(-1)

        # Residual gating preserves scale
        out = x + self.gamma * mixed  # per‑feature gating
        return out, attn_weights


class ClimFormer(tr.InformerForPrediction):
    """InformerForPrediction with feature‑wise residual self‑attention."""

    def __init__(self, config):
        super().__init__(config)
        self.feature_attention = FeatureSelfAttention(config.input_size)

    def _apply_feature_attention(self, values: torch.Tensor) -> torch.Tensor:
        weighted_values, _ = self.feature_attention(values)
        return weighted_values

    # ------------------------------------------------------------
    # Forward – identical signature to InformerForPrediction
    # ------------------------------------------------------------
    def forward(
        self,
        past_values: torch.Tensor,
        past_time_features: torch.Tensor,
        past_observed_mask: torch.Tensor,
        static_categorical_features: Optional[torch.Tensor] = None,
        static_real_features: Optional[torch.Tensor] = None,
        future_values: Optional[torch.Tensor] = None,
        future_time_features: Optional[torch.Tensor] = None,
        future_observed_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        # 1) Apply feature self‑attention with residual scaling
        past_values = self._apply_feature_attention(past_values)
        if future_values is not None:
            future_values = self._apply_feature_attention(future_values)

        # 2) Continue through Informer
        return super().forward(
            past_values=past_values,
            past_time_features=past_time_features,
            past_observed_mask=past_observed_mask,
            static_categorical_features=static_categorical_features,
            static_real_features=static_real_features,
            future_values=future_values,
            future_time_features=future_time_features,
            future_observed_mask=future_observed_mask,
            **kwargs,
        )
