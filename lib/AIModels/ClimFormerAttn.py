import torch
import torch.nn as nn
from typing import Optional
import transformers as tr

class FeatureAttention(nn.Module):
    """
    A simple additive attention over the *feature* dimension.

    Given an input tensor of shape `(batch, seq_len, feature_dim)`, this layer
    learns a set of attention weights for each feature *per time step* and
    returns the element‑wise product `values * attn_weights` together with the
    weights themselves.  The formulation is deliberately lightweight so that it
    can be plugged in *before* the standard Informer embedding/projection logic.
    """

    def __init__(self, feature_dim: int, hidden_dim: Optional[int] = None):
        super().__init__()
        hidden_dim = hidden_dim or feature_dim
        self.proj1 = nn.Linear(feature_dim, hidden_dim)
        self.proj2 = nn.Linear(hidden_dim, feature_dim)
        self.activation = nn.Tanh()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, values: torch.Tensor):
        # values: (B, T, F)
        scores = self.proj2(self.activation(self.proj1(values)))  # (B, T, F)
        attn_weights = self.softmax(scores)                       # (B, T, F)
        return values * attn_weights, attn_weights


class ClimFormer(tr.InformerForPrediction):
    """
    InformerForPrediction with *feature wise* attention preprocessing.

    The attention layer rescales the raw *past* and (optionally) *future*
    value tensors along the feature dimension before they are consumed by the
    Informer architecture.  Everything else including lagged subsequence
    construction, scaling, encoder/decoder stacks, and distribution head is
    left untouched by delegating to `super().forward`.

    """

    def __init__(self, config):
        super().__init__(config)
        # ``config.input_size`` is the number of input variables (features)
        self.feature_attention = FeatureAttention(config.input_size)

    def _apply_feature_attention(self, values: torch.Tensor) -> torch.Tensor:
        weighted_values, _ = self.feature_attention(values)
        return weighted_values

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
        # 1) Apply feature‑wise attention
        past_values = self._apply_feature_attention(past_values)
        if future_values is not None:
            future_values = self._apply_feature_attention(future_values)

        # 2) Call the original Informer forward pass
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
