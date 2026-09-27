"""
Variational Autoencoder model compatible with Informer-style data preparation.

This module provides a VAE that accepts the same inputs used by the notebook
(past_values, past_time_features, future_time_features, etc.) and is configured
using the InformerConfig class.

Config Parameter Correspondence:
================================
- input_size (config): Number of input time series variables
- output_size (config): Number of output variables (default: same as input_size)
- context_length (config): Length of input sequence
- lags_sequence (config): List of lag indices (max used to calculate TIN)
- prediction_length (config): Length of output sequence
- d_model (config): Hidden dimension for encoder/decoder (default latent_dim)
- encoder_ffn_dim (config): Hidden dimension for encoder FFN
- decoder_ffn_dim (config): Hidden dimension for decoder FFN
- num_time_features (config): Number of time feature channels
- dropout (config): Dropout rate

Calculated Dimensions:
======================
- TIN = context_length + max(lags_sequence): Total input time steps
- encoder_input_dim = TIN * (input_size + num_time_features): Flattened encoder input
- latent_dim = d_model: Latent space dimension
- decoder_condition_dim = prediction_length * num_time_features: Future time features
- decoder_input_dim = latent_dim + decoder_condition_dim
- decoder_output_dim = prediction_length * output_size: Flattened output
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.utils import ModelOutput
from transformers.modeling_outputs import SampleTSPredictionOutput

from .Informer.configuration_informer import InformerConfig


@dataclass
class VAESeq2SeqOutput(ModelOutput):
    """
    Output container for the VAE model.

    Attributes
    ----------
    loss: torch.Tensor | None
        Total loss = reconstruction + KL (when future_values provided).
    prediction_outputs: torch.Tensor
        Predicted sequence of shape (B, prediction_length, output_size).
    mu: torch.Tensor
        Latent mean of shape (B, latent_dim).
    logvar: torch.Tensor
        Latent log-variance of shape (B, latent_dim).
    """

    loss: Optional[torch.Tensor] = None
    prediction_outputs: Optional[torch.Tensor] = None
    mu: Optional[torch.Tensor] = None
    logvar: Optional[torch.Tensor] = None


class VAEForPrediction(nn.Module):
    """
    Variational Autoencoder for sequence-to-sequence prediction.

    Compatible with the Informer-style data preparation used in the notebook.
    All dimensions are calculated from InformerConfig parameters.
    """

    def __init__(
        self,
        config: InformerConfig,
        use_time_features: bool = True,
        beta: float = 1.0,
    ) -> None:
        super().__init__()

        self.config = config
        self.beta = beta
        self.use_time_features = use_time_features

        # Extract config parameters
        self.input_size = config.input_size
        self.output_size = getattr(config, "output_size", self.input_size)
        self.prediction_length = config.prediction_length
        self.context_length = config.context_length
        self.lags_sequence = config.lags_sequence or [1, 2, 3, 4, 5, 6, 7]
        self.num_time_features = config.num_time_features
        self.dropout = config.dropout

        # Latent and hidden dimensions from config
        self.latent_dim = config.d_model
        self.encoder_hidden_dim = config.encoder_ffn_dim
        self.decoder_hidden_dim = config.decoder_ffn_dim

        # Calculate TIN (total input timesteps)
        self.tin = self.context_length + max(self.lags_sequence)

        # Calculate dimensions
        encoder_feature_size = self.input_size + (self.num_time_features if use_time_features else 0)
        self.encoder_input_dim = self.tin * encoder_feature_size

        self.decoder_condition_dim = 0
        if use_time_features and self.num_time_features > 0:
            self.decoder_condition_dim = self.prediction_length * self.num_time_features

        self.decoder_input_dim = self.latent_dim + self.decoder_condition_dim
        self.decoder_output_dim = self.prediction_length * self.output_size

        # Build encoder
        self.encoder = nn.Sequential(
            nn.Linear(self.encoder_input_dim, self.encoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.encoder_hidden_dim, self.encoder_hidden_dim),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(self.encoder_hidden_dim, self.latent_dim)
        self.fc_logvar = nn.Linear(self.encoder_hidden_dim, self.latent_dim)

        # Build decoder
        self.decoder = nn.Sequential(
            nn.Linear(self.decoder_input_dim, self.decoder_hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.decoder_hidden_dim, self.decoder_hidden_dim),
            nn.ReLU(),
            nn.Linear(self.decoder_hidden_dim, self.decoder_output_dim),
        )

    def _build_encoder_input(
        self,
        past_values: torch.Tensor,
        past_time_features: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Build encoder input by flattening and optionally concatenating time features."""
        if self.use_time_features and past_time_features is not None:
            enc_in = torch.cat([past_values, past_time_features], dim=-1)
        else:
            enc_in = past_values
        batch_size = enc_in.shape[0]
        return enc_in.reshape(batch_size, -1)

    def _build_decoder_condition(
        self,
        future_time_features: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Build decoder condition from future time features or return zeros."""
        if self.use_time_features and self.num_time_features > 0:
            if future_time_features is None:
                return torch.zeros(
                    batch_size, self.decoder_condition_dim, device=device, dtype=dtype
                )
            return future_time_features.reshape(future_time_features.shape[0], -1)
        return torch.zeros((batch_size, 0), device=device, dtype=dtype)

    @staticmethod
    def reparameterize(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(
        self,
        past_values: torch.Tensor,
        past_time_features: Optional[torch.Tensor] = None,
        past_observed_mask: Optional[torch.Tensor] = None,
        static_categorical_features: Optional[torch.Tensor] = None,
        static_real_features: Optional[torch.Tensor] = None,
        future_values: Optional[torch.Tensor] = None,
        future_time_features: Optional[torch.Tensor] = None,
        future_observed_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = True,
        **kwargs,
    ) -> VAESeq2SeqOutput | Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of VAE.

        Args:
            past_values: (batch_size, TIN, input_size)
            past_time_features: (batch_size, TIN, num_time_features)
            future_values: (batch_size, prediction_length, output_size)
            future_time_features: (batch_size, prediction_length, num_time_features)
            return_dict: Whether to return VAESeq2SeqOutput or tuple

        Returns:
            VAESeq2SeqOutput or tuple containing (loss, prediction_outputs, mu, logvar)
        """
        device = past_values.device
        dtype = past_values.dtype

        # Encode
        enc_in = self._build_encoder_input(past_values, past_time_features)
        hidden = self.encoder(enc_in)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        z = self.reparameterize(mu, logvar)

        # Decode
        cond = self._build_decoder_condition(future_time_features, past_values.shape[0], device, dtype)
        dec_in = torch.cat([z, cond], dim=-1)
        recon = self.decoder(dec_in)
        recon = recon.view(-1, self.prediction_length, self.output_size)

        # Compute loss
        loss = None
        if future_values is not None:
            recon_loss = F.mse_loss(recon, future_values, reduction="mean")
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = recon_loss + self.beta * kl_loss

        if not return_dict:
            return recon, loss

        return VAESeq2SeqOutput(
            loss=loss,
            prediction_outputs=recon,
            mu=mu,
            logvar=logvar,
        )

    @torch.no_grad()
    def generate(
        self,
        past_values: torch.Tensor,
        past_time_features: Optional[torch.Tensor] = None,
        future_time_features: Optional[torch.Tensor] = None,
        past_observed_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> SampleTSPredictionOutput:
        """
        Generate predictions without computing loss.

        Returns:
            SampleTSPredictionOutput with sequences of shape (batch_size, 1, prediction_length, output_size)
        """
        device = past_values.device
        dtype = past_values.dtype

        # Encode
        enc_in = self._build_encoder_input(past_values, past_time_features)
        hidden = self.encoder(enc_in)
        mu = self.fc_mu(hidden)
        logvar = self.fc_logvar(hidden)
        z = self.reparameterize(mu, logvar)

        # Decode
        cond = self._build_decoder_condition(future_time_features, past_values.shape[0], device, dtype)
        dec_in = torch.cat([z, cond], dim=-1)
        recon = self.decoder(dec_in)
        recon = recon.view(-1, self.prediction_length, self.output_size)

        sequences = recon.unsqueeze(1)
        return SampleTSPredictionOutput(sequences=sequences)


__all__ = ["VAEForPrediction", "VAESeq2SeqOutput"]
