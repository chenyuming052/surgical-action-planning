from __future__ import annotations

import torch
import torch.nn as nn


class EventDyn(nn.Module):
    """Version B event-conditioned transition (core contribution).

    FiLM-conditioned MLP: mu_plus = h_t + gamma * r + beta
    tau_embed: K=20 learnable hazard-bin embeddings (64-d each).
    Outputs: mu_plus [B,T,512], log_var [B,T,512].
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        num_bins: int = 20,
        bin_embed_dim: int = 64,
        film_hidden_dim: int = 256,
    ):
        super().__init__()
        self.tau_embed = nn.Embedding(num_bins, bin_embed_dim)

        # FiLM generator: from bin embedding -> gamma, beta
        self.film_gen = nn.Sequential(
            nn.Linear(bin_embed_dim, film_hidden_dim),
            nn.GELU(),
            nn.Linear(film_hidden_dim, hidden_dim * 2),  # gamma + beta
        )

        # Residual predictor
        self.residual_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),  # mu_residual + log_var
        )

    def forward(
        self,
        h_t: torch.Tensor,
        tau_bin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_t: [B, T, 512] encoder hidden states
            tau_bin: [B, T] predicted hazard bin indices

        Returns:
            mu_plus: [B, T, 512] predicted post-change state
            log_var: [B, T, 512] uncertainty
        """
        # Embed tau_bin: [B, T] -> [B, T, 64]
        tau_embed = self.tau_embed(tau_bin)  # [B, T, 64]

        # FiLM generator: tau_embed -> gamma, beta
        film_out = self.film_gen(tau_embed)  # [B, T, 1024]
        gamma, beta = film_out.chunk(2, dim=-1)  # each [B, T, 512]

        # Residual predictor: h_t -> mu_residual, log_var
        residual_out = self.residual_mlp(h_t)  # [B, T, 1024]
        mu_residual, log_var = residual_out.chunk(2, dim=-1)  # each [B, T, 512]

        # Apply FiLM conditioning
        mu_plus = h_t + gamma * mu_residual + beta  # [B, T, 512]

        return mu_plus, log_var


class ActionConditionedTransition(nn.Module):
    """Version A ablation extension with action token input.

    Extends HorizonConditionedTransition concept with action token.
    Input: [h_t; a_t; e_delta] = 640-d, horizons {1,3,5,10}.
    """

    def __init__(
        self,
        hidden_dim: int = 512,
        action_dim: int = 64,
        horizon_embed_dim: int = 64,
        horizons: tuple = (1, 3, 5, 10),
    ):
        super().__init__()
        self.horizons = list(horizons)
        self.horizon_to_idx = {h: i for i, h in enumerate(self.horizons)}
        self.embed = nn.Embedding(len(self.horizons), horizon_embed_dim)
        input_dim = hidden_dim + action_dim + horizon_embed_dim  # 640
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim * 2),  # pred_state + log_var
        )

    def forward(
        self,
        h_t: torch.Tensor,
        a_t: torch.Tensor,
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            h_t: [B, T, 512] encoder hidden states
            a_t: [B, T, 64] action token
            horizon: one of {1, 3, 5, 10}

        Returns:
            pred_state: [B, T, 512]
            log_var: [B, T, 512]
        """
        B, T, _ = h_t.shape

        # Get horizon index and embed
        horizon_index = self.horizon_to_idx[horizon]
        idx_tensor = torch.tensor(horizon_index, device=h_t.device)
        horizon_embed = self.embed(idx_tensor)  # [64]
        horizon_embed = horizon_embed.unsqueeze(0).unsqueeze(0).expand(B, T, -1)  # [B, T, 64]

        # Concat [h_t; a_t; horizon_embed] -> [B, T, 640]
        x = torch.cat([h_t, a_t, horizon_embed], dim=-1)

        # MLP -> chunk into pred_state and log_var
        out = self.mlp(x)  # [B, T, 1024]
        pred_state, log_var = out.chunk(2, dim=-1)  # each [B, T, 512]

        return pred_state, log_var
