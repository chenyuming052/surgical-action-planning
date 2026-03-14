from __future__ import annotations

import torch
import torch.nn as nn


class ActionTokenEncoder(nn.Module):
    """Encodes surgical action state for dynamics conditioning.

    Coarse token: Linear(13, 64) from instrument(6) + phase(7).
    Fine token: SetTransformer over active triplet embeddings -> 64-d.
    Fusion MLP: combine coarse + fine -> 64-d output.
    Default embedding for G7 videos (no action labels).
    Teacher forcing: rho param controls GT vs predicted label mixing.
    """

    def __init__(
        self,
        instrument_dim: int = 6,
        phase_dim: int = 7,
        triplet_vocab_size: int = 100,
        triplet_embed_dim: int = 32,
        token_dim: int = 64,
        set_transformer_heads: int = 4,
    ):
        super().__init__()
        # Coarse token: instrument + phase -> token_dim
        self.coarse = nn.Linear(instrument_dim + phase_dim, token_dim)

        # Fine token: set transformer over active triplet embeddings
        self.triplet_embedding = nn.Embedding(triplet_vocab_size, triplet_embed_dim)
        self.set_attn = nn.MultiheadAttention(
            embed_dim=triplet_embed_dim,
            num_heads=set_transformer_heads,
            batch_first=True,
        )
        self.set_proj = nn.Linear(triplet_embed_dim, token_dim)

        # Fusion: coarse + fine -> token_dim
        self.fusion = nn.Sequential(
            nn.Linear(token_dim * 2, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )

        # Default embedding for G7 (no action labels)
        self.default_token = nn.Parameter(torch.zeros(token_dim))

    def forward(
        self,
        instrument_labels: torch.Tensor,
        phase_labels: torch.Tensor,
        triplet_indices: torch.Tensor,
        triplet_mask: torch.Tensor,
        has_action_labels: torch.Tensor,
        predicted_instrument: torch.Tensor | None = None,
        predicted_phase: torch.Tensor | None = None,
        rho: float = 1.0,
    ) -> torch.Tensor:
        """
        Args:
            instrument_labels: [B, T, 6] ground truth instrument binary
            phase_labels: [B, T, 7] ground truth phase one-hot
            triplet_indices: [B, T, max_triplets] active triplet indices
            triplet_mask: [B, T, max_triplets] mask for valid triplets
            has_action_labels: [B] boolean mask for videos with action labels
            predicted_instrument: [B, T, 6] model predictions (for teacher forcing)
            predicted_phase: [B, T, 7] model predictions (for teacher forcing)
            rho: teacher forcing ratio (1.0 = all GT, 0.0 = all predicted)

        Returns:
            action_token: [B, T, 64]
        """
        B, T = instrument_labels.shape[:2]

        # --- Teacher forcing: mix GT and predicted labels ---
        if predicted_instrument is not None:
            inst = rho * instrument_labels + (1 - rho) * predicted_instrument
        else:
            inst = instrument_labels
        if predicted_phase is not None:
            phase = rho * phase_labels + (1 - rho) * predicted_phase
        else:
            phase = phase_labels

        # --- Coarse token: concat instrument(6) + phase(7) -> Linear -> 64-d ---
        coarse_input = torch.cat([inst, phase], dim=-1)  # [B, T, 13]
        coarse_token = self.coarse(coarse_input)  # [B, T, 64]

        # --- Fine token: embed triplet indices, set self-attention, pool, project ---
        # triplet_indices: [B, T, max_triplets], triplet_mask: [B, T, max_triplets]
        max_triplets = triplet_indices.shape[-1]
        flat_indices = triplet_indices.reshape(B * T, max_triplets)  # [B*T, M]
        flat_mask = triplet_mask.reshape(B * T, max_triplets)  # [B*T, M]

        triplet_embeds = self.triplet_embedding(flat_indices)  # [B*T, M, 32]
        # key_padding_mask: True means ignore, so invert the valid mask
        key_padding_mask = ~flat_mask.bool()  # [B*T, M]
        attn_out, _ = self.set_attn(
            triplet_embeds, triplet_embeds, triplet_embeds,
            key_padding_mask=key_padding_mask,
        )  # [B*T, M, 32]

        # Mean pool over valid triplets
        valid_counts = flat_mask.float().sum(dim=-1, keepdim=True).clamp(min=1)  # [B*T, 1]
        pooled = (attn_out * flat_mask.float().unsqueeze(-1)).sum(dim=1) / valid_counts  # [B*T, 32]
        fine_token = self.set_proj(pooled)  # [B*T, 64]
        fine_token = fine_token.reshape(B, T, -1)  # [B, T, 64]

        # --- Fusion: concat coarse + fine -> fusion MLP -> 64-d ---
        fused = self.fusion(torch.cat([coarse_token, fine_token], dim=-1))  # [B, T, 64]

        # --- For videos without action labels, use default_token ---
        # has_action_labels: [B] boolean
        if has_action_labels is not None:
            mask = has_action_labels.float().unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1]
            default = self.default_token.unsqueeze(0).unsqueeze(0).expand(B, T, -1)  # [B, T, 64]
            fused = mask * fused + (1 - mask) * default

        return fused
