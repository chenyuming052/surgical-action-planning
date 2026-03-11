from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BackboneSpec:
    name: str
    feature_dim: int
    frozen: bool = True


DINOV3_VITB16 = BackboneSpec('dinov3_vitb16', 768, True)
LEMONFM = BackboneSpec('lemonfm', 1536, True)
