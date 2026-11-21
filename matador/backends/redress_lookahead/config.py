"""Configuration for the redress_lookahead backend.

Everything RedressAcceleratorConfig has, plus the two look-ahead knobs.
e_lookahead_depth=0 with cc_lookahead=False reproduces redress exactly,
including under faults: that is the A/B switch.
"""

from __future__ import annotations

from pydantic import Field, field_validator

from matador.backends.redress.config import RedressAcceleratorConfig

# Depths with an RTL implementation. Depth 2 exists only in the software model
# (decode_faithful / campaign), because a two-word look-ahead needs a prefetch
# window, not just the one idle read slot (see STAGE_4 README).
RTL_E_DEPTHS = (0, 1)


class RedressLookaheadConfig(RedressAcceleratorConfig):
    """redress plus E/CC look-ahead decoding (redress_la_accel.v)."""

    e_lookahead_depth: int = Field(
        default=1,
        description=(
            "E look-ahead depth: 0 = as built (a single E flip shifts every later "
            "class), 1 = accept an E change only if the next word carries it. "
            "Classes are padded to at least 4 words at depth 1."
        ),
    )
    cc_lookahead: bool = Field(
        default=True,
        description=(
            "Apply the same one-word rule to CC (clause boundaries). The encoder "
            "then stores every one-literal clause's literal twice."
        ),
    )

    @field_validator("e_lookahead_depth")
    @classmethod
    def _depth_has_rtl(cls, v: int) -> int:
        if v not in RTL_E_DEPTHS:
            raise ValueError(
                f"e_lookahead_depth={v} has no RTL implementation (supported: "
                f"{', '.join(map(str, RTL_E_DEPTHS))}). Depth 2 is available in the "
                "software model only: python -m matador.backends.redress_lookahead."
                "campaign --variant v3d2 v4d2"
            )
        return v

    @property
    def decoder_mode(self) -> dict:
        """decode_faithful() keyword arguments matching this config."""
        return {
            "e_depth": self.e_lookahead_depth,
            "cc_lookahead": self.cc_lookahead,
            "e_init": 1 if self.e_lookahead_depth > 0 else None,
        }
