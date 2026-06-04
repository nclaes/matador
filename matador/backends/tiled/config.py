"""Tiled backend configuration.

TiledAcceleratorConfig is the canonical name for this backend's config.
The class is currently defined in matador.config.schema as TMAcceleratorConfig
and aliased here.  It will migrate to this module in a future cleanup once
all call sites have been updated.
"""

from matador.config.schema import TMAcceleratorConfig as TiledAcceleratorConfig

__all__ = ["TiledAcceleratorConfig"]
