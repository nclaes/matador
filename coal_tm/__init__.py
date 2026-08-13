"""coal_tm — minimal CLI for the Coalesced Tsetlin Machine RTL flow.

Exclusively supports the Coalesced TM (shared clause bank, per-class
weighted sum, argmax). Vanilla TM, synthesis, and deployment are out of
scope here — see legacy/ for the retired code that used to attempt them.
"""

import sys
from pathlib import Path

# The vendored utils/tmu library's own C-extension build script names the
# compiled module "tmu.tmulib" (see utils/tmu/lib/tmulib_extension_build.py),
# and most of its internal modules import it as the top-level "tmu" package
# accordingly — but until this package normalized a handful of stray
# "from utils.tmu..." imports (added when the library was nested under
# utils/, but never rewritten consistently), `import tmu` failed on this
# branch under any sys.path setup at all. Putting utils/ on sys.path here
# makes "import tmu.*" resolve the way the library's own internals expect.
_UTILS_DIR = Path(__file__).resolve().parent.parent / "utils"
if _UTILS_DIR.is_dir() and str(_UTILS_DIR) not in sys.path:
    sys.path.insert(0, str(_UTILS_DIR))

__version__ = "0.1.0"
