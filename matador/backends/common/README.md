# matador/backends/common

RTL modules shared verbatim by more than one backend — tested once, copied
into every consuming backend's generated `RTL/src/` by its own `generate()`,
same convention as each backend's own static (`vendor/src/`, `src/`) files.
Not a Python package; nothing imports `matador.backends.common` — these are
plain `.v` files referenced by path from each consuming backend's `rtl.py`.

| File | Used by |
|---|---|
| `src/argmax_rt.v` | `vanilla_gp_tiled`, `redress`, `vanilla_hardwired` (`scoring_mode: templated`) |

Before editing a file here, remember every consumer copies it verbatim at
`generate()` time — re-run each consuming backend's test suite
(`tests/backends/test_gp_tiled.py`, `test_redress_rtl.py`, `test_hardwired.py`)
after a change.
