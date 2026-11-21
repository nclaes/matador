# Stage 5 — Register `redress_lookahead` as a backend

## 1. What changed and why

- **Registry.** `redress_lookahead` is in `_BACKENDS` and `_DESCRIPTIONS`. `list-backends`,
  `registry`, the splash screen and `clean` all pick it up automatically.
- **Config.** `RedressLookaheadConfig` (added in Stage 4) subclasses `RedressAcceleratorConfig` and adds
  `e_lookahead_depth` (0/1; 2 is rejected with a message pointing to the software-only `v3d2` campaign)
  and `cc_lookahead`. It also has a `decoder_mode` property, so the generator, the emulator and replays
  all derive the same `decode_faithful` arguments from one place.
- **New `emulator.py`.** `RedressLookaheadEmulator.run()` is built on `decode_faithful` over the
  *hardened* stream. It optionally takes a corrupted `words` list, so you can emulate a faulty Instruction Memory.
- **`rtl.py`** gains `emulator_class` and a generated `RTL/README.md` section explaining the look-ahead.
  The redress README's "checked against `codec.mustang_decode_and_infer()`" line is corrected to `decode_faithful`.
- **New `examples/redress_lookahead.yaml`.**
- **CLI.** The `reprogram-suite` help text lists the new backend. `export-model-json` now **accepts**
  `redress_lookahead`. The decision, made for you since you asked me to run unattended: the exported
  JSON is the TMIR include array, and hardening happens at encode time, so the export is identical to redress's.
- **Docs.** A row in the README backend table, and a Developer.md section,
  "Extending an existing backend by subclassing".
- **Bug fixed in shared code (affects every backend).** `matador simulate` failed for any config with a
  *relative* `output_dir`. `validate_rtl` ran `vvp` and the Verilator binary with `cwd=` set to the sim
  folder, using paths relative to the caller. It's the same mistake I made in Stage 2's `run_iverilog`.
  The fix is one line: `rtl_dir = (config.output_dir / "RTL").resolve()`. There's a regression test,
  and plain `redress` was broken the same way before the fix.
- **New `tests/backends/test_redress_lookahead.py`** (19 functions, 24 cases), mirroring `test_redress_rtl.py`:
  - registry and config;
  - emulator vs reference, plus a corrupted-stream emulation;
  - generated files and README;
  - the A/B switch (depth 0 + CC off gives a byte-identical stimulus and expected outputs to redress);
  - capacity errors, including "hardening words count toward `max_includes`";
  - single- and multi-batch iverilog, the Verilator harness and a two-model reprogram suite;
  - the CLI (`generate`, `simulate`, relative paths, `list-backends`, `export-model-json`) and the example config.

**Where the code disagrees with EXERCISE.pdf**

| EXERCISE.pdf | This repo |
|---|---|
| `config.py` is created here | It was created in Stage 4 (the RTL tests needed it). This stage completes it. |
| "Decide with the user whether `export-model-json` should accept the new backend" | Accepted, for the reason above. To reverse it, remove the name from the tuple at `matador/cli.py:1006`. |
| Backend naming convention `<tm_variant>_<architecture>` (Developer.md) | `redress_lookahead` follows the brief and the existing `redress` name, not the convention. |
| Full suite green | `pytest -q`: **527 passed, 7 failed, 10 skipped**. The 7 are the same failures present before Stage 0, and none involve redress. 5 are stale `TestEmulateCLI` tests that call `matador emulate` without the now-required `--backend` option; 2 are TMU weighted/coalesced reference mismatches. |

## 2. Files touched

- `matador/backends/registry.py`: 2 entries
- `matador/backends/redress_lookahead/emulator.py`: new
- `matador/backends/redress_lookahead/rtl.py`: `emulator_class`, `_gen_readme`
- `matador/cli.py`: reprogram-suite help, `export-model-json`
- `matador/models/validator.py`: relative-path fix (shared)
- `examples/redress_lookahead.yaml`: new
- `README.md`: backend table row
- `docs/Developer.md`: subclassing section
- `tests/backends/test_redress_lookahead.py`: new
- `docs/lookahead_exercise/STAGE_5_backend.md`: this file

## 3. Run it yourself

Make a config from the example. `build/` is gitignored:
```bash
mkdir -p build/la_stage5
python -c "from matador.backends.redress_lookahead import campaign; \
campaign.make_synthetic_tmir(n_classes=4, cpc=12, density=0.08, n_samples=70, seed=7).to_yaml('build/la_stage5/model.yaml')"
sed -e 's#model_path: /work/TMIR/.*#model_path: build/la_stage5/model.yaml#' \
    -e 's#^output_dir: /work#output_dir: build/la_stage5#' \
    examples/redress_lookahead.yaml > build/la_stage5/redress_lookahead.yaml
```
Then (real output, trimmed):
```
$ matador list-backends
  ...
  redress_lookahead
    redress with E/CC look-ahead decoding: a single flipped E/CC bit in Instruction Memory no longer
    shifts every later class/clause. No extra cycles. Knobs: redress's, plus e_lookahead_depth (0/1), cc_lookahead.
    Config template:  examples/redress_lookahead.yaml
$ matador generate --backend redress_lookahead --config build/la_stage5/redress_lookahead.yaml
  vanilla TM  |  15 features  |  4 classes  |  48 clauses
  Generating [redress_lookahead]…
RTL written to: build/la_stage5/redress_lookahead/RTL
$ matador simulate --backend redress_lookahead --config build/la_stage5/redress_lookahead.yaml
  tb_redress_model           PASS   .../tb_redress_model.v:123: $finish called at 18325000 (1ps)
  tb_system [verilator]      PASS   tb_top: ALL TESTS PASSED (96 beats checked)
2 passed, 0 failed
$ matador emulate --backend redress_lookahead --config build/la_stage5/redress_lookahead.yaml --verify
70 passed, 0 failed
Cross-checking emulator against reference inference…
  Emulator matches reference on all 70 vector(s).
Emulation passed.
```
Before the validator fix, the same `simulate` printed
`Error: [Errno 2] No such file or directory: 'build/la_stage5/redress_lookahead/RTL/sim/verilator/obj_dir/Vredress_la_accel'`,
and so did `--backend redress`.

Tests:
```
$ pytest -q tests/backends/test_redress_lookahead.py
24 passed in 51.51s
$ pytest -q tests/test_cli.py tests/test_splash.py tests/test_clean.py      # registry-wide
61 passed in 18.89s
$ pytest -q
7 failed, 527 passed, 10 skipped in 181.07s (0:03:01)
```
The regression test fails without the validator fix (`1 failed`) and passes with it (`1 passed`).

## 4. Look here

- `matador/backends/registry.py:25` and `:39`: the two new lines. They are the whole of "registration".
- `matador/backends/redress_lookahead/rtl.py`: the complete backend, 133 lines, about half of them the generated README text.
  Compare it with `redress/rtl.py` (1,050 lines). That's what Stage 1 bought.
- `build/la_stage5/redress_lookahead/RTL/README.md`, section "Look-ahead (what this backend adds to
  redress)". Would a colleague understand it without the PDF?
- `tests/backends/test_redress_lookahead.py::test_depth0_cc_off_reproduces_redress_bundle_inputs`:
  the A/B switch, as a test.
- `matador/models/validator.py:135-138`: the one-line shared fix.

## 5. Concept notes

**The plugin registry.** `_BACKENDS` maps a name to a `(module, class)` pair that's imported *lazily*.
Importing the registry never imports any backend, so a broken optional backend can't break `matador --help`.
`list-backends`, `registry`, the splash dashboard and `clean` iterate over `list_backends()`, which is
why they needed no changes. Only two CLI places hard-code backend names (`reprogram-suite` help and
`export-model-json`), and both were updated.

**Pydantic config inheritance.** `RedressLookaheadConfig(RedressAcceleratorConfig)` inherits every
field *and every validator*. The BRAM-budget `model_validator` still runs, and the new
`field_validator` adds one rule. `model_config = ConfigDict(extra="ignore")` is inherited too, so an
old `redress.yaml` validates as a `redress_lookahead` config (with the look-ahead defaults).

**The three-layer contract, test by test**

| Layer | Meaning | Tests in this stage |
|---|---|---|
| Spec | `reference.predict` | `test_emulator_matches_reference` (emulator == spec) |
| Emulator | `decode_faithful` via `RedressLookaheadEmulator` | `test_emulator_*`, `test_depth0_cc_off_reproduces_redress_bundle_inputs` |
| RTL | iverilog / Verilator | `test_rtl_simulation_passes`, `test_verilator_harness_passes`, `test_reprogram_suite_across_two_models`, `test_cli_generate_and_simulate` (RTL == emulator's expectations) |

## 6. Check yourself

1. What does `supports_reprogramming = True` commit you to?
2. Which layer of the three-layer contract does each of your tests cover?
3. Why does `test_capacity_check_counts_hardening_words` matter for someone reprogramming a deployed core?
4. Why was `matador simulate` broken for relative paths, when the generated `run_iverilog.sh` worked fine from anywhere?

## 7. If something goes wrong

- **`Unknown backend: 'redress_lookahead'`.** You're running an older install. Use `pip install -e ".[dev]"` in this venv.
- **`model encodes to N MUSTANG instruction words, exceeding max_includes`, for a model that fits redress.**
  The hardening words count. Raise `max_includes` (the BRAM budget is rechecked).
- **`e_lookahead_depth=2 has no RTL implementation`.** That's intended. Use the campaign's `v3d2`/`v4d2` in software.
- **`simulate` fails with `No such file or directory ... obj_dir/V...` or `Unable to open input file`.**
  You're on a version without the `validator.py` fix. Use an absolute `output_dir`, or update.

## 8. Answers

1. That `build_reprogram_suite()` works: an already-generated core must accept *any* model that fits its
   capacity at runtime, hardened per model at load time, and the multi-model suite must pass across
   models. The look-ahead parameters are synthesis-time, so every model loaded into a depth-1 core must be
   hardened for depth 1, and the backend's `encode_stream` guarantees that.
2. See the table in Section 5. The registry, config and CLI tests check the plumbing *around* the contract.
3. A deployed core has a fixed `MAX_INCLUDES`. A model that fit as a plain MUSTANG stream may not fit once
   hardened. Finding out at generate or reprogram time, with a clear message, beats a silently truncated load.
4. `run_iverilog.sh` resolves its own location at runtime (`$(dirname "$0")`). `validate_rtl` built paths
   from the config's `output_dir` as given, then ran the tools with `cwd=` set to a *different* folder, so
   relative paths broke. Absolute paths are the fix (`Path.resolve()`).
