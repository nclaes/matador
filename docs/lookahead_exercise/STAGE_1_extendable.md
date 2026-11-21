# Stage 1 — Make RedressBackend extendable

## 1. What changed and why

- **`RedressBackend` now has hooks.** A subclass can change the source folder, top module,
  encoder, expected-output model and extra Verilog parameters without copying ~1,000 lines.
- **Four class attributes:** `_src_dir`, `_verilator_dir`, `_top_module` and `_static_sources`.
- **Three methods:** `encode_stream()`, `expected_predictions()` and `_extra_params()`.
- **Every hard-coded `redress_accel` that reaches a generated file now goes through
  `self._top_module`.** That covers the top file, the testbench DUT, `run_iverilog.sh`,
  the Verilator `Makefile`/`tb_top.cpp` and the generated README.
- **The Verilator files are copied with the top-module name substituted.**
  For `redress` the substitution is an identity copy, so the files stay byte-identical.
  A `make` variable would have changed the generated Makefile.
- **Static submodules are resolved per file:** `common/src`, then the subclass's own `_src_dir`,
  then redress's `src/`. A subclass only ships the files it actually changes.
- **Proof of no behaviour change:** 4 bundles (70 files) generated before and after are
  byte-identical, and all 59 existing redress tests still pass.
- **2 new tests** use a throwaway subclass (`alt_accel`, with an extra `EXTRA_KNOB` parameter)
  to check that every hook reaches every generated file. One of them simulates the subclass
  bundle and a reprogram suite under iverilog.
- **Added `docs/lookahead_exercise/scripts/make_bundles.py`**, the before/after bundle generator.
  You can reuse it for any later refactor.
- `emulator.py` is untouched. It still calls the codec directly, and Stage 5 replaces it.

**Where the code disagrees with EXERCISE.pdf**

| EXERCISE.pdf | This repo / this stage |
|---|---|
| "Python must pass ruff" | `ruff check .` already reports **1817** findings repo-wide, and `rtl.py` alone has 20. Both counts are the same before and after this stage. The new and changed lines are ruff-clean, and the new files pass ruff fully. |
| Diff "a bundle for the same model and config" | The diff covers 4 cases: 4-of-4 classes, a 97-vector multi-batch model, `batch_width=16`, and a two-model reprogram suite. |
| `expected_predictions(words, n_classes, X_pack, config)` | Implemented with exactly that signature. `W` comes from `config.batch_width`, the same value the old code passed. |

## 2. Files touched

- `matador/backends/redress/rtl.py`: hooks, plus replacing hard-coded names with hook calls
- `tests/backends/test_redress_rtl.py`: `_alt_backend` helper and 2 subclass tests
- `docs/lookahead_exercise/scripts/make_bundles.py`: deterministic before/after bundle generator
- `docs/lookahead_exercise/STAGE_1_extendable.md`: this file

## 3. Run it yourself

**a) The before/after bundle diff.** This is the main proof for this stage.
```
$ git stash push -- matador/backends/redress/rtl.py          # or: git switch --detach la-stage-0
$ python docs/lookahead_exercise/scripts/make_bundles.py build/bundles_before
redress: wrote 70 files under build/bundles_before
$ git stash pop                                               # or: git switch production_ready
$ python docs/lookahead_exercise/scripts/make_bundles.py build/bundles_after
redress: wrote 70 files under build/bundles_after
$ diff -r build/bundles_before build/bundles_after && echo IDENTICAL
IDENTICAL
```
Running the script twice on the *same* code also gives `IDENTICAL`. That's worth checking
first: a diff tool that reports differences between two unchanged runs proves nothing.

**b) The new tests: written first, failing on the old `rtl.py`**
```
$ git stash push -- matador/backends/redress/rtl.py
$ pytest -q tests/backends/test_redress_rtl.py -k subclass
FAILED tests/backends/test_redress_rtl.py::test_subclass_hooks_reach_every_generated_file
FAILED tests/backends/test_redress_rtl.py::test_subclass_bundle_and_reprogram_suite_simulate
2 failed, 31 deselected in 0.45s
$ git stash pop
$ pytest -q tests/backends/test_redress_rtl.py -k subclass
2 passed, 31 deselected in 0.47s
```

**c) Regression check**
```
$ pytest -q tests/backends/test_redress_codec.py tests/backends/test_redress_rtl.py
61 passed in 9.93s
$ ruff check tests/backends/test_redress_rtl.py docs/lookahead_exercise/scripts/make_bundles.py
All checks passed!
$ pytest -q
7 failed, 450 passed, 10 skipped in 103.08s (0:01:43)
```
The 7 failures are the same ones recorded in Stage 0, which don't involve redress.
Stage 0 had 448 passed; the 2 extra are this stage's new tests.

**d) Read the refactor as a diff**
```
git diff la-stage-0 la-stage-1 -- matador/backends/redress/
```

## 4. Look here

- `matador/backends/redress/rtl.py:108-122`: the hook list in the class docstring and the four
  class attributes. Each default value is exactly the constant the old code used.
- `rtl.py:416`: `_check_capacity` now calls `self.encode_stream(...)`. Generation, the
  reprogram suite and the resource estimate all go through here, so one hook covers them all.
- `rtl.py:595`: `_build_stimulus` gets expected outputs from `self.expected_predictions(...)`.
  In Stage 2 this is where `decode_faithful` plugs in.
- `rtl.py:481-496`: `_static_source_path` (per-file lookup order) and `_render_verilator_file`
  (name substitution).
- `rtl.py:541`: `+ self._extra_params(config)`. Because this is appended to `_capacity_params`,
  the top-module rendering *and* the testbench's `#(...)` overrides both pick up new parameters,
  so the two can't drift apart.
- `tests/backends/test_redress_rtl.py:372`: `_alt_backend` builds a subclass inside the test,
  with closures counting hook calls. It's a cheap way to prove a hook is actually called.

## 5. Concept notes

**What a subclass needs to change, and the hook for each**

| Need | Old code | Hook |
|---|---|---|
| Source folder | `_SRC_DIR` constant | `_src_dir` |
| Verilator folder | `_VERILATOR_DIR` constant | `_verilator_dir` |
| Top-module file and name | `"redress_accel.v"` / `redress_accel` (13 places in `rtl.py`, 9 in `verilator/`) | `_top_module` |
| Static submodules | `_STATIC_SOURCES` + `_SRC_DIR` | `_static_sources` + `_static_source_path()` |
| Encoder | `codec.mustang_encode(tmir)` | `encode_stream(tmir, config)` |
| Expected-output model | `codec.mustang_decode_and_infer(...)` | `expected_predictions(words, n_classes, X_pack, config)` |
| Extra Verilog parameters | none | `_extra_params(config)` |
| README text | f-string in `_gen_readme` | already a method; it now uses `self.name` / `self._top_module` |
| Sim script source list | `_STATIC_SOURCES` + `"redress_accel.v"` | built from the hooks |

**Template method pattern.** The base class owns the *order* of work (encode → capacity check →
render → stimulus → testbench), and subclasses override individual *steps*. The hooks are
small and pure (data in, data out), so a subclass can't skip a step by accident.

**Why "byte-identical" is the right bar for a refactor.** If a single byte differs, you have to
argue that the difference doesn't matter. If nothing differs, there's nothing to argue about.
That's why the Verilator files use text substitution and not a `make` variable.

## 6. Check yourself

1. Why make this its own stage instead of folding it into Stage 5?
2. What could an identical-bundle diff miss? Think about other configs and paths that only run at runtime.
3. `_static_source_path` checks the subclass's `_src_dir` before redress's `src/`. What would go wrong
   with the opposite order?
4. Why is `_extra_params` appended inside `_capacity_params` and not applied separately in `_render_top`?

## 7. If something goes wrong

- **`diff -r` reports differences on unchanged code.** Something non-deterministic got into the bundle
  (a timestamp, or a path from a different output folder). `make_bundles.py` always generates into
  `build/_bundle_work` for this reason. Check you didn't point two runs at different working folders.
- **`RuntimeError: expected exactly one 'parameter X = ...' in <file>`.** An `_extra_params` name
  doesn't exist in the top module, or appears twice. The message now names the actual file.
- **Verilator can't find `V<name>.h`.** The top-module name wasn't substituted in `tb_top.cpp`.
  Check `_render_verilator_file` and that `_VERILATOR_TOP_TOKEN` still matches the static files.
- **The subclass bundle is missing `redress_bram.v`.** `_static_source_path` couldn't find it in
  common, `_src_dir` or redress `src/`. Check the file name in `_static_sources`.

## 8. Answers

1. A refactor stage and a feature stage that can each be checked on their own. If Stage 5 changes
   behaviour, you know it's the new feature and not the restructuring. The bundle diff only works
   as proof when nothing else changes at the same time.
2. It only checks the configs and models you tried. It misses other parameter values, paths that
   only run at runtime (the CLI's `simulate`/`emulate`, the Verilator build itself), and anything
   that isn't written to disk. The 4 cases and the reprogram suite narrow the gap; the existing
   tests (including the Verilator one) cover some of the rest. Neither is a proof for every input.
3. A subclass could never override a submodule. Its own `redress_bram.v` would be ignored in
   favour of redress's copy, with no error, which is the worst kind of bug.
4. `_capacity_params` feeds both `_render_top` (the parameter defaults in the copied `.v`) and the
   testbench's DUT `#(...)` overrides. Appending there means one list drives both, so a new
   parameter can't end up in one and not the other.
