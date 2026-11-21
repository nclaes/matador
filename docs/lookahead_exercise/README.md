# REDRESS look-ahead decoder — staged exercise

The brief is [EXERCISE.pdf](EXERCISE.pdf). It is project-internal, so don't circulate it outside the project.
Each stage below is one commit on `production_ready`, tagged `la-stage-N`, so
`git diff la-stage-3 la-stage-4` shows exactly one stage of work. Every stage README follows the same
eight sections: what changed, files, run it yourself, look here, concepts, check yourself,
troubleshooting, answers.

| Stage | README | What it delivers |
|---|---|---|
| 0 | [STAGE_0_orientation.md](STAGE_0_orientation.md) | Decode-path map with a real cycle trace; the **argmax class-count bug** fixed in `redress` (full power-of-two models always predicted class 0) |
| 1 | [STAGE_1_extendable.md](STAGE_1_extendable.md) | `RedressBackend` hooks; refactor proven byte-identical over 4 bundles |
| 2 | [STAGE_2_faithful.md](STAGE_2_faithful.md) | `decode_faithful` (RTL-exact under faults), corrupted-RTL replay with a negative control, first campaign |
| 3 | [STAGE_3_lookahead_software.md](STAGE_3_lookahead_software.md) | v3/v4 in software, hardened encoder, minimum class lengths **4 / 6 / 7** found by exhaustive search |
| 4 | [STAGE_4_lookahead_rtl.md](STAGE_4_lookahead_rtl.md) | `redress_la_accel.v` (17 code lines changed), zero extra cycles, 5 RTL mutations all caught |
| 5 | [STAGE_5_backend.md](STAGE_5_backend.md) | `redress_lookahead` registered: config, emulator, CLI, docs; shared `simulate` path bug fixed |
| 6 | [STAGE_6_real_model.md](STAGE_6_real_model.md) + [walkthrough](../walkthroughs/redress_lookahead.md) | Mammographic: N × P **288 → 3.37 (85×)** for +0.1% words; 200/200 replays match |
| 7 | [STAGE_7_handoff.md](STAGE_7_handoff.md) | Register table for EMBER; mini saboteur; **the E latch is the new weak point**; open items |

## Scripts (`scripts/`)

| Script | Use |
|---|---|
| `make_bundles.py` | Generate a fixed set of bundles for before/after refactor diffs (`diff -r`). |
| `vcd_table.py` | Print selected DUT signals per clock cycle from a `.vcd`: GTKWave as text. |
| `replay_sample.py` | Random (or `--fields`-targeted) corrupted-memory replays through both RTLs. |
| `saboteur.py` | Flip a named register at a random inference cycle; for learning, not a substitute for EMBER. |

The campaign itself is a module: `python -m matador.backends.redress_lookahead.campaign --help`.

## Tests

```
pytest -q tests/backends/test_redress_rtl.py tests/backends/test_redress_codec.py \
          tests/backends/test_redress_lookahead*.py
```
