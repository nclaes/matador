# Reproducing arXiv:2502.05640 (ETHEREAL)'s Table 1 setup

Booleanization + training configs for the eight datasets in Table 1 of
["ETHEREAL: Energy-efficient and High-throughput Inference using
Compressed Tsetlin Machine"](https://arxiv.org/pdf/2502.05640)
(tinyML Research Symposium '25), matching the paper's reported
`Classes`/`Features`/`Literals`/`Epochs`/`(Clauses per class, T, s)` as
closely as matador's real ingest recipes allow.

**Use these when you specifically want to compare against that paper's
numbers.** They are deliberately *not* the same as `data/Raw_Data_Bank.yaml`'s
own `booleanization.default_encoder` for each dataset — those are tuned as
sensible general-purpose defaults, independent of this one paper's specific
choices, and changing them would be a regression for anyone not trying to
reproduce this particular comparison.

## How the numbers were derived

The paper's `Literals` column is always `2 × Boolean bits` (standard TM
positive/negated-literal convention, matching matador's own
`n_literals = 2 * n_features`), so `Literals / 2` is the real target
Boolean bit count for each dataset. Where matador's real raw feature count
(verified via live `matador ingest`, not assumed) divides cleanly into that
target, the per-feature bit width below is exact. `clauses:` in matador's
training config is **per class** already (traced through
`matador/models/tmu_adapter.py`: `n_clauses_total = n_classes *
n_clauses_per_class` where `n_clauses_per_class` is exactly the value
passed as `config.clauses`) — so the paper's `Clauses*` column (itself
footnoted "number of clauses per class") maps directly with no conversion.

| Dataset | Target Boolean bits (`Literals/2`) | matador raw features | bits/feature used | Match |
|---|---|---|---|---|
| EMG | 160 | 8 | 20 | exact bit count — **see caveat below** |
| Gas sensor | 128 | 128 | 1 | exact |
| Gesture Phase (paper's "GPS") | 180 | 32 | 6 (→192) | approximate — **see caveat below** |
| Human Activity | 560 | 561 | 1 (→561) | effectively exact (off by 1) |
| Mammographic mass | 15 | 5 | 3 | exact |
| Sensorless drive | 144 | 48 | 3 | exact |
| Sport activity | 45 | 5625 | — | **not achievable, see caveat below** |
| Statlog | 360 | 18 | 20 | exact |

## Two honest caveats — read before trusting EMG or Sport activity

**Sport activity:** the paper's `Features=45` is the raw per-timestep
sensor-channel count (5 body-worn units × 9 axes) — implying their
classification unit is roughly one instantaneous 45-value reading, not a
whole segment. Matador's `sports` ingest recipe instead flattens each
125-row, 45-column segment *file* into one 5625-dimension sample (one
sample per file, not per row) — a fundamentally different sample
granularity, not just a different bit width. Hitting 45 target bits on a
5625-feature sample isn't a booleanization-width problem, it's a
dimensionality-reduction problem matador's per-column encoders don't solve.
Reproducing this exactly would need a *new* ingest capability (row-level
samples inheriting a file-derived label — `dir_of_txt`'s file-level mode
today only supports whole-file-flatten, not per-row-with-path-label), which
is a real, separate feature request, not implemented here. The config
below uses matador's existing segment-level samples with a 1-bit threshold
encoding (5625 bits) — internally consistent, but not literal parity with
the paper.

**EMG:** matador's live ingest reads raw per-timestep 8-channel rows
(verified: 3,390,325 train + 847,582 test samples — one sample per
timestamp). The paper's `Features=160` for an 8-channel EMG dataset
strongly implies a *windowed* representation (e.g. multiple statistical
features per channel per window), which is also what the dataset's old,
now-untracked matador reference data used ("8 channels x 12 bits, one RMS
value per channel per window", ~57K windowed samples — two orders of
magnitude fewer than matador's current raw per-row ingest). The config
below hits the paper's 160-bit target via a 20-bit thermometer per raw
channel, which is internally consistent and trains fine, but on
fundamentally different (per-instant, not per-window) samples than the
paper almost certainly used. Real windowed-feature ingest (RMS/MAV/etc.
over a rolling window) is, like Sport activity, a genuine feature gap, not
implemented here.

Everything else in the table is either exact or within rounding of exact.

## A third caveat: bit-count parity is not accuracy parity

Verified by actually running these configs (not just deriving them): full,
unmodified `mammographic`/`statlog` runs land close to the paper's own
reported vanilla-TM accuracy (82.81% vs. 82.38%; 76.33% vs. 71.76% — Table 2)
where the bit-width match is exact. **`gas_sensor` does not**: the full
200-epoch run above gets 19.02% test accuracy — barely above the 16.7%
random-guess floor for 6 classes, nowhere near the paper's ~87%. This isn't
a bug: retraining the identical architecture with an 8-bit-per-feature
thermometer instead of 1-bit (1024 bits instead of 128, so *not* matching
the paper's reported Features=128 anymore) reaches 98.71% on the same data.
The paper's own pipeline (Figure 9: "quantile binning" + "one-hot or
thermometer") evidently extracts more signal per feature than a single
global quantile split does for this specific dataset — matching their
*bit-count* exactly does not reproduce their *accuracy*, at least not with
matador's plain per-column quantile-thermometer/threshold encoders. Where
this config set uses 1 bit/feature (`gas_sensor`, `human_activity`,
`sport activity`), treat the resulting model's accuracy as unverified
against the paper even though its geometry matches Table 1 exactly; where
it uses more bits (`EMG`, `gesture_phase`, `mammographic`, `sensorless_drive`,
`statlog`), accuracy tracked the paper reasonably well in the two cases
actually checked.

## Usage

```bash
matador ingest --dataset <name> --output-dir /work/raw
matador booleanize --config examples/paper_reproduction/<name>_booleanisation_config.yaml
matador train --config examples/paper_reproduction/<name>_training_config.yaml
```

`gesture_phase`/`emg`/`sports` (unverified-recipe datasets) still need
`matador ingest --dataset <name>` run first per usual — the booleanisation
config here replaces `--show-recipe`'s skeleton, it doesn't replace ingest.

Some of these (`emg` especially, at ~4.2M raw samples; `human_activity` and
`sensorless_drive` at hundreds of epochs × hundreds of clauses) are slow to
train at the paper's full epoch counts — that's realistic, not a bug; budget
real time or reduce `epochs`/`clauses` for a quick smoke test.
