<img src="/images/banner.png" width=900/>

# Matador — Automated RTL Accelerator Generator for Tsetlin Machines

Matador trains Tsetlin Machines and generates synthesisable Verilog RTL for FPGA deployment.
It produces a complete hardware design from a trained model in one command, with built-in
simulation, emulation, and reproducibility tooling — and, upstream of training, tooling to
fetch/booleanize raw data in the first place.

```bash
matador ingest      --dataset digits --output-dir /work/raw   # optional: fetch a registered dataset -> npz
matador booleanize  --dataset digits --raw-dir /work/raw      # optional: npz -> Boolean train/test
matador train        --config /work/training_config.yaml
matador generate     --backend vanilla_tiled --config /work/vanilla_tiled.yaml
matador simulate     --backend vanilla_tiled --config /work/vanilla_tiled.yaml
```

Own or unregistered data instead? `matador ingest --config data_source_config.yaml` /
`matador booleanize --config booleanisation_config.yaml` take the same shape as a
registered dataset, just described by you — see [docs/Usage.md](docs/Usage.md).

Not sure where to start? `matador faena` is an interactive wizard that walks through
whichever of the above your workspace doesn't have yet.

---

## Everyday commands

```bash
matador                    # splash + workspace dashboard (interactive terminal only)
matador status              # same dashboard, non-interactive (scripts/CI)
matador registry            # browse every registered dataset + accelerator backend
matador faena                # interactive wizard -- walks through whatever's missing
matador clean --dry-run      # preview a full workspace reset (nothing removed without confirmation)
```

---

## Documentation

| Document | Contents |
|---|---|
| [docs/Setup.md](docs/Setup.md) | Docker setup, building the container, first run |
| [docs/Usage.md](docs/Usage.md) | Full ingest → booleanize → train → generate → simulate workflow with YAML examples |
| [docs/GeneratedOutputs.md](docs/GeneratedOutputs.md) | A map of `/work` — what every command writes, what each file is for, what's safe to delete |
| [docs/Developer.md](docs/Developer.md) | Extending Matador — adding a new dataset, booleanisation technique, or accelerator backend |

---

## Extending Matador

Matador is designed as a set of plugin registries, not a fixed pipeline — you don't need
to touch any Python to add to it. Register your own dataset by adding an entry to
[`data/Raw_Data_Bank.yaml`](data/Raw_Data_Bank.yaml) (a plain YAML file: where to fetch
the raw data from and how to parse it — `matador ingest --dataset <key>` picks it up
automatically once it's there). The same self-contained-plugin shape applies to adding a
new Boolean encoding technique or a new accelerator backend. `docs/Developer.md` has a
worked example for each:

- [Adding a new dataset](docs/Developer.md#adding-a-new-dataset)
- [Adding a new booleanisation technique](docs/Developer.md#adding-a-new-booleanisation-technique)
- [Adding a new accelerator backend](docs/Developer.md#adding-a-new-accelerator-backend)

---

## Registered datasets

`matador ingest --dataset <key>` / `matador booleanize --dataset <key>` — see `matador
registry` (or `matador list-datasets`) for the live, authoritative list (source:
`data/Raw_Data_Bank.yaml`).

| Dataset | Description | Verified booleanization recipe |
|---|---|---|
| `digits` | Optical Recognition of Handwritten Digits (optdigits) | yes |
| `sports` | Daily and Sports Activities (DSA) | no (`--show-recipe` only) |
| `statlog` | Statlog (Vehicle Silhouettes) | yes |
| `gesture_phase` | Gesture Phase Segmentation | no (`--show-recipe` only) |
| `human_activity` | Human Activity Recognition Using Smartphones (UCI HAR) | no (`--show-recipe` only) |
| `mammographic` | Mammographic Mass | yes |
| `emg` | EMG Data for Gestures | no (`--show-recipe` only) |
| `sensorless_drive` | Dataset for Sensorless Drive Diagnosis | yes |
| `mnist` | MNIST handwritten digits | yes |

See [docs/Developer.md](docs/Developer.md#adding-a-new-dataset) to add another one.

---

## Registered backends

| Backend | Architecture |
|---|---|
| `vanilla_tiled` | Vanilla TM — sequential FSM + tile ROM. Knobs: `feat_slice`, `clause_slice`. |
| `vanilla_hardwired` | Vanilla TM — HCB streaming + adder tree. Knobs: `pipeline_stages`. |
| `vanilla_gp_tiled` | Vanilla TM — runtime-reprogrammable tiled core (vendored from GP_TM_Inference_Accelerator). One synthesis, reprogrammable at runtime via AXI-Stream `CMD_LOAD` — no resynthesis to swap models. Capacity is checked against a target FPGA's BRAM budget. Knobs: `target_fpga`, `feat_slice`, `clause_slice`, `max_features`, `max_clauses_total`, `max_classes`. Multi-model/dataset reprogramming can be proven end to end with `matador reprogram-suite`. |

See [docs/Developer.md](docs/Developer.md#adding-a-new-accelerator-backend) to add another one.

---

## Background reading

| | Paper |
|---|---|
| 1 | [Tsetlin Machine](https://arxiv.org/abs/1804.01508) |
| 2 | [Coalesced Tsetlin Machine](https://arxiv.org/abs/2108.07594) |
| 3 | [Hardware Tsetlin Machines](https://royalsocietypublishing.org/doi/epdf/10.1098/rsta.2019.0593) |
| 4 | [Matador paper](https://arxiv.org/abs/2403.10538) |

---

## License

Apache License 2.0 — see [LICENSE](LICENSE). You're free to use, modify, and
redistribute this code (including commercially); modified files must carry a
notice stating what you changed (License §4(b)). Third-party components
(`tmu/`, `matador/backends/gp_tiled/vendor/`) retain their own licenses/
copyright — see [NOTICE](NOTICE).

---

## Contact

Developed by Tousif Rahman, Gang Mao, Bob Pattison, Sidharth Maheshwari, Marcos Sartori and Han Wu Microsystems Group, Newcastle University.  
Issues: tousif.rahman@newcastle.ac.uk
