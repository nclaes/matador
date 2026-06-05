<img src="/images/banner.png" width=900/>

# Matador — Automated RTL Accelerator Generator for Tsetlin Machines

Matador trains Tsetlin Machines and generates synthesisable Verilog RTL for FPGA deployment.
It produces a complete hardware design from a trained model in one command, with built-in
simulation, emulation, and reproducibility tooling.

```bash
matador train    --config /work/training_config.yaml
matador generate --backend vanilla_tiled --config /work/vanilla_tiled.yaml
matador simulate --backend vanilla_tiled --config /work/vanilla_tiled.yaml
```

---

## Documentation

| Document | Contents |
|---|---|
| [docs/Setup.md](docs/Setup.md) | Docker setup, building the container, first run |
| [docs/Usage.md](docs/Usage.md) | Full training → generate → simulate workflow with YAML examples |
| [docs/Developer.md](docs/Developer.md) | Plugin architecture — adding new accelerator backends |

---

## Registered backends

| Backend | Architecture |
|---|---|
| `vanilla_tiled` | Vanilla TM — sequential FSM + tile ROM. Knobs: `feat_slice`, `clause_slice`. |
| `vanilla_hardwired` | Vanilla TM — HCB streaming + adder tree. Knobs: `pipeline_stages`. |

---

## Background reading

| | Paper |
|---|---|
| 1 | [Tsetlin Machine](https://arxiv.org/abs/1804.01508) |
| 2 | [Coalesced Tsetlin Machine](https://arxiv.org/abs/2108.07594) |
| 3 | [Hardware Tsetlin Machines](https://royalsocietypublishing.org/doi/epdf/10.1098/rsta.2019.0593) |
| 4 | [Matador paper](https://arxiv.org/abs/2403.10538) |

---

## Contact

Developed by Tousif Rahman and Gang Mao, Microsystems Group, Newcastle University.  
Issues: tousifsrahman@gmail.com
