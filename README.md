# SIS Epidemic Dynamics on Temporal Multiplex Networks

**Paper:** "Epidemic Mean-Field Thresholds of SIS Dynamics on Temporal Multiplex Networks with Activity-Driven Layers"

**Authors:** Matin Marjani, Shirshendu Chatterjee, Sharmodeep Bhattacharyya, Caterina Scoglio

**Status:** Revision submitted to *Applied Network Science*

---

## Overview

We study the SIS (Susceptible–Infected–Susceptible) epidemic model on a **temporal multiplex network** composed of two layers:

- **Static backbone** — a fixed random graph (Erdős–Rényi, Watts–Strogatz, or Barabási–Albert) representing persistent contacts.
- **Activity-driven temporal layer** — at each time step, each node activates independently with probability *a* and makes *m* one-step contacts with uniformly chosen targets.

The combined early-time operator is **A + 2am·(1/N)·11ᵀ**, where **A** is the static adjacency matrix. Linearizing near the disease-free equilibrium gives a compact mean-field threshold:

> **τ_c = 1 / (λ₁ + 2am)**

where λ₁ is the spectral radius of the static layer. Simulations on N = 10,000 networks confirm the threshold on ER and WS graphs; for BA networks a finite-N correction factor C₀ ≈ 1.42 is needed.

---

## Repository Structure

```
config.py              — all parameters in one place (N, T, τ-grid, m-grid, …)
requirements.txt       — Python dependencies

networks/
  generate.py          — build or load ER / WS / BA graphs, compute λ₁
  graphs/              — pre-generated adjlists (N = 10 000, seed 42)
    ER-10000.adjlist
    WS-10000.adjlist
    BA-10000.adjlist

simulation/
  engine.py            — discrete-time SIS engine (vectorised NumPy)

analysis/
  metrics.py           — rho_naive, rho_surviving, extinction_prob, r_sim, r_th

plotting/
  figures.py           — all figure-generation functions used by the paper

scripts/
  run_sweep.py         — main (m, τ) grid sweep for one network family
  run_ba_zoom.py       — extended m-sweep for BA at fixed τ (m = 0 … 60)
  run_ba_e4.py         — early-time growth-rate sweep for BA (exploratory)
  make_figures.py      — read CSVs and produce all paper figures
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

Python 3.10+ recommended. No GPU required.

### 2. Run the (m, τ) sweep for each network

Run from the **repository root**:

```bash
python scripts/run_sweep.py --network ER
python scripts/run_sweep.py --network WS
python scripts/run_sweep.py --network BA
```

Each command writes `data/<NETWORK>_results.csv`. The sweep is parallelised across CPU cores and supports **resuming** — if you stop mid-run, re-running the same command picks up where it left off.

Use `--workers N` to control parallelism (default: half of CPU count).  
Use `--test` for a quick smoke-test (~30 s).

### 3. Extended BA sweep (for the m-sweep figures)

The standard BA grid only goes to m = 15. For Figure 5 (BA prevalence vs m at τ = 0.060) you need the zoom data out to m = 60:

```bash
python scripts/run_ba_zoom.py
```

This writes `data/BA_zoom.csv` and also reuses existing BA data at τ = 0.060.

### 4. Generate all figures

```bash
python scripts/make_figures.py
```

Figures are saved to `figures/<NETWORK>/`. Requires the CSVs from steps 2–3.

---

## Simulation Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| N | 10 000 | Network size |
| ⟨k⟩ | 4 | Target average degree |
| a | 0.1 | Node activation probability |
| μ | 1.0 | Recovery probability per step |
| T | 500 | Total simulation steps |
| T_burn | 250 | Burn-in steps (discarded) |
| R | 100 | Replicates per (m, τ) point |
| I₀ | 10% | Initial infected fraction |

All parameters are centralised in `config.py`.

---

## Network Parameters

| Network | λ₁ | τ_c(m=0) |
|---------|----|----------|
| Erdős–Rényi | 5.22 | 0.192 |
| Watts–Strogatz | 4.21 | 0.237 |
| Barabási–Albert | 16.36 | 0.061 (theory) / 0.087 (empirical, finite-N) |

Pre-generated adjlists (N = 10 000, seed 42) are included in `networks/graphs/` so simulations are exactly reproducible without re-running network generation.

---

## Output Columns (results CSV)

| Column | Description |
|--------|-------------|
| `m`, `tau` | Sweep parameters |
| `rho_naive` | Mean prevalence averaged over all replicates |
| `rho_surviving` | Mean prevalence over surviving replicates only |
| `extinction_prob` | Fraction of replicates that went extinct |
| `r_sim` | Empirical early-time growth rate |
| `r_th` | Theoretical growth rate μ(τ(λ₁ + 2am) − 1) |
| `tau_c_mf` | Mean-field threshold 1/(λ₁ + 2am) |
| `lambda1` | Spectral radius of the static layer |

---

## Citation

If you use this code, please cite:

```
Marjani, M., Chatterjee, S., Bhattacharyya, S., & Scoglio, C.
"Epidemic Mean-Field Thresholds of SIS Dynamics on Temporal Multiplex Networks
with Activity-Driven Layers."
Applied Network Science (under review).
```
