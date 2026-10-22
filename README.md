# HX-CP: Distribution-free SOH intervals for unseen battery types

This repository contains the code for the paper

> **Distribution-Free Battery State-of-Health Intervals for Unseen Cell Types: Hierarchical Cross-Domain Conformal Prediction With In-Service Check-Up Adaptation** (submitted to *IEEE Transactions on Industrial Informatics*).

It covers two methods:

- **HX-CP** (day zero, no target labels). It computes cross-domain nonconformity scores by leaving whole domains out. A heterogeneous domain-subsample ensemble (LightGBM + MLP members) normalises the scores, and they are pooled along the domain → cell → cycle hierarchy.
- **HX-CP+** (in service). A shrunk anchored estimator fuses sparse capacity check-ups with model-predicted degradation. A fleet-level online quantile tracker then gives long-run coverage under arbitrary shift.

Everything runs on a CPU. The experiments in the paper ran on a laptop (Ryzen 7 7730U, 16 GB RAM, no GPU).

## 1. Data

We use the preprocessed per-cycle charging features of the XJTU, TJU, HUST and MIT datasets released with Wang et al., *Nat. Commun.* 15, 4332 (2024). The data are **not** redistributed here. Clone them next to the code:

```bash
git clone --depth 1 https://github.com/wang-fujin/PINN4SOH.git data/PINN4SOH
```

The original sources are:
- XJTU: Wang et al. (2024)
- TJU: Zhu et al., *Nat. Commun.* 13, 2261 (2022), Zenodo 6405084
- HUST: Ma et al., *Energy Environ. Sci.* 15, 4083 (2022), Mendeley nsc7hnsg4s
- MIT: Severson et al., *Nat. Energy* 4, 383 (2019), and Attia et al., *Nature* 578, 397 (2020), data.matr.io

Please cite the original dataset papers.

## 2. Environment

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Tested with Python 3.10.

## 3. Reproducing the paper

Run all commands from `code/`:

| Step | Command | Output | Approx. time (8-core laptop) |
|---|---|---|---|
| Leave-one-domain-out: baselines + GBDT backbone | `python run_offline.py --seeds 0 1 2` | `results/offline/lodo/*.pkl` | ~1.5 h / seed |
| Leave-one-domain-out: heterogeneous backbone (HDSE) | `python run_proposed.py --seeds 0 1 2` | `results/offline/lodo_hdse/*.pkl` | ~1.3 h / seed |
| Leave-one-dataset-out | `python run_offline.py --protocol lodso` and `python run_proposed.py --protocol lodso` | `results/offline/lodso*` | ~1 h |
| Calibration, online simulation, metrics | `python -c "import evaluate as E; E.evaluate('lodo'); E.evaluate('lodso')"` | `results/metrics_*.csv` | ~1 h |
| Sensitivity, trajectory certificate, life-stage analysis | `python analysis_extra.py`, `python analysis_more.py`, `python analysis_deploy.py`, `python analysis_decision.py` | `results/*.csv` | ~20 min |
| In-domain benchmark vs. Wang et al. | `python benchmark_wang.py` and `python leakage.py` | `results/benchmark_wang*.csv`, `results/leakage.csv` | ~10 min |
| Tables and figures | `python make_tables.py`, `python make_tables_extra.py`, `python make_tab_deploy.py`, `python make_tab_decision.py`, `python make_tab_data.py`, `python make_tab_point.py`, `python figures.py data coverage L traj extra` | `paper/*.tex`, `paper/figs/*` | ~2 min |

`run_all.sh` runs the whole pipeline.

## 4. Code map

| File | Content |
|---|---|
| `code/data.py` | loading, domains, causal commissioning-referenced features, and the look-ahead preprocessing of prior work (used only for the leakage analysis) |
| `code/models.py` | LightGBM, Gaussian-NLL MLP (deep ensemble, MC dropout), GP, GBDT domain-subsample ensemble |
| `code/run_proposed.py` | heterogeneous domain-subsample ensemble (HDSE) and cross-domain scores |
| `code/conformal.py` | split/weighted quantiles, hierarchical weights, trajectory threshold, online quantile tracker |
| `code/evaluate.py` | all day-zero and in-service methods, nested source-only hyper-parameter selection, metrics |
| `code/analysis_extra.py`, `code/analysis_more.py` | step-size sensitivity, trajectory certificate, life stage, adaptivity, online threshold, label budget |
| `code/analysis_deploy.py`, `code/analysis_decision.py` | noisy/irregular check-ups, per-cell vs fleet tracking, fleet size; maintenance-alarm evaluation |
| `code/run_offline.py` | fits every baseline once per fold and stores predictions |
| `code/exploratory/` | early prototypes, kept for transparency (not used for any reported number) |

## 5. License

Code: MIT (see `LICENSE`). The datasets keep their original licenses.

## 6. Citation

Citation details will be added upon publication.
