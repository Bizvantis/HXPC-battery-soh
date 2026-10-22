"""Data loading for the four public Li-ion ageing datasets (XJTU, TJU, HUST, MIT).

Uses the per-cycle charging-segment features released with Wang et al. (Nat. Commun. 2024),
but WITHOUT their per-cell whole-life min-max normalisation and 3-sigma filtering, both of
which use statistics of a test cell's entire (future) trajectory. Here every transformation
is causal: it only uses information available at commissioning (first K cycles) or from
training cells.
"""
import glob
import os

import numpy as np
import pandas as pd

ROOT = os.path.join(os.path.dirname(__file__), "..", "data", "PINN4SOH", "data")

FEATURES = [
    "voltage mean", "voltage std", "voltage kurtosis", "voltage skewness", "CC Q",
    "CC charge time", "voltage slope", "voltage entropy", "current mean", "current std",
    "current kurtosis", "current skewness", "CV Q", "CV charge time", "current slope",
    "current entropy",
]

# nominal capacities (Ah) used to define SOH, as in the dataset papers
NOMINAL = {"XJTU": 2.0, "HUST": 1.1, "MIT": 1.1,
           "TJU-NCA": 3.5, "TJU-NCM": 3.5, "TJU-NCM+NCA": 2.5}

# descriptive metadata per domain (chemistry / protocol) for the paper
DOMAIN_INFO = {
    "XJTU-2C": ("XJTU", "NCM", "2C constant discharge"),
    "XJTU-3C": ("XJTU", "NCM", "3C constant discharge"),
    "XJTU-R2.5": ("XJTU", "NCM", "random 2.5 A discharge"),
    "XJTU-R3": ("XJTU", "NCM", "random 3 A discharge"),
    "XJTU-RW": ("XJTU", "NCM", "random walk discharge"),
    "XJTU-Sim": ("XJTU", "NCM", "satellite (simulated orbit) profile"),
    "TJU-NCA": ("TJU", "NCA", "varied C-rate / temperature"),
    "TJU-NCM": ("TJU", "NCM", "varied C-rate / temperature"),
    "TJU-NCM+NCA": ("TJU", "NCM+NCA", "varied C-rate / temperature"),
    "MIT-b1": ("MIT", "LFP", "multi-step fast charging (batch 2017-05-12)"),
    "MIT-b2": ("MIT", "LFP", "multi-step fast charging (batch 2017-06-30)"),
    "MIT-b3": ("MIT", "LFP", "multi-step fast charging (batch 2018-04-12)"),
    "HUST": ("HUST", "LFP", "77 multi-stage discharge protocols"),
}


def _domain_of(dataset, path):
    name = os.path.basename(path)
    parent = os.path.basename(os.path.dirname(path))
    if dataset == "XJTU":
        b = name.split("_")[0]
        return "XJTU-" + {"satellite": "Sim"}.get(b, b)
    if dataset == "TJU":
        return {"Dataset_1_NCA_battery": "TJU-NCA", "Dataset_2_NCM_battery": "TJU-NCM",
                "Dataset_3_NCM_NCA_battery": "TJU-NCM+NCA"}[parent]
    if dataset == "MIT":
        return {"2017-05-12": "MIT-b1", "2017-06-30": "MIT-b2", "2018-04-12": "MIT-b3"}[parent]
    return "HUST"


def load_all(cache=os.path.join(os.path.dirname(__file__), "..", "results", "all_cells.parquet")):
    """Return a long DataFrame: one row per (cell, cycle) with raw features, SOH and ids."""
    if os.path.exists(cache.replace(".parquet", ".pkl")):
        return pd.read_pickle(cache.replace(".parquet", ".pkl"))
    frames = []
    for dataset in ["XJTU", "TJU", "HUST", "MIT"]:
        files = sorted(glob.glob(os.path.join(ROOT, f"{dataset} data", "**", "*.csv"), recursive=True))
        for f in files:
            dom = _domain_of(dataset, f)
            nom = NOMINAL[dom] if dom in NOMINAL else NOMINAL[dataset]
            df = pd.read_csv(f)
            df = df.replace([np.inf, -np.inf], np.nan).dropna().reset_index(drop=True)
            out = df[FEATURES].astype(float).copy()
            out["soh"] = df["capacity"].values / nom
            out["cycle"] = np.arange(len(df))
            out["life_frac"] = out["cycle"] / max(len(df) - 1, 1)  # analysis only, never a feature
            out["cell"] = dom + "/" + os.path.splitext(os.path.basename(f))[0]
            out["domain"] = dom
            out["dataset"] = dataset
            out["nominal"] = nom
            frames.append(out)
    data = pd.concat(frames, ignore_index=True)
    data.to_pickle(cache.replace(".parquet", ".pkl"))
    return data


def add_causal_features(data, k_ref=5):
    """Commissioning-referenced features.

    For every cell, the reference vector is the median of its first `k_ref` cycles (known at
    commissioning). We provide (i) raw features, (ii) feature deltas w.r.t. the reference,
    and (iii) capacity-like features expressed per nominal Ah (nominal capacity is on the
    datasheet). No statistic of future cycles is used.
    """
    data = data.copy()
    ref = (data[data["cycle"] < k_ref].groupby("cell")[FEATURES].median())
    ref = ref.reindex(data["cell"]).values
    raw = data[FEATURES].values
    for j, f in enumerate(FEATURES):
        data["d_" + f] = raw[:, j] - ref[:, j]
    for f in ["CC Q", "CV Q"]:
        data["n_" + f] = data[f] / data["nominal"]
        data["dn_" + f] = data["d_" + f] / data["nominal"]
    # relative change of time-like features (protocol-invariant scale)
    for f in ["CC charge time", "CV charge time"]:
        j = FEATURES.index(f)
        data["r_" + f] = raw[:, j] / np.maximum(np.abs(ref[:, j]), 1e-6) - 1.0
    return data


FEATURE_SETS = {
    "raw": FEATURES,
    "causal": FEATURES + ["d_" + f for f in FEATURES]
              + ["n_CC Q", "n_CV Q", "dn_CC Q", "dn_CV Q", "r_CC charge time", "r_CV charge time"],
}


def wang_leaky_features(data):
    """Wang et al. (2024) preprocessing: per-cell whole-life min-max scaling to [-1, 1].

    Used ONLY to quantify the optimism induced by look-ahead normalisation.
    """
    data = data.copy()
    g = data.groupby("cell")[FEATURES]
    mn, mx = g.transform("min"), g.transform("max")
    data[FEATURES] = 2 * (data[FEATURES] - mn) / (mx - mn).replace(0, 1) - 1
    return data


if __name__ == "__main__":
    d = load_all()
    print(d.shape, d["cell"].nunique(), "cells")
    print(d.groupby("domain").agg(cells=("cell", "nunique"), rows=("soh", "size"),
                                  soh_min=("soh", "min"), soh_max=("soh", "max")))
