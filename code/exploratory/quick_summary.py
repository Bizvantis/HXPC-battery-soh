"""Exploratory prototype kept for transparency (not used for the paper tables)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
"""Quick look at evaluation results: python quick_summary.py [alpha] [L...]"""
import sys
import time

import pandas as pd

import evaluate as E

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 300)
alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
Ls = tuple(int(x) for x in sys.argv[2:]) or (100,)
t = time.time()
df = E.evaluate("lodo", alphas=(alpha,), Ls=Ls, n_jobs=4)
print(f"{time.time() - t:.0f}s, folds x seeds:", df.groupby(["fold", "seed"]).ngroups)
cols = ["cov", "width", "winkler", "cell_p10", "cell_ok", "rmse"]
print(df.groupby(["stage", "L", "method"])[cols].mean().round(3))
print("\nworst-domain coverage:")
print(df.groupby(["stage", "L", "method", "fold"])["cov"].mean().groupby(["stage", "L", "method"]).min().round(3))
