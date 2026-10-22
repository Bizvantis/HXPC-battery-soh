#!/usr/bin/env bash
# Full reproduction pipeline (CPU). Run from the repository root.
set -e
[ -d data/PINN4SOH ] || git clone --depth 1 https://github.com/wang-fujin/PINN4SOH.git data/PINN4SOH
cd code
python run_offline.py --seeds 0 1 2
python run_proposed.py --seeds 0 1 2
python run_offline.py --seeds 0 --protocol lodso
python run_proposed.py --seeds 0 --protocol lodso
python -c "import evaluate as E; E.evaluate('lodo'); E.evaluate('lodso')"
python analysis_extra.py
python analysis_more.py
python benchmark_wang.py
python leakage.py
python make_tab_data.py
python make_tab_point.py
python make_tables.py
python make_tables_extra.py
python figures.py data coverage L traj extra
