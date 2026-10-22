"""Decision-level evaluation: maintenance alarm when the lower interval bound of SOH falls below a
threshold (default 85% of nominal). Missed alarm: true SOH below threshold but no alarm (unsafe).
False alarm: alarm although true SOH is above threshold (costly). alpha = 0.1, L = 100."""
import glob
import os

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

import evaluate as E

BB, ALPHA, L = "hdse", 0.1, 100
RES = E.RES


def fold(name, seed, thresholds=(0.85, 0.9)):
    tg, inner0, bbs = E.load_fold("lodo", name, seed)
    mu, sd, inner = bbs[BB]
    d0b = E.day0_baselines(tg, ALPHA)
    d0p, beta = E.day0_proposed(mu, sd, inner, ALPHA)
    insb = E.inservice_baselines(tg, inner0, ALPHA, L)
    insp, _ = E.inservice_proposed(tg, mu, sd, inner, ALPHA, L, beta)
    meth = {"Split CP": (d0b["split"], False), "CQR": (d0b["cqr"], False), "GP": (d0b["gp"], False),
            "HX-CP": (d0p["hxcp"], False), "point (no UQ)": ((tg["mu_split"], tg["mu_split"], tg["mu_split"]), False),
            "ACI": (insb["aci_split"], True), "Persist.+online": (insb["persist_online"], True),
            "HX-CP+": (insp["hxcp+"], True)}
    ev_ins = ~E.checkup_mask(tg, L)
    soh = tg["soh"].values
    ref = tg["soh_ref"].values
    rows = []
    for m, ((lo, hi, c), ins) in meth.items():
        lo_soh = ref + np.asarray(lo)
        mask = ev_ins if ins else np.ones(len(tg), bool)
        for th in thresholds:
            below = (soh < th) & mask
            above = (soh >= th) & mask
            alarm = lo_soh < th
            # per-cell detection delay: first persistent alarm (3 consecutive evaluated cycles) minus
            # first true crossing (3 consecutive cycles below threshold), in cycles
            delays, early = [], []
            for _, idx in tg.groupby("cell", sort=False).indices.items():
                idx = idx[np.argsort(tg["cycle"].values[idx])]
                idx = idx[mask[idx]]
                if len(idx) < 5:
                    continue
                cyc = tg["cycle"].values[idx]
                b, a = soh[idx] < th, alarm[idx]
                run = lambda v: np.where(np.convolve(v.astype(int), np.ones(3, int), "valid") == 3)[0]
                rb, ra = run(b), run(a)
                if len(rb) == 0:
                    continue
                t_cross = cyc[rb[0]]
                t_alarm = cyc[ra[0]] if len(ra) else cyc[-1] + 1
                delays.append(t_alarm - t_cross)
                early.append(t_alarm < t_cross - 100)
            d = np.array(delays)
            rows.append(dict(fold=name, seed=int(seed), method=m, threshold=th,
                             miss=np.nan if below.sum() == 0 else np.mean(~alarm[below]),
                             false_alarm=np.nan if above.sum() == 0 else np.mean(alarm[above]),
                             on_time=np.nan if len(d) == 0 else np.mean(d <= 0),
                             within_L=np.nan if len(d) == 0 else np.mean(d <= L),
                             median_delay=np.nan if len(d) == 0 else np.median(d),
                             early_100=np.nan if len(d) == 0 else np.mean(early)))
    return rows


def main():
    jobs = [os.path.basename(f)[:-4].rsplit("_s", 1) for f in sorted(glob.glob(os.path.join(RES, "offline", "lodo", "*.pkl")))]
    out = Parallel(n_jobs=4)(delayed(fold)(n, s) for n, s in jobs)
    df = pd.DataFrame([r for rr in out for r in rr])
    df.to_csv(os.path.join(RES, "decision.csv"), index=False)
    g = df.groupby(["threshold", "method", "fold"]).mean(numeric_only=True).groupby(["threshold", "method"])[
        ["miss", "false_alarm", "on_time", "within_L", "median_delay", "early_100"]]
    pd.set_option("display.width", 200)
    print(g.mean().round(3))


if __name__ == "__main__":
    main()
