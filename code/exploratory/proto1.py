"""Exploratory prototype kept for transparency (not used for the paper tables)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd, time, sys
from sklearn.ensemble import HistGradientBoostingRegressor as H
from data import *
d=add_causal_features(load_all())
ref=d[d.cycle<5].groupby('cell').soh.median(); d['soh_ref']=ref.reindex(d.cell).values; d['dsoh']=d.soh-d.soh_ref
F=FEATURE_SETS['causal']
rng=np.random.default_rng(0)
def fit(df):
    df=df.sample(min(len(df),60000),random_state=0)
    return H(max_iter=250,learning_rate=0.06,random_state=0).fit(df[F],df.dsoh)
alpha=0.1
doms=sorted(d.domain.unique()); out=[]
for T in doms:
    t0=time.time()
    S=d[d.domain!=T]; te=d[(d.domain==T)&(d.cycle>=5)]
    sdoms=[k for k in doms if k!=T]
    # naive split CP: random 20% of source cells for calibration
    cells=S.cell.unique(); rng.shuffle(cells); cal=set(cells[:len(cells)//5])
    m=fit(S[~S.cell.isin(cal)]); C=S[S.cell.isin(cal)&(S.cycle>=5)]
    r=np.abs(C.dsoh-m.predict(C[F])); qn=np.quantile(r,1-alpha)
    # cross-domain scores: inner LODO
    inner=[];preds=[]
    for k in sdoms:
        mk=fit(S[S.domain!=k]); Dk=S[(S.domain==k)&(S.cycle>=5)]
        rk=np.abs(Dk.dsoh-mk.predict(Dk[F])); inner.append(pd.DataFrame({'r':rk,'dom':k,'cell':Dk.cell}))
        preds.append(mk.predict(te[F]))
    R=pd.concat(inner)
    # domain-balanced pooled quantile: each domain weight 1, each cell within domain equal, each cycle within cell equal
    nc=R.groupby('dom').cell.transform('nunique'); ncy=R.groupby('cell').r.transform('size')
    w=1/(nc*ncy); o=np.argsort(R.r.values); cw=np.cumsum(w.values[o])/w.sum()
    qx=R.r.values[o][np.searchsorted(cw,1-alpha)]
    mf=fit(S); pf=mf.predict(te[F]); pe=np.mean(preds,0)
    y=te.dsoh.values
    def cov(p,q): return np.mean(np.abs(y-p)<=q)
    percell=lambda p,q: te.assign(c=(np.abs(y-p)<=q)).groupby('cell').c.mean()
    out.append(dict(dom=T,rmse_full=np.sqrt(np.mean((y-pf)**2))*100,rmse_ens=np.sqrt(np.mean((y-pe)**2))*100,
        cov_naive=cov(pf,qn),w_naive=2*qn*100,cov_xd=cov(pf,qx),w_xd=2*qx*100,
        cellcov_naive_p10=percell(pf,qn).quantile(.1),cellcov_xd_p10=percell(pf,qx).quantile(.1),sec=time.time()-t0))
    print(out[-1],flush=True)
r=pd.DataFrame(out); print(r.round(3).to_string()); print(r.mean(numeric_only=True).round(3))
r.to_csv('../results/proto1.csv',index=False)
