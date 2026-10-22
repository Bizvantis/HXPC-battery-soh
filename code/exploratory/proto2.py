"""Exploratory prototype kept for transparency (not used for the paper tables)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd, time, lightgbm as lgb, warnings
warnings.filterwarnings('ignore')
from data import *
d=add_causal_features(load_all())
ref=d[d.cycle<5].groupby('cell').soh.median(); d['soh_ref']=ref.reindex(d.cell).values; d['dsoh']=d.soh-d.soh_ref
d=d[d.cycle>=0]
F=FEATURE_SETS['causal']; alpha=0.1; M=5
def fit(df,seed):
    df=df.groupby('cell',group_keys=False).apply(lambda g:g.iloc[::max(1,len(g)//150)]) # thin long trajectories
    return lgb.LGBMRegressor(n_estimators=300,learning_rate=0.05,num_leaves=31,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,random_state=seed,verbose=-1,n_jobs=8).fit(df[F],df.dsoh)
def ens(train,X,seed=0):
    doms=train.domain.unique(); rng=np.random.default_rng(seed); P=[]
    for m in range(M):
        keep=rng.choice(doms,size=max(2,int(round(0.7*len(doms)))),replace=False)
        P.append(fit(train[train.domain.isin(keep)],seed*100+m).predict(X))
    P=np.array(P); return P.mean(0),P.std(0)
doms=sorted(d.domain.unique()); out=[]
for T in doms:
    t0=time.time(); S=d[d.domain!=T]; te=d[(d.domain==T)&(d.cycle>=5)]; sdoms=[k for k in doms if k!=T]
    R=[]
    for k in sdoms:
        Dk=S[(S.domain==k)&(S.cycle>=5)]; mu,sd=ens(S[S.domain!=k],Dk[F])
        R.append(pd.DataFrame({'r':np.abs(Dk.dsoh.values-mu),'sd':sd,'dom':k,'cell':Dk.cell.values}))
    R=pd.concat(R); mu,sd=ens(S,te[F]); y=te.dsoh.values
    w=1/(R.groupby('dom').cell.transform('nunique')*R.groupby('cell').r.transform('size')).values
    def wq(s): o=np.argsort(s); cw=np.cumsum(w[o])/w.sum(); return s[o][np.searchsorted(cw,1-alpha)]
    res={'dom':T,'rmse':np.sqrt(np.mean((y-mu)**2))*100,'sd_tgt':sd.mean()*100,'sd_src':R.sd.mean()*100}
    for name,s,width in [('abs',R.r.values,lambda q:q+0*sd),('norm',R.r.values/(R.sd.values+1e-3),lambda q:q*(sd+1e-3))]:
        q=wq(s); hw=width(q); c=np.abs(y-mu)<=hw
        res['cov_'+name]=c.mean(); res['w_'+name]=2*hw.mean()*100
        res['p10_'+name]=pd.Series(c).groupby(te.cell.values).mean().quantile(.1)
    res['sec']=time.time()-t0; out.append(res); print({k:(round(v,3) if isinstance(v,float) else v) for k,v in res.items()},flush=True)
r=pd.DataFrame(out); print(r.round(3).to_string()); print(r.mean(numeric_only=True).round(3)); r.to_csv('../results/proto2.csv',index=False)
