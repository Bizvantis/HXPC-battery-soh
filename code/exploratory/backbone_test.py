"""Exploratory prototype kept for transparency (not used for the paper tables)."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd, time, torch, lightgbm as lgb, warnings
warnings.filterwarnings('ignore'); torch.set_num_threads(4)
from run_offline import prepare, F
from models import thin, train_mlp, predict_mlp
d=prepare()
rows=[]
for T in ['HUST','MIT-b2','TJU-NCM','XJTU-3C','TJU-NCM+NCA','XJTU-RW']:
    S=thin(d[d.domain!=T]); te=d[(d.domain==T)&(d.cycle>=5)]; y=te.dsoh.values
    r={'dom':T}
    for name,kw in [('lgb',{}),('lgb_lin',dict(linear_tree=True,linear_lambda=1.0)),('lgb_shallow',dict(num_leaves=8,n_estimators=600))]:
        t=time.time(); m=lgb.LGBMRegressor(n_estimators=kw.pop('n_estimators',300),learning_rate=0.05,num_leaves=kw.pop('num_leaves',31),subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1,n_jobs=4,random_state=0,**kw).fit(S[F],S.dsoh)
        r[name]=np.sqrt(np.mean((m.predict(te[F])-y)**2))*100; r[name+'_s']=time.time()-t
    mean,std=S[F].mean(),S[F].std().replace(0,1)
    t=time.time(); net=train_mlp(((S[F]-mean)/std).values,S.dsoh.values,0,epochs=40); mu,_=predict_mlp(net,((te[F]-mean)/std).values)
    r['mlp']=np.sqrt(np.mean((mu-y)**2))*100; r['mlp_s']=time.time()-t
    from sklearn.linear_model import Ridge
    m=Ridge(1.0).fit(((S[F]-mean)/std).values,S.dsoh); r['ridge']=np.sqrt(np.mean((m.predict(((te[F]-mean)/std).values)-y)**2))*100
    rows.append(r); print(r,flush=True)
print(pd.DataFrame(rows).round(2).to_string())
