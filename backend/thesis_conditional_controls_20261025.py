"""Development-selected strong controls for conditional novelty."""
from collections import Counter
import json,itertools
import numpy as np
from backend.thesis_conditional_models_20261025 import OUT,SEED,views,read,write,sha,method_profiles
from backend import thesis_adaptive_policy_20261024 as common

def rank_features(e):
    x=e['X'];return np.column_stack([x.mean(1),x.std(1),x.max(1),x.min(1),e['C'],e['oldM'],e['residual'],e['maxLex'],e['maxDense']])

def support_values(e,weights):
    s=e['components']@np.asarray(weights);old=e['ownedComponents']@np.asarray(weights);prior=old.max(0) if e['w'] else np.zeros(s.shape[1])
    return s.mean(1),np.maximum(s-prior,0).mean(1),(s*(1-prior)).mean(1),(s*np.prod(1-old,axis=0)).mean(1)

def develop():
    import joblib
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor,HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    if (OUT/'control-selection.json').exists() or (OUT/'test-pair-features.npz').exists():raise RuntimeError('Controls sealed')
    write(OUT/'control-plan.json',dict(supportMixtures=35,rawKernelPolicies=63,
        ranker='Ridge alpha1/10/100; HGB160trees,leaf7/15,L2=10/100,min50,lr.04. Same candidate+owned inputs, unconstrained regression of observed added links per context. Uniform parent weights.',
        unconstrained='Refit the selected conditional model architecture without monotone constraints; same alpha, no further tuning.'))
    tr=views('training');dv=views('development');profiles={};search=[]
    def record(name,config,ss):
        q=float(common.parent_quality(ss,dv).mean());row=dict(family=name,config=config,developmentQuality=q);search.append(row)
        if name not in profiles or q>profiles[name]['developmentQuality']+1e-12:profiles[name]=row
    for raw in itertools.product(range(5),repeat=4):
        if sum(raw)!=4:continue
        weights=np.asarray(raw)/4;z=[support_values(e,weights) for e in dv]
        for name,j in [('development-selected-C',0),('residual-max',2),('residual-product',3)]:record(name,dict(weights=weights.tolist()),[v[j] for v in z])
        for name,key in [('char-novelty','maxLex'),('dense-novelty','maxDense')]:record(name,dict(weights=weights.tolist()),[v[0]*(1-e[key]) for v,e in zip(z,dv)])
    for k in range(9):
        zz=[]
        for e in dv:
            c=e['rawKernels'][:,:,k];w=e['oldRawKernels'][:,:,k];prior=w.max(0) if e['w'] else np.zeros(c.shape[1]);zz.append((c.mean(1),np.maximum(c-prior,0).mean(1)))
        for alpha in [0,.1,.3,.5,2/3,.8,1]:record('old-CM',dict(kernel=k,alpha=alpha),[alpha*c+(1-alpha)*m for c,m in zz])
    counts=Counter(e['parent'] for e in tr);tx=np.concatenate([rank_features(e) for e in tr]);ty=np.concatenate([e['gain']/len(e['q']) for e in tr]);weight=np.concatenate([np.full(len(e['c']),1/(counts[e['parent']]*len(e['c']))) for e in tr]);weight*=len(weight)/weight.sum();dx=np.concatenate([rank_features(e) for e in dv]);ends=np.cumsum([len(e['c']) for e in dv])[:-1]
    for reg in [1,10,100]:
        m=make_pipeline(StandardScaler(),Ridge(alpha=reg));m.fit(tx,ty,ridge__sample_weight=weight);path=f'ranker-ridge-{reg}.joblib';joblib.dump(m,OUT/path);record('learned-ranker',dict(path=path,sha256=sha(OUT/path)),np.split(m.predict(dx),ends))
    for leaves in [7,15]:
        for reg in [10,100]:
            m=HistGradientBoostingRegressor(max_iter=160,max_leaf_nodes=leaves,min_samples_leaf=50,l2_regularization=reg,learning_rate=.04,early_stopping=False,random_state=SEED)
            m.fit(tx,ty,sample_weight=weight);path=f'ranker-leaf{leaves}-reg{reg}.joblib';joblib.dump(m,OUT/path);record('learned-ranker',dict(path=path,sha256=sha(OUT/path)),np.split(m.predict(dx),ends))
    selected=method_profiles()['RAG'];xx=[];yy=[];ww=[];counts=Counter(e['parent'] for e in tr if e['w'] and e['positive'].sum())
    for e in tr:
        if not e['w']:continue
        mask=e['positive'].astype(bool);n=int(mask.sum())
        if not n:continue
        xx.append(e['X'][mask]);yy.extend(np.broadcast_to(e['novel'],e['positive'].shape)[mask]);ww.extend([1/(counts[e['parent']]*n)]*n)
    w=np.asarray(ww);w*=len(w)/w.sum();m=HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=selected['leaves'],min_samples_leaf=30,l2_regularization=selected['reg'],learning_rate=.04,early_stopping=False,random_state=SEED)
    m.fit(np.concatenate(xx),np.asarray(yy),sample_weight=w);joblib.dump(m,OUT/'unconstrained-novelty.joblib')
    write(OUT/'control-selection.json',dict(profiles=profiles,search=search,unconstrainedModelSha256=sha(OUT/'unconstrained-novelty.joblib'),codeSha256=sha(__file__),testOpened=False))
    print('CONTROLS',json.dumps(profiles),flush=True)

if __name__=='__main__':develop()
