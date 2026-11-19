"""Convex model averaging of the same bounded additional-link estimand.

Weights are selected only on development, before the reserved31 predictions.
Every component is candidate-wise, bounded by its own C, and monotone inW.
Nonnegative weights preserve those properties. No ranking normalization uses
other purchase candidates. Earlier unsuccessful final studies remain intact.
"""
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
import itertools,json,argparse
import numpy as np
from backend.thesis_neural_pair_20261025 import OUT,SEED
from backend.thesis_study_20261023 import read,write,sha
from backend import thesis_adaptive_policy_20261024 as metric
from backend import thesis_conditional_models_20261025 as old
from backend import thesis_conditional_controls_20261025 as control
from backend import thesis_neural_selection_20261025 as neural
old.OUT=OUT;control.OUT=OUT

def mix(C,M,w,alpha):
    cc=[sum(weight*C[j][i] for j,weight in enumerate(w)) for i in range(len(C[0]))]
    mm=[sum(weight*M[j][i] for j,weight in enumerate(w)) for i in range(len(M[0]))]
    return cc,mm,[alpha*c+(1-alpha)*m for c,m in zip(cc,mm)]

def parts(stage,path='RAG'):
    import joblib
    es=old.views(stage);npth='RAG' if path=='RAG' else 'no-runtime-retrieval';ns,nbase=neural.neural_views(stage,npth)
    assert len(es)==len(ns) and all((a['parent'],a['view'])==(b['parent'],b['view']) and np.array_equal(a['gain'],b['gain']) for a,b in zip(es,ns))
    p=read(OUT/'method-selection.json')['profiles']['RAG' if path=='RAG' else 'no-RAG'];nn=read(OUT/'neural-method-selection.json')['profiles'][npth];key='X' if path=='RAG' else 'noRAGX';basekey='base' if path=='RAG' else 'noRAGBase';ckey='C' if path=='RAG' else 'noRAGC'
    model=joblib.load(OUT/p['model']);x=np.concatenate([e[key].reshape(-1,e[key].shape[-1]) for e in es]);ends=np.cumsum([e[key].shape[0]*e[key].shape[1] for e in es])[:-1];pr=np.split(model.predict_proba(x)[:,1],ends);co=[e[ckey] for e in es];mo=[(e[basekey]*(v.reshape(e[basekey].shape) if e['w'] else 1)).mean(1) for e,v in zip(es,pr)]
    model=joblib.load(OUT/nn['model']);x=np.concatenate([e['X'].reshape(-1,e['X'].shape[-1]) for e in ns]);pr=np.split(model.predict_proba(x)[:,1],ends);bases=[e['components']@np.asarray(nn['weights']) for e in ns];cn=[b.mean(1) for b in bases];mn=[(b*(v.reshape(b.shape) if e['w'] else 1)).mean(1) for b,v,e in zip(bases,pr,ns)]
    controls=read(OUT/'control-selection.json')
    if path=='RAG':res=controls['profiles']['residual-product']
    else:res=max([r for r in controls['search'] if r['family']=='residual-product' and sum(r['config']['weights'][:2])==0],key=lambda r:r['developmentQuality'])
    z=[control.support_values(e,res['config']['weights']) for e in es];cr=[v[0] for v in z];mr=[v[3] for v in z]
    return es,[co,cn,cr],[mo,mn,mr],dict(original=p,neural=nn,residual=res)

def rankx(e,weights):
    base=e['components']@np.asarray(weights);own=e['ownedComponents']@np.asarray(weights);prior=own.max(0) if len(own) else np.zeros(base.shape[1]);x=e['X']
    return np.column_stack([x.mean(1),x.std(1),x.max(1),x.min(1),base.mean(1),np.maximum(base-prior,0).mean(1),(base*(1-prior)).mean(1),x[:,0,-5],x[:,0,-4]])

def develop():
    import joblib
    from threadpoolctl import threadpool_limits
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    threadpool_limits(1)
    if (OUT/'ensemble-selection.json').exists():raise RuntimeError('Ensemble development already sealed')
    assert not (OUT/'test-pair-features.npz').exists()
    plan=dict(createdAt=datetime.now(timezone.utc).isoformat(),
        components=['Original-view conditional estimator','Owned-subset-augmented conditional estimator with reference-trained neural features','Probabilistic residual-product estimator'],
        weights='Quarter-simplex onthree nonnegative weights summingtoone:15mixtures;alpha0,.1,.3,.5,2/3,1;90policies perRAG/noRuntimeRetrievalpath. Same rawestimandG/Q andcandidate-wiseinputs. No per-list standardization.',
        selection='Maximize development225parentNG under nonnegative rich/ownedrich differences vsidenticalensembleC andalpha<1. Ties: fewer activecomponents,smalleralpha,smaller neuralweight,larger residualweight. No testretuning. This adds a finite model-averaging family before the31 reservedoutcomes, not a reinterpretation of earlier failedtests.',
        strongestC='Old35mixtures;all32neuralcalibratedcheckpointCcontrols;all15componentCmixtures foreachRAG/noRuntimeRetrievalpath. SelectbydevelopmentNG. SameC useschosenensembleweights.',
        neuralRanker='Strong717-feature control:mean/std/max/min of178pairfeatures plusC,D,residual,maxLex,maxDense;Ridge1/10/100 orHGB160leaf7/15L2=10/100,min50,lr.04;development-select best. Uniformparentweights. Original701-feature control also retained.',
        primaryContrasts=['same-support-C','development-selected-C','old-CM'],adoption=read(OUT/'protocol.json')['adoption'],
        protocolSha256=sha(OUT/'protocol.json'),codeSha256=sha(__file__))
    write(OUT/'ensemble-plan.json',plan);weights=[(np.asarray(w)/4).tolist() for w in itertools.product(range(5),repeat=3) if sum(w)==4];profiles={};audit=[];cache={};strong=[]
    for path in ['RAG','no-runtime-retrieval']:
        es,C,M,meta=parts('development','RAG' if path=='RAG' else 'no-RAG');cache[path]=(es,C,M,meta);best=None
        for w in weights:
            for alpha in [0,.1,.3,.5,2/3,1]:
                cc,mm,s=mix(C,M,w,alpha);q=float(metric.parent_quality(s,es).mean());diff={v:float(np.mean([metric.quality(ss,e['gain'])-metric.quality(c,e['gain']) for ss,c,e in zip(s,cc,es) if e['view']==v])) for v in ['rich','owned-rich']};eligible=alpha<1 and min(diff.values())>=-1e-12;r=dict(path=path,weights=w,alpha=alpha,developmentQuality=q,richDifferences=diff,eligible=eligible);audit.append(r)
                if eligible and (best is None or (round(q,12),-sum(x>0 for x in w),-alpha,-w[1],w[2])>(round(best['developmentQuality'],12),-sum(x>0 for x in best['weights']),-best['alpha'],-best['weights'][1],best['weights'][2])):best=r
            cc,_,_=mix(C,M,w,1);strong.append(dict(kind='component-C',path=path,weights=w,developmentQuality=float(metric.parent_quality(cc,es).mean())))
        if best is None:raise RuntimeError('No development-safe ensemble: '+path)
        profiles[path]=best;print('ENSEMBLE',json.dumps(best),flush=True)
    es=cache['RAG'][0];controls=read(OUT/'control-selection.json');p=controls['profiles']['development-selected-C'];strong.append(dict(kind='old-C',weights=p['config']['weights'],developmentQuality=p['developmentQuality']))
    allnew=np.load(OUT/'neural-development-calibrated.npz')['P']
    for j in range(allnew.shape[-2]):
        for channel in [0,1]:
            scores=[allnew[e['parent'],e['c']][:,e['q'],j,channel].mean(1) for e in es];strong.append(dict(kind='neural-C',checkpoint=j,channel=channel,developmentQuality=float(metric.parent_quality(scores,es).mean())))
    strongest=max(strong,key=lambda r:r['developmentQuality']);write(OUT/'ensemble-strong-C.json',dict(selected=strongest,search=strong))
    tr,base=neural.neural_views('training','RAG');dv,base=neural.neural_views('development','RAG');wgt=read(OUT/'neural-method-selection.json')['profiles']['RAG']['weights'];tx=np.concatenate([rankx(e,wgt) for e in tr]);ty=np.concatenate([e['gain']/len(e['q']) for e in tr]);count=Counter(e['parent'] for e in tr);tw=np.concatenate([np.full(len(e['c']),1/(count[e['parent']]*len(e['c']))) for e in tr]);tw*=len(tw)/tw.sum();dx=np.concatenate([rankx(e,wgt) for e in dv]);ends=np.cumsum([len(e['c']) for e in dv])[:-1];rankers=[]
    for reg in [1,10,100]:
        m=make_pipeline(StandardScaler(),Ridge(alpha=reg));m.fit(tx,ty,ridge__sample_weight=tw);name=f'neural-ranker-ridge{reg}.joblib';joblib.dump(m,OUT/name);q=float(metric.parent_quality(np.split(m.predict(dx),ends),dv).mean());rankers.append(dict(model=name,sha256=sha(OUT/name),developmentQuality=q))
    for leaves in [7,15]:
        for reg in [10,100]:
            m=HistGradientBoostingRegressor(max_iter=160,max_leaf_nodes=leaves,min_samples_leaf=50,l2_regularization=reg,learning_rate=.04,early_stopping=False,random_state=SEED);m.fit(tx,ty,sample_weight=tw);name=f'neural-ranker-leaf{leaves}-reg{reg}.joblib';joblib.dump(m,OUT/name);q=float(metric.parent_quality(np.split(m.predict(dx),ends),dv).mean());rankers.append(dict(model=name,sha256=sha(OUT/name),developmentQuality=q))
    nr=max(rankers,key=lambda r:r['developmentQuality']);write(OUT/'ensemble-neural-ranker.json',dict(selected=nr,search=rankers))
    write(OUT/'ensemble-selection.json',dict(profiles=profiles,search=audit,strongC=strongest,neuralRanker=nr,planSha256=sha(OUT/'ensemble-plan.json'),componentProfiles={k:v[3] for k,v in cache.items()},sourceHashes={n:sha(OUT/n) for n in ['method-selection.json','neural-method-selection.json','control-selection.json','neural-calibration-lock.json']},codeSha256=sha(__file__)))
    print('STRONG C',json.dumps(strongest),'NEURAL RANKER',json.dumps(nr),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['develop']);a=p.parse_args();globals()[a.stage]()
