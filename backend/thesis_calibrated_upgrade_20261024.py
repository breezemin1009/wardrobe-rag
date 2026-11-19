"""Calibrate retrieved pair evidence using only training wardrobes.

Cross-fitted training probabilities prevent a second-stage ranker from seeing
in-sample compatibility predictions. The final test is inaccessible until a
selection lock exists. No candidate role, source degree or target enters X.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,time
import numpy as np
from backend.thesis_study_20261023 import read,write,sha,h
from backend.thesis_upgrade_20261024 import OUT,SEED,CONFIGS,targets,score_quality,fit_models,predict_model

def pair_features(stage):
    import joblib
    from sklearn.decomposition import PCA
    from backend.pog_refined_retrieval import Retriever
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    if stage=='test' and not (OUT/'calibrated-selection.json').exists():raise RuntimeError('Test is sealed')
    dest=OUT/f'{stage}-pair-features.npz'
    if dest.exists():raise RuntimeError('Frozen pair features exist')
    rows=read(OUT/f'{stage}-inputs.json');engine=Retriever(OUT)
    pp=OUT/'reference-pca.joblib'
    if not pp.exists():
        if stage!='training':raise RuntimeError('Reference projection must be fitted in training stage')
        refids=sorted({r[s] for r in engine.records for s in ['top','bottom']})
        pca=PCA(n_components=32,svd_solver='randomized',random_state=SEED)
        pca.fit(engine.vectors[[engine.ix[p] for p in refids]])
        joblib.dump(pca,pp)
    pca=joblib.load(pp);vec=pca.transform(engine.vectors);data=[];start=time.monotonic()
    for i,c in enumerate(rows):
        qs=[engine.ix[p] for p in c['contexts']];cs=[engine.ix[p] for p in c['candidates']+c['owned']]
        matrices=[engine.matrix(c['contexts'],c['candidates']+c['owned'],cfg) for cfg in CONFIGS]
        directlex=(engine.lex[cs]@engine.lex[qs].T).toarray();directdense=engine.vectors[cs]@engine.vectors[qs].T
        q=np.broadcast_to(vec[qs][None,:,:],(9,4,32));b=np.broadcast_to(vec[cs][:,None,:],(9,4,32))
        x=np.concatenate([np.stack(matrices+[directlex,directdense],axis=-1),q,b,np.abs(q-b),q*b],axis=-1)
        data.append(x.astype(np.float32));engine.cache.clear()
        if i%100==0 or i+1==len(rows):print('pair features',stage,i+1,'/',len(rows),'seconds',round(time.monotonic()-start,1),flush=True)
    np.savez_compressed(dest,X=np.asarray(data))
    write(OUT/f'{stage}-pair-feature-lock.json',dict(featureSha256=sha(dest),pcaSha256=sha(pp),codeSha256=sha(__file__),
        source='Nine source-case support scores, direct title similarities, and 32-dimensional reference-only PCA of both frozen text embeddings; no role, degree or observation labels.'))

def pair_targets(stage):
    return np.array([r['observed']+r['history'] for r in read(OUT/f'{stage}-targets.json')],dtype=np.uint8)

def new_pair_model(family):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    if family=='linear':return make_pipeline(StandardScaler(),LogisticRegression(C=.01,solver='lbfgs',max_iter=1000,random_state=SEED))
    return HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=7,min_samples_leaf=40,l2_regularization=10,learning_rate=.04,
                early_stopping=False,random_state=SEED)

def summaries(p):
    s=p[:,:8];old=p[:,8:].max(1);d=s-old[:,None,:];C=s.mean(2);M=np.maximum(d,0).mean(2)
    expected=(s*(1-old[:,None,:])).mean(2)
    return np.stack([C,M,expected,s.max(2),s.min(2),s.std(2),(d>0).mean(2),np.maximum(d,0).max(2),M/np.maximum(C,1e-8)],axis=-1)

def augment(stage,probabilities):
    base=np.load(OUT/f'{stage}-features.npz');X=[base['X']];meta=read(OUT/f'{stage}-features-lock.json');names=list(meta['names']);without=list(meta['withoutM'])
    B=list(np.moveaxis(base['baselines'],2,0));labels=list(CONFIGS)
    # Basic ownership similarity is a strong non-relational redundancy control.
    lex=base['X'][:,:,-2];dense=base['X'][:,:,-1]
    for family,p in probabilities.items():
        z=summaries(p);offset=len(names);X.append(z)
        keys=['C','M','expected-coverage-increment','maxSupport','minSupport','stdSupport','positiveFraction','maxIncrease','relativeM']
        names.extend(f'calibrated-{family}/{k}' for k in keys)
        without.extend(offset+j for j,k in enumerate(keys) if k not in ['M','expected-coverage-increment','positiveFraction','maxIncrease','relativeM'])
        C=z[:,:,0];M=z[:,:,1];E=z[:,:,2]
        B.append(np.stack([C,M,C*(1-lex),C*(1-dense),E],axis=-1));labels.append(dict(mode='calibrated-'+family,k=None))
    path=OUT/f'{stage}-calibrated-features.npz';np.savez_compressed(path,X=np.concatenate(X,axis=-1),baselines=np.stack(B,axis=2))
    write(OUT/f'{stage}-calibrated-feature-lock.json',dict(names=names,withoutM=without,configs=labels,featureSha256=sha(path),codeSha256=sha(__file__)))

def calibrate():
    import joblib
    from sklearn.metrics import average_precision_score,brier_score_loss,log_loss
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    if (OUT/'calibration-lock.json').exists():raise RuntimeError('Calibration already frozen')
    if (OUT/'test-pair-features.npz').exists():raise RuntimeError('Test was opened before calibration')
    tr=np.load(OUT/'training-pair-features.npz')['X'];dv=np.load(OUT/'development-pair-features.npz')['X'];y=pair_targets('training');dy=pair_targets('development')
    ids=[r['id'] for r in read(OUT/'training-inputs.json')];fold=np.array([int(h(f'{SEED}:calibration:{p}')[:12],16)%3 for p in ids])
    trainpred={};devpred={};audit={}
    for family in ['linear','boosted']:
        oof=np.zeros(y.shape,dtype=np.float64)
        for f in range(3):
            model=new_pair_model(family);mask=fold!=f;valid=~mask
            model.fit(tr[mask].reshape(-1,tr.shape[-1]),y[mask].ravel())
            oof[valid]=model.predict_proba(tr[valid].reshape(-1,tr.shape[-1]))[:,1].reshape(y[valid].shape)
            print('calibration OOF',family,f,'heldout groups',int(valid.sum()),flush=True)
        model=new_pair_model(family);model.fit(tr.reshape(-1,tr.shape[-1]),y.ravel());joblib.dump(model,OUT/f'pair-{family}-model.joblib')
        pred=model.predict_proba(dv.reshape(-1,dv.shape[-1]))[:,1].reshape(dy.shape)
        trainpred[family]=oof;devpred[family]=pred
        z=summaries(pred);g=targets('development')
        audit[family]=dict(modelSha256=sha(OUT/f'pair-{family}-model.joblib'),developmentAveragePrecision=float(average_precision_score(dy.ravel(),pred.ravel())),
            developmentBrier=float(brier_score_loss(dy.ravel(),pred.ravel())),developmentLogLoss=float(log_loss(dy.ravel(),pred.ravel())),
            developmentC=float(score_quality(z[:,:,0],g).mean()),developmentM=float(score_quality(z[:,:,1],g).mean()),developmentExpectedM=float(score_quality(z[:,:,2],g).mean()))
        print('calibrated',family,json.dumps(audit[family]),flush=True)
    for stage,ps in [('training',trainpred),('development',devpred)]:
        np.savez_compressed(OUT/f'{stage}-pair-probabilities.npz',**ps);augment(stage,ps)
    write(OUT/'calibration-lock.json',dict(createdAt=datetime.now(timezone.utc).isoformat(),models=audit,folds=fold.tolist(),
        trainingOOF=True,testOpened=False,codeSha256=sha(__file__),scope='Probabilities estimate sampled observed links, not universal aesthetic compatibility.'))

def develop(freeze=False):
    import joblib
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    if (OUT/'calibrated-selection.json').exists():raise RuntimeError('Final selection exists')
    if (OUT/'test-pair-features.npz').exists():raise RuntimeError('Final test was opened')
    tr=np.load(OUT/'training-calibrated-features.npz');dv=np.load(OUT/'development-calibrated-features.npz');tg=targets('training');dg=targets('development')
    meta=read(OUT/'training-calibrated-feature-lock.json');chosen={};search=[]
    for family,allowed in [('upgraded',list(range(tr['X'].shape[-1]))),('learned-without-M',meta['withoutM'])]:
        candidates=[]
        base=len(read(OUT/'training-features-lock.json')['names'])
        compact=[j for j in allowed if j>=base or j in [base-2,base-1]]
        for group,columns in [('compact',compact),('all',allowed)]:
            for order,(name,model,kind) in enumerate(fit_models(tr['X'],tg,columns)):
                scores=predict_model(model,kind,dv['X'],columns);quality=float(score_quality(scores,dg).mean())
                r=dict(model=name,columns=columns,kind=kind,featureGroup=group,developmentQuality=quality)
                search.append(dict(family=family,**r));candidates.append((round(quality,12),-order-(0 if group=='compact' else 100),r,model))
        _,_,profile,model=max(candidates,key=lambda r:(r[0],r[1]));chosen[family]=profile
        if freeze:
            joblib.dump(model,OUT/f'calibrated-{family}-model.joblib');chosen[family]['modelSha256']=sha(OUT/f'calibrated-{family}-model.joblib')
    for j,name in enumerate(['same-C','old-M','char-novelty','dense-novelty','residual']):
        # old-M remains the previous uncalibrated algorithm; all other strong
        # baseline families may choose a calibrated support on development.
        upto=len(CONFIGS) if name=='old-M' else dv['baselines'].shape[2]
        qualities=[float(score_quality(dv['baselines'][:,:,k,j],dg).mean()) for k in range(upto)]
        k=max(range(upto),key=lambda k:(round(qualities[k],12),-k))
        chosen[name]=dict(index=k,column=j,config=meta['configs'][k],developmentQuality=qualities[k])
    result=dict(createdAt=datetime.now(timezone.utc).isoformat(),profiles=chosen,search=search,
        protocolSha256=sha(OUT/'protocol.json'),calibrationLockSha256=sha(OUT/'calibration-lock.json'),testOpened=False,
        primaryComparators=read(OUT/'protocol.json')['primaryComparators'],codeSha256=sha(__file__))
    write(OUT/('calibrated-selection.json' if freeze else 'calibrated-development-exploration.json'),result)
    if freeze:(OUT/'frozen-calibrated-ranker.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({n:{k:v for k,v in r.items() if k!='columns'} for n,r in chosen.items()},ensure_ascii=False),flush=True)

def test_features():
    import joblib
    if not (OUT/'calibrated-selection.json').exists():raise RuntimeError('Final selection must be frozen')
    x=np.load(OUT/'test-pair-features.npz')['X'];ps={}
    lock=read(OUT/'calibration-lock.json')
    for family in ['linear','boosted']:
        p=OUT/f'pair-{family}-model.joblib';assert sha(p)==lock['models'][family]['modelSha256']
        ps[family]=joblib.load(p).predict_proba(x.reshape(-1,x.shape[-1]))[:,1].reshape(x.shape[:-1])
    np.savez_compressed(OUT/'test-pair-probabilities.npz',**ps);augment('test',ps)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['pair-features','calibrate','develop','select','test-features']);p.add_argument('--split',choices=['training','development','test']);a=p.parse_args()
    if a.stage=='pair-features':pair_features(a.split)
    elif a.stage=='select':develop(True)
    elif a.stage=='test-features':test_features()
    else:globals()[a.stage]()
