"""Fixed, equal-budget support learning on core parents and real source pairs."""
import argparse,time,json
from pathlib import Path
import joblib
import warnings
import numpy as np
from threadpoolctl import threadpool_limits
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from backend import thesis_repair_expanded_training_20261026 as study
from backend import thesis_repair_models_20261026 as base

OUT=study.OUT;read=base.read;write=base.write;sha=base.sha

def atomic(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp');write(temporary,value);temporary.replace(path)

def feature_view(X,F,R,path,view):
    x=X[...,512:514] if view=='compact' else X
    return np.concatenate([x,F,R],axis=-1) if path=='RAG' else x

def make_model(name):
    if name.startswith('LR'):
        return make_pipeline(StandardScaler(),LogisticRegression(C=float(name[4:]),max_iter=2000,random_state=2026102641))
    return HistGradientBoostingClassifier(max_iter=200,learning_rate=.05,max_leaf_nodes=int(name.split('leaf')[1]),
        l2_regularization=10,min_samples_leaf=20,early_stopping=False,random_state=2026102641)

def fit(m,X,y,w):
    with warnings.catch_warnings():
        warnings.simplefilter('error',ConvergenceWarning)
        if hasattr(m,'steps'):m.fit(X,y,standardscaler__sample_weight=w,logisticregression__sample_weight=w)
        else:m.fit(X,y,sample_weight=w)
    return m

def parent_ng(s,c,a):
    return float(base.evaluate_support([s],[c],a)[0])

def reserve():
    study.verify();assert not (OUT/'selection-lock.json').exists() and not (OUT/'test-features.npz').exists()
    assert not (OUT/'learning-lock.json').exists()
    for part in ['core','extra']:
        binding=read(OUT/f'{part}-feature-lock.json')
        assert binding['protocolSha256']==sha(OUT/'protocol.json')
        assert binding['featuresSha256']==sha(OUT/f'{part}-features.npz')
        with np.load(OUT/f'{part}-features.npz') as arrays:
            assert all(np.isfinite(arrays[k]).all() for k in arrays.files)
    from backend import pog_retrieval,pog_refined_retrieval
    dependencies={m.__name__:dict(sourceFile=Path(m.__file__).name,sha256=sha(m.__file__)) for m in [pog_retrieval,pog_refined_retrieval]}
    atomic(OUT/'learning-lock.json',dict(createdAt=base.stamp(),modelCodeSha256=sha(__file__),protocolSha256=sha(OUT/'protocol.json'),dependencies=dependencies,
        coreFeaturesSha256=sha(OUT/'core-features.npz'),extraFeaturesSha256=sha(OUT/'extra-features.npz'),
        extraPairSha256=sha(OUT/'extra-pairs.json'),featureOrder='Same514title and24fixedpointwise kernels; compact keeps only2titlecos before optional24kernels',
        tieBreak='Highest mean parent OOFNG,then lowest core weighted OOF logloss,then smaller alpha; stable model enumeration if all equal'))
    print('EXPANDED LEARNING LOCKED',sha(OUT/'learning-lock.json'),flush=True)

def develop():
    study.verify();p=read(OUT/'protocol.json');lock=read(OUT/'learning-lock.json')
    assert lock['modelCodeSha256']==sha(__file__) and lock['protocolSha256']==sha(OUT/'protocol.json')
    for dependency in lock['dependencies'].values():assert sha(Path(__file__).parent/dependency['sourceFile'])==dependency['sha256']
    assert not (OUT/'selection-lock.json').exists() and not (OUT/'test-features.npz').exists()
    for part in ['core','extra']:assert sha(OUT/f'{part}-features.npz')==lock[f'{part}FeaturesSha256']
    threadpool_limits(1);frames=read(OUT/'core-frames.json');z=np.load(OUT/'core-features.npz');e=np.load(OUT/'extra-features.npz')
    extra=read(OUT/'extra-pairs.json');n=len(frames);q=[len(c['contexts']) for c in frames]
    folds=np.asarray([int(base.stable('2026102641:fold:'+c['id'])[:8],16)%5 for c in frames])
    y=np.concatenate([np.asarray(c['observed'][:4]).ravel() for c in frames]);ey=np.asarray([r['y'] for r in extra])
    corew=np.concatenate([np.full(4*k,5/k) for k in q]);ew=np.full(len(extra),20*n/len(extra))
    groups=np.concatenate([np.full(4*k,i) for i,k in enumerate(q)]);rowfold=folds[groups]
    offsets=np.cumsum([0]+[4*k for k in q]);search=[];profiles={};oof={};start=time.monotonic()
    for path in ['noRAG','RAG']:
        for spec in p['models']:
            view=spec['view'];name=spec['family'];tag=path+'__'+view+'__'+name
            X=np.concatenate([feature_view(z[f'X{i}'],z[f'F{i}'],z[f'R{i}'],path,view).reshape(4*q[i],-1) for i in range(n)])
            EX=feature_view(e['X'],e['F'],e['R'],path,view);prediction=np.empty(len(y))
            cp=OUT/(tag+'-checkpoint.json');predfile=OUT/(tag+'-oof.npy');modelpath=OUT/(tag+'.joblib')
            if cp.exists():
                checkpoint=read(cp)
                assert checkpoint['learningLockSha256']==sha(OUT/'learning-lock.json')
                assert checkpoint['modelSha256']==sha(modelpath) and checkpoint['OOFSha256']==sha(predfile)
                prediction=np.load(predfile)
                print('RESUMED COMPLETE MODEL',tag,flush=True)
            else:
                for fold in range(5):
                    tr=rowfold!=fold;dv=~tr;m=make_model(name)
                    fold_extra_weight=ew*(corew[tr].sum()/ew.sum())
                    fit(m,np.concatenate([X[tr],EX]),np.concatenate([y[tr],ey]),np.concatenate([corew[tr],fold_extra_weight]))
                    prediction[dv]=m.predict_proba(X[dv])[:,1]
                fitted=fit(make_model(name),np.concatenate([X,EX]),np.concatenate([y,ey]),np.concatenate([corew,ew]))
                if modelpath.exists():modelpath.replace(OUT/(tag+'-interrupted-'+base.stable(str(time.time_ns()))[:8]+'.joblib'))
                joblib.dump(fitted,modelpath);np.save(predfile,prediction)
                atomic(cp,dict(createdAt=base.stamp(),learningLockSha256=sha(OUT/'learning-lock.json'),modelSha256=sha(modelpath),OOFSha256=sha(predfile)))
            ss=[prediction[offsets[i]:offsets[i+1]].reshape(4,q[i]) for i in range(n)]
            oof[tag]=prediction
            loss=float(np.average(-(y*np.log(np.clip(prediction,1e-7,1-1e-7))+(1-y)*np.log(np.clip(1-prediction,1e-7,1-1e-7))),weights=corew))
            for alpha in p['alphas']:
                ng=[parent_ng(s,c,alpha) for s,c in zip(ss,frames)]
                search.append(dict(path=path,view=view,model=name,alpha=alpha,OOF_NG=float(np.mean(ng)),OOF_logloss=loss,tag=tag))
            print('EXPANDED OOF',tag,'NG',max(t['OOF_NG'] for t in search if t['tag']==tag),'seconds',round(time.monotonic()-start,1),flush=True)
        profiles[path]=max([v for v in search if v['path']==path],key=lambda v:(v['OOF_NG'],-v['OOF_logloss'],-v['alpha']))
    for name in p['directModels']:
        ss=[(1+(z[f'X{i}'][...,512] if name=='dense' else z[f'X{i}'][...,513] if name=='char' else (z[f'X{i}'][...,512]+z[f'X{i}'][...,513])/2))/2 for i in range(n)]
        for alpha in p['alphas']:
            search.append(dict(path='direct',model=name,alpha=alpha,OOF_NG=float(np.mean([parent_ng(s,c,alpha) for s,c in zip(ss,frames)]))))
    profiles['direct']=max([v for v in search if v['path']=='direct'],key=lambda v:(v['OOF_NG'],-v['alpha']))
    for name in ['RAG','noRAG']:
        v=profiles[name];v['modelFile']=v['tag']+'.joblib';v['modelSha256']=sha(OUT/v['modelFile'])
    np.savez_compressed(OUT/'oof-support.npz',**oof)
    result=dict(createdAt=base.stamp(),profiles=profiles,search=search,folds=folds.tolist(),parents=n,pairRows=len(y),sourcePairRows=len(extra),
        sourceCorpusWeight=float(ew.sum()),coreCorpusWeight=float(corew.sum()),
        protocolSha256=sha(OUT/'protocol.json'),learningLockSha256=sha(OUT/'learning-lock.json'),modelCodeSha256=sha(__file__),
        OOFsupportSha256=sha(OUT/'oof-support.npz'),warning='Selected OOF estimates are development. The earlier1200-case test is now training/development and is not reused as final evidence.')
    atomic(OUT/'selection-lock.json',result)
    atomic(OUT/'selection-binding.json',dict(createdAt=base.stamp(),selectionSha256=sha(OUT/'selection-lock.json'),protocolSha256=sha(OUT/'protocol.json'),newTestOpened=False))
    print('EXPANDED MODEL SELECTION FROZEN',json.dumps(profiles),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['reserve','develop']);globals()[parser.parse_args().stage]()
