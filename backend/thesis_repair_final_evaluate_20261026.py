"""A single opening of the new study after development and evaluation locks.

The primary family was specified before model fitting. This file records the
numerical implementation before new test features/predictions are opened.
"""
import argparse, importlib.metadata, platform, time
from pathlib import Path
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from backend import thesis_repair_expanded_training_20261026 as study
from backend.thesis_repair_expanded_models_20261026 import feature_view
from backend import thesis_repair_expanded_models_20261026 as modeling
from backend import thesis_repair_stable_kernel_20261026 as kernel
from backend import thesis_repair_models_20261026 as base

OUT=study.OUT
COMPARATORS=['noRAG','direct','oldRAG','sameSupportC']

def chosen_value(scores, values):
    scores=np.asarray(scores,dtype=float);values=np.asarray(values,dtype=float)
    chosen=np.flatnonzero(np.abs(scores-scores.max())<=1e-12)
    return float(values[chosen].mean()),chosen.tolist()

def values(S, frame, alpha):
    """Observed-link utility; absent records are not aesthetic negatives."""
    S=np.asarray(S,dtype=float)
    assert S.shape==(4,len(frame['contexts'])) and np.isfinite(S).all()
    assert S.min()>=-1e-7 and S.max()<=1+1e-7 and 0<=alpha<=1
    observed=np.asarray(frame['observed'],dtype=float)
    history=np.asarray(frame['knownHistory'],dtype=float)
    w=frame['views']['primary']['w'];q=len(frame['contexts'])
    old=history[w].max(0) if w else np.zeros(q)
    actualold=observed[w].max(0) if w else np.zeros(q)
    gain=(observed[:4]*(1-actualold)).sum(1)
    C=S.mean(1);M=(S*(1-old)).mean(1);U=alpha*C+(1-alpha)*M
    selected,ix=chosen_value(U,gain)
    ng=selected/gain.max() if gain.max()>0 else 0.
    return dict(C=C.tolist(),M=M.tolist(),U=U.tolist(),gain=gain.tolist(),
                chosen=ix,NG=float(ng),selectedGain=selected,
                selectedNewLinkFraction=selected/q,maxGain=float(gain.max()),
                knownCoverage=int(old.sum()))

def paired(d,seed_boot=2026102632,seed_sign=2026102633,B=10000,P=50000):
    d=np.asarray(d,dtype=float);n=len(d)
    assert n and np.isfinite(d).all()
    rng=np.random.default_rng(seed_boot);boots=[]
    for begin in range(0,B,256):
        count=min(256,B-begin);idx=rng.integers(0,n,size=(count,n))
        boots.extend(d[idx].mean(1).tolist())
    target=abs(float(d.mean()));rng=np.random.default_rng(seed_sign);extreme=0
    nz=d[np.abs(d)>1e-15]
    if not len(nz):pvalue=1.;extreme=P
    else:
        for begin in range(0,P,512):
            count=min(512,P-begin)
            signs=rng.integers(0,2,size=(count,len(nz)),dtype=np.int8)*2-1
            perm=np.abs((signs*nz).sum(1)/n)
            extreme+=int((perm>=target-1e-14).sum())
        pvalue=(extreme+1)/(P+1)
    return dict(n=n,mean=float(d.mean()),CI95=np.quantile(boots,[.025,.975]).tolist(),
                rawP=pvalue,extreme=extreme,permutations=P,
                finiteMCStandardError=float(np.sqrt(pvalue*(1-pvalue)/(P+1))),
                changedParents=int(len(nz)))

def holm(raw):
    raw=np.asarray(raw,float);order=np.argsort(raw,kind='stable');result=np.empty(len(raw))
    previous=0.
    for i,j in enumerate(order):
        previous=max(previous,min(1.,float(raw[j]*(len(raw)-i))))
        result[j]=previous
    return result.tolist()

def reserve():
    study.verify();assert base.read(OUT/'raw-source-audit.json')['allPassed'];binding=base.read(OUT/'selection-binding.json')
    assert binding['newTestOpened'] is False
    assert not (OUT/'test-features.npz').exists() and not (OUT/'test-results.json').exists()
    assert not (OUT/'evaluation-lock.json').exists()
    lock=base.read(OUT/'selection-lock.json')
    assert lock['modelCodeSha256']==base.sha(modeling.__file__)
    assert binding['selectionSha256']==base.sha(OUT/'selection-lock.json')
    for name in ['RAG','noRAG']:
        p=lock['profiles'][name];assert base.sha(OUT/p['modelFile'])==p['modelSha256']
    secondary=['learning-audit.json','numerical-feature-audit.json','extra-numerical-audit.json','dependence-plan.json','dependence-groups.json','near-training-plan.json','near-training-mask.json','near-training-models.json','illustration-selection.json']
    assert all((OUT/n).exists() for n in secondary)
    for n in ['learning-audit.json','numerical-feature-audit.json','extra-numerical-audit.json']:
        assert base.read(OUT/n)['allPassed']
    base.write(OUT/'evaluation-lock.json',dict(createdAt=base.stamp(),secondaryBindings={n:base.sha(OUT/n) for n in secondary},numericalKernelSha256=base.sha(kernel.__file__),
        codeSha256=base.sha(__file__),selectionSha256=base.sha(OUT/'selection-lock.json'),
        protocolSha256=base.sha(OUT/'protocol.json'),
        learningLockSha256=base.sha(OUT/'learning-lock.json'),
        primaryComparisons=COMPARATORS,metric='Mean parent NG; all-zero gain NG=0; max-score ties averaged at1e-12',
        numericalConvention='Canonical sorted query IDs; compute each candidate separately (Nc=1), then restore stored request order. Matches research API and prevents candidate batch size/order from changing floating-point feature evaluation. Fixed before final features/predictions.',
        statistics='Paired percentile bootstrap10000, paired two-sided sign-flip50000; (extreme+1)/(P+1); Holm4; fixed seeds from original protocol',
        gate='Each comparison: positive observed difference, lower95%CI>0 and min(1,2*Holm-adjusted p)<.05; factor2 retains both development/test rounds. No changing the four-comparison family.',
        oldRAG='Forward hybrid density K32 F[...,11], alpha0, same conservative reference exclusions. This is the earlier kernel under the new common conditions, not a claim of identical original deployment.',
        secondary='Uniform and H-only descriptive; kind,query count,owned count,coverage and nonzero-task groups descriptive. No subgroup claim selected after outcomes.',
        runtime=dict(python=platform.python_version(),**{n:importlib.metadata.version(n) for n in ['numpy','scipy','scikit-learn','joblib']})))
    print('EVALUATION IMPLEMENTATION LOCKED',base.sha(OUT/'evaluation-lock.json'),flush=True)

def extract():
    from backend.pog_refined_retrieval import Retriever
    study.verify();threadpool_limits(1)
    e=base.read(OUT/'evaluation-lock.json')
    assert e['codeSha256']==base.sha(__file__) and not (OUT/'test-features.npz').exists()
    assert e['selectionSha256']==base.sha(OUT/'selection-lock.json')
    assert e['numericalKernelSha256']==base.sha(kernel.__file__)
    for name,digest in e['secondaryBindings'].items():assert base.sha(OUT/name)==digest
    frames=base.read(OUT/'test-frames.json');engine=Retriever(OUT);arrays={};start=time.monotonic()
    parts=OUT/'test-feature-parts';parts.mkdir(exist_ok=True)
    signature=dict(evaluationSha256=base.sha(OUT/'evaluation-lock.json'),framesSha256=base.sha(OUT/'test-frames.json'))
    if (parts/'binding.json').exists():assert base.read(parts/'binding.json')==signature
    else:base.write(parts/'binding.json',signature)
    for i,c in enumerate(frames):
        checkpoint=parts/f'{i:04d}.npz'
        if checkpoint.exists():
            with np.load(checkpoint) as cached:
                for n,width in [('X',514),('F',12),('R',12)]:
                    a=cached[n];assert a.shape==(4,len(c['contexts']),width) and np.isfinite(a).all();arrays[f'{n}{i}']=a
            continue
        stored=c['contexts'];qs=sorted(stored);cs=c['bottoms'][:4];qi=[qs.index(q) for q in stored];xx=[];ff=[];rr=[]
        for candidate in cs:
            xx.append(kernel.base_features(engine,qs,[candidate])[:,qi])
            f,r,_=kernel.support_features(engine,qs,[candidate]);ff.append(f[:,qi]);rr.append(r[:,qi])
        x=np.concatenate(xx);f=np.concatenate(ff);r=np.concatenate(rr)
        arrays.update({f'X{i}':x,f'F{i}':f,f'R{i}':r})
        temporary=checkpoint.with_suffix('.tmp')
        with temporary.open('wb') as handle:np.savez_compressed(handle,X=x,F=f,R=r)
        temporary.replace(checkpoint)
        if i%100==0:print('FINAL LOCKED FEATURES',i,'/',len(frames),'seconds',round(time.monotonic()-start,1),flush=True)
    temporary=OUT/'test-features.tmp'
    with temporary.open('wb') as handle:np.savez_compressed(handle,**arrays)
    temporary.replace(OUT/'test-features.npz')
    base.write(OUT/'test-feature-binding.json',dict(featuresSha256=base.sha(OUT/'test-features.npz'),
        protocolSha256=base.sha(OUT/'protocol.json'),learningLockSha256=e['learningLockSha256'],codeSha256=base.sha(__file__)))
    print('FINAL FEATURES SEALED',len(frames),flush=True)

def run():
    threadpool_limits(1);start=time.monotonic();study.verify()
    if (OUT/'test-results.json').exists() or (OUT/'test-predictions.json').exists():
        raise RuntimeError('New test already opened; refusing overwrite')
    e=base.read(OUT/'evaluation-lock.json');lock=base.read(OUT/'selection-lock.json')
    assert e['codeSha256']==base.sha(__file__)
    assert e['selectionSha256']==base.sha(OUT/'selection-lock.json')
    assert e['protocolSha256']==base.sha(OUT/'protocol.json')
    assert e['numericalKernelSha256']==base.sha(kernel.__file__)
    for name,digest in e['secondaryBindings'].items():assert base.sha(OUT/name)==digest
    featurebinding=base.read(OUT/'test-feature-binding.json')
    assert featurebinding['featuresSha256']==base.sha(OUT/'test-features.npz')
    assert featurebinding['learningLockSha256']==e['learningLockSha256']
    models={}
    for name in ['RAG','noRAG']:
        p=lock['profiles'][name];assert base.sha(OUT/p['modelFile'])==p['modelSha256']
        models[name]=joblib.load(OUT/p['modelFile'])
    z=np.load(OUT/'test-features.npz');frames=base.read(OUT/'test-frames.json');rows=[];supports={}
    for i,c in enumerate(frames):
        X=z[f'X{i}'];F=z[f'F{i}'];R=z[f'R{i}'];q=len(c['contexts'])
        S={}
        for name in ['RAG','noRAG']:
            features=feature_view(X,F,R,name,lock['profiles'][name]['view'])
            S[name]=models[name].predict_proba(features.reshape(-1,features.shape[-1]))[:,1].reshape(4,q)
        p=lock['profiles']['direct'];feature=p['model']
        D=X[...,512] if feature=='dense' else X[...,513] if feature=='char' else (X[...,512]+X[...,513])/2
        S['direct']=(1+D)/2;S['oldRAG']=F[...,11].astype(float);S['sameSupportC']=S['RAG']
        S['uniform']=np.ones((4,q));S['Honly']=np.ones((4,q))
        results={}
        for name,support in S.items():
            alpha=lock['profiles'][name]['alpha'] if name in lock['profiles'] else 1 if name in ['sameSupportC','uniform'] else 0
            results[name]=values(support,c,alpha)
            supports[f'{name}{i}']=support
        rows.append(dict(id=c['id'],kind=c['kind'],qCount=q,wCount=len(c['bottoms'])-4,methods=results))
        if i%100==0:print('NEW TEST PREDICTION',i+1,'/',len(frames),'seconds',round(time.monotonic()-start,1),flush=True)
    names=list(rows[0]['methods']);summary={}
    for name in names:
        rr=[r['methods'][name] for r in rows]
        summary[name]=dict(NG=float(np.mean([r['NG'] for r in rr])),
            selectedNewLinkFraction=float(np.mean([r['selectedNewLinkFraction'] for r in rr])),
            meanSelectedGain=float(np.mean([r['selectedGain'] for r in rr])),
            nonzeroTaskNG=float(np.mean([r['NG'] for r in rr if r['maxGain']>0])))
    comparisons={name:paired([r['methods']['RAG']['NG']-r['methods'][name]['NG'] for r in rows]) for name in COMPARATORS}
    corrected=holm([comparisons[n]['rawP'] for n in COMPARATORS])
    for name,pvalue in zip(COMPARATORS,corrected):
        v=comparisons[name];v['holmP']=pvalue
        v['roundAdjustedP']=min(1.,2*pvalue)
        v['passed']=bool(v['mean']>0 and v['CI95'][0]>0 and v['roundAdjustedP']<.05)
    groups={}
    for key in ['kind','qCount','wCount','knownCoverage']:
        get=lambda r:r['methods']['RAG']['knownCoverage'] if key=='knownCoverage' else r[key]
        for category in sorted(set(get(r) for r in rows),key=str):
            sub=[r for r in rows if get(r)==category]
            groups[f'{key}:{category}']=dict(n=len(sub),methods={name:float(np.mean([r['methods'][name]['NG'] for r in sub])) for name in names})
    np.savez_compressed(OUT/'test-support.npz',**supports)
    base.write(OUT/'test-predictions.json',rows)
    result=dict(createdAt=base.stamp(),n=len(rows),summary=summary,comparisons=comparisons,groups=groups,
        allFourPrimaryPassed=all(v['passed'] for v in comparisons.values()),
        nonzeroGainParents=sum(r['methods']['RAG']['maxGain']>0 for r in rows),
        protocolSha256=base.sha(OUT/'protocol.json'),evaluationLockSha256=base.sha(OUT/'evaluation-lock.json'),
        selectionSha256=base.sha(OUT/'selection-lock.json'),predictionsSha256=base.sha(OUT/'test-predictions.json'),
        supportSha256=base.sha(OUT/'test-support.npz'),featuresSha256=base.sha(OUT/'test-features.npz'),
        codeSha256=base.sha(__file__),seconds=time.monotonic()-start)
    base.write(OUT/'test-results.json',result)
    base.write(OUT/'test-opened.json',dict(createdAt=base.stamp(),resultSha256=base.sha(OUT/'test-results.json'),selectionSha256=base.sha(OUT/'selection-lock.json')))
    print('RESULTS',__import__('json').dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['reserve','extract','run'])
    globals()[parser.parse_args().stage]()
