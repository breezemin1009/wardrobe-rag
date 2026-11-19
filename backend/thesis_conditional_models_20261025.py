"""Cross-fitted relation support and monotone conditional novelty models."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import argparse,itertools,json,time
import numpy as np
from backend.thesis_conditional_novelty_20261025 import OUT,SEED
from backend.thesis_study_20261023 import read,write,sha,h
from backend.learned_case_utility import CONFIGS
from backend import thesis_adaptive_policy_20261024 as old

def features(stage):
    import joblib
    from sklearn.decomposition import PCA
    from backend.pog_refined_retrieval import Retriever
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    if stage in ['test','strict-test'] and not (OUT/'final-plan.json').exists():raise RuntimeError('Final predictions sealed')
    dest=OUT/f'{stage}-pair-features.npz'
    if dest.exists():raise RuntimeError('Frozen features exist')
    engine=Retriever(OUT);pp=OUT/'reference-pca.joblib'
    if not pp.exists():
        if stage!='training':raise RuntimeError('Reference PCA must be fitted first')
        ids=sorted({r[s] for r in engine.records for s in ['top','bottom']});pca=PCA(n_components=32,svd_solver='randomized',random_state=SEED)
        pca.fit(engine.vectors[[engine.ix[p] for p in ids]]);joblib.dump(pca,pp)
    projection=joblib.load(pp).transform(engine.vectors);rows=read(OUT/f'{stage}-frames.json');data=[];start=time.monotonic()
    for i,c in enumerate(rows):
        qs=[engine.ix[p] for p in c['contexts']];cs=[engine.ix[p] for p in c['bottoms']];nq=len(qs);nb=len(cs)
        matrices=[engine.matrix(c['contexts'],c['bottoms'],cfg) for cfg in CONFIGS]
        lex=(engine.lex[cs]@engine.lex[qs].T).toarray();dense=engine.vectors[cs]@engine.vectors[qs].T
        q=np.broadcast_to(projection[qs][None,:,:],(nb,nq,32));b=np.broadcast_to(projection[cs][:,None,:],(nb,nq,32))
        x=np.concatenate([np.stack(matrices+[lex,dense],axis=-1),q,b,np.abs(q-b),q*b],axis=-1)
        pad=np.zeros((9,12,139),dtype=np.float32);pad[:nb,:nq]=x;data.append(pad);engine.cache.clear()
        if i%50==0 or i+1==len(rows):print('features',stage,i+1,'/',len(rows),round(time.monotonic()-start,1),flush=True)
    np.savez_compressed(dest,X=np.asarray(data));write(OUT/f'{stage}-pair-feature-lock.json',dict(featureSha256=sha(dest),pcaSha256=sha(pp),codeSha256=sha(__file__)))

def pair_model(family):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    if family=='linear':return make_pipeline(StandardScaler(),LogisticRegression(C=.1,solver='lbfgs',max_iter=1000,random_state=SEED))
    return HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=15,min_samples_leaf=30,l2_regularization=10,learning_rate=.04,early_stopping=False,random_state=SEED)

def valid_pairs(X,rows,mask):
    xx=[];yy=[];ww=[]
    for i,c in enumerate(rows):
        if not mask[i]:continue
        nb=len(c['bottoms']);nq=len(c['contexts']);n=nb*nq;xx.append(X[i,:nb,:nq].reshape(n,-1));yy.extend(np.asarray(c['observed']).ravel());ww.extend([1/n]*n)
    weight=np.asarray(ww);weight*=len(weight)/weight.sum();return np.concatenate(xx),np.asarray(yy),weight

def fit_support():
    import joblib
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    if (OUT/'support-lock.json').exists() or (OUT/'test-pair-features.npz').exists():raise RuntimeError('Support fitting sealed')
    write(OUT/'development-plan.json',dict(createdAt=datetime.now(timezone.utc).isoformat(),
        pairModels='Four fixed models: standardized logistic C=.1 or HGB160trees,15leaves,min30,L2=10,lr.04,each with139 retrieval+title or130 title-only features. Fit ALL real master-context pairs, never zero padding.',
        supportSelection='35 quarter-simplex mixtures for a parent-balanced log-loss diagnostic. Final method uses the eight predeclared mixtures in protocol.json jointly with conditional head and fixed alpha, maximizing development parent NG under both rich-view safeguards. No-RAG uses five quarter-grid text mixtures.',
        noveltyModels='HGB160trees,leaf7/15 x L2 1/10,min30,lr.04. Conditional probability that no owned bottom has an observed link, trained only when the candidate link is observed. Parent-balanced weights. Last31 input features monotone decreasing; no-retrieval last9.',
        rankingAlphas=[0,.1,.3,.5,2/3,1],conditionalHeadsPerPath=4,maximumJointRAGPolicies=192,maximumJointNoRAGPolicies=120,
        primaryContrasts=['same-support-C','development-selected-C','old-CM'],primaryCorrection='Two-sided paired parent sign-flips,Holm over these3 predeclared contrasts. Remaining controls descriptive with individual paired intervals.',
        adoption=read(OUT/'protocol.json')['adoption'],
        codeSha256=sha(__file__)))
    rows=read(OUT/'training-frames.json');dev=read(OUT/'development-frames.json');tr=np.load(OUT/'training-pair-features.npz')['X'];dv=np.load(OUT/'development-pair-features.npz')['X'];fold=np.array([int(h(f'{SEED}:support:{c["id"]}')[:12],16)%3 for c in rows]);ps=[];ds=[];models={}
    for source,sl in [('retrieval',slice(None)),('text',slice(9,None))]:
        for family in ['linear','boosted']:
            oof=np.zeros(tr.shape[:-1],dtype=np.float32)
            for f in range(3):
                xx,y,w=valid_pairs(tr,rows,fold!=f);m=pair_model(family)
                if family=='linear':m.fit(xx[:,sl],y,logisticregression__sample_weight=w)
                else:m.fit(xx[:,sl],y,sample_weight=w)
                x=tr[fold==f,...,sl];oof[fold==f]=m.predict_proba(x.reshape(-1,x.shape[-1]))[:,1].reshape(x.shape[:-1]);print('OOF',source,family,f,flush=True)
            xx,y,w=valid_pairs(tr,rows,np.ones(len(rows),bool));m=pair_model(family)
            if family=='linear':m.fit(xx[:,sl],y,logisticregression__sample_weight=w)
            else:m.fit(xx[:,sl],y,sample_weight=w)
            name=f'{source}-{family}.joblib';joblib.dump(m,OUT/name);models[name]=sha(OUT/name);x=dv[...,sl]
            ps.append(oof);ds.append(m.predict_proba(x.reshape(-1,x.shape[-1]))[:,1].reshape(x.shape[:-1]).astype(np.float32))
    for stage,p in [('training',ps),('development',ds)]:np.savez_compressed(OUT/f'{stage}-probabilities.npz',P=np.stack(p,axis=-1))
    DP=np.stack(ds,axis=-1);search=[]
    for raw in itertools.product(range(5),repeat=4):
        if sum(raw)!=4:continue
        weight=np.asarray(raw)/4;loss=[]
        for i,c in enumerate(dev):
            p=np.clip((DP[i]@weight)[:len(c['bottoms']),:len(c['contexts'])],1e-7,1-1e-7);y=np.asarray(c['observed']);loss.append(float(-(y*np.log(p)+(1-y)*np.log(1-p)).mean()))
        search.append(dict(weights=weight.tolist(),developmentLogLoss=float(np.mean(loss))))
    chosen=min(search,key=lambda r:(r['developmentLogLoss'],-sum(r['weights'][:2])));no=min((r for r in search if sum(r['weights'][:2])==0),key=lambda r:r['developmentLogLoss'])
    write(OUT/'support-lock.json',dict(createdAt=datetime.now(timezone.utc).isoformat(),models=models,weights=chosen['weights'],noRAGWeights=no['weights'],search=search,folds=fold.tolist(),
        fitPairs=int(len(y)) if False else int(sum(len(c['bottoms'])*len(c['contexts']) for c in rows)),fitObservedLinks=int(sum(np.asarray(c['observed']).sum() for c in rows)),trainingOOF=True,allMasterContexts=True,
        selection=chosen,noRAGSelection=no,codeSha256=sha(__file__)))
    print('SUPPORT MIXTURE',json.dumps(chosen),'NO RAG',json.dumps(no),flush=True)

def novelty_inputs(X,P,cindices,windices,qindices,lex_similarity,dense_similarity,no_rag=False):
    """Fixed candidate features plus monotone nonnegative owned aggregates.

    Adding an owned garment can only increase the last31 (or9) features.
    The classifier constrains their effect on P(no existing link) to decrease.
    """
    qs=list(qindices);ci=list(cindices);wi=list(windices);n=len(ci);q=len(qs);keep=slice(2,None) if no_rag else slice(None)
    cx=X[ci][:,qs,9:] if no_rag else X[ci][:,qs];cp=P[ci][:,qs,keep];features=[cx,cp]
    op=P[wi][:,qs,keep] if wi else np.zeros((0,q,2 if no_rag else 4));extras=[]
    extras.extend([op.max(0) if wi else np.zeros(op.shape[1:]),op.sum(0)])
    if not no_rag:
        ok=X[wi][:,qs,:9] if wi else np.zeros((0,q,9));extras.extend([ok.max(0) if wi else np.zeros((q,9)),ok.sum(0)])
    owned=np.concatenate(extras,axis=-1);features.append(np.broadcast_to(owned,(n,*owned.shape)))
    sim=np.stack([lex_similarity.max(1) if wi else np.zeros(n),dense_similarity.max(1) if wi else np.zeros(n),lex_similarity.sum(1),dense_similarity.sum(1),np.full(n,len(wi))],axis=-1)
    features.append(np.broadcast_to(sim[:,None,:],(n,q,5)));result=np.concatenate(features,axis=-1).astype(np.float32)
    return result

def marginal_estimates(model,X,P,cindices,windices,qindices,lex_similarity,dense_similarity,weights,no_rag=False):
    """Canonical index inference, including exact empty/owned identities."""
    ci=list(cindices);oldwi=list(windices);wi=list(dict.fromkeys(oldwi));qs=list(dict.fromkeys(qindices))
    columns=[oldwi.index(i) for i in wi];lex_similarity=np.asarray(lex_similarity)[:,columns];dense_similarity=np.asarray(dense_similarity)[:,columns]
    base=P[ci][:,qs]@np.asarray(weights)
    if wi:
        x=novelty_inputs(X,P,ci,wi,qs,lex_similarity,dense_similarity,no_rag)
        factor=model.predict_proba(x.reshape(-1,x.shape[-1]))[:,1].reshape(base.shape)
    else:factor=np.ones(base.shape)
    for j,c in enumerate(ci):
        if c in wi:factor[j]=0
    return base.mean(1),(base*factor).mean(1),factor

def views(stage):
    from backend.pog_refined_retrieval import Retriever
    rows=read(OUT/f'{stage}-frames.json');X=np.load(OUT/f'{stage}-pair-features.npz')['X'];P=np.load(OUT/f'{stage}-probabilities.npz')['P'];engine=Retriever(OUT);lock=read(OUT/'support-lock.json');es=[]
    if (OUT/'method-selection.json').exists():
        profiles=read(OUT/'method-selection.json')['profiles'];lock={**lock,'weights':profiles['RAG']['supportWeights'],'noRAGWeights':profiles['no-RAG']['supportWeights']}
    for i,c in enumerate(rows):
        obs=np.asarray(c['observed']);ix=[engine.ix[p] for p in c['bottoms']];lex=np.clip((engine.lex[ix]@engine.lex[ix].T).toarray(),0,1);dense=np.clip(engine.vectors[ix]@engine.vectors[ix].T,0,1)
        for name,v in c['views'].items():
            ci=v['c'];wi=v['w'];qs=v['q'];prior=obs[np.ix_(wi,qs)].max(0) if wi else np.zeros(len(qs));y=obs[np.ix_(ci,qs)];novel=1-prior;gain=(y*novel).sum(1)
            if not gain.max():continue
            p=P[i,ci][:,qs];base=p@np.asarray(lock['weights']);npbase=p@np.asarray(lock['noRAGWeights']);op=P[i,wi][:,qs] if wi else np.zeros((0,len(qs),4))
            oldbase=(op@np.asarray(lock['weights'])).max(0) if wi else np.zeros(len(qs));oldnp=(op@np.asarray(lock['noRAGWeights'])).max(0) if wi else np.zeros(len(qs))
            simlex=lex[np.ix_(ci,wi)];simdense=dense[np.ix_(ci,wi)]
            es.append(dict(parent=i,id=c['id'],view=name,c=ci,w=wi,q=qs,positive=y,novel=novel,gain=gain,
                C=base.mean(1),oldM=np.maximum(base-oldbase,0).mean(1),residual=(base*(1-oldbase)).mean(1),base=base,noRAGBase=npbase,noRAGC=npbase.mean(1),noRAGOldM=np.maximum(npbase-oldnp,0).mean(1),
                X=novelty_inputs(X[i],P[i],ci,wi,qs,simlex,simdense),noRAGX=novelty_inputs(X[i],P[i],ci,wi,qs,simlex,simdense,True),
                rawKernels=X[i,ci][:,qs,:9],oldRawKernels=X[i,wi][:,qs,:9] if wi else np.zeros((0,len(qs),9)),
                maxLex=simlex.max(1) if wi else np.zeros(len(ci)),maxDense=simdense.max(1) if wi else np.zeros(len(ci)),components=p,ownedComponents=op))
    return es

def train_novelty():
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    if (OUT/'novelty-selection.json').exists() or (OUT/'test-pair-features.npz').exists():raise RuntimeError('Conditional model selection sealed')
    tr=views('training');dv=views('development');search=[];profiles={};models={};weights=Counter(e['parent'] for e in tr if e['w'] and e['positive'].sum())
    for path in ['RAG','no-RAG']:
        key='X' if path=='RAG' else 'noRAGX';basekey='base' if path=='RAG' else 'noRAGBase';ckey='C' if path=='RAG' else 'noRAGC';xx=[];yy=[];ww=[]
        for e in tr:
            if not e['w']:continue
            mask=e['positive'].astype(bool);n=int(mask.sum())
            if not n:continue
            xx.append(e[key][mask]);yy.extend(np.broadcast_to(e['novel'],e['positive'].shape)[mask]);ww.extend([1/(weights[e['parent']]*n)]*n)
        x=np.concatenate(xx);y=np.asarray(yy,dtype=int);weight=np.asarray(ww);weight*=len(weight)/weight.sum();monotone=np.r_[np.zeros(x.shape[1]-(31 if path=='RAG' else 9),dtype=int),-np.ones(31 if path=='RAG' else 9,dtype=int)]
        allx=np.concatenate([e[key].reshape(-1,x.shape[-1]) for e in dv]);sizes=np.cumsum([e[key].shape[0]*e[key].shape[1] for e in dv])[:-1]
        best=None
        for leaves in [7,15]:
            for reg in [1,10]:
                m=HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=leaves,min_samples_leaf=30,l2_regularization=reg,learning_rate=.04,early_stopping=False,monotonic_cst=monotone,random_state=SEED)
                m.fit(x,y,sample_weight=weight);probs=np.split(m.predict_proba(allx)[:,1],sizes);estimates=[(e[basekey]*(p.reshape(e[basekey].shape) if e['w'] else 1)).mean(1) for e,p in zip(dv,probs)]
                name=f'{path}-novelty-leaf{leaves}-reg{reg}.joblib';joblib.dump(m,OUT/name);models[name]=sha(OUT/name)
                for alpha in [0,.1,.3,.5,2/3,1]:
                    ss=[alpha*e[ckey]+(1-alpha)*M for e,M in zip(dv,estimates)];q=float(old.parent_quality(ss,dv).mean());row=dict(path=path,model=name,sha256=models[name],alpha=alpha,developmentQuality=q,dimension=x.shape[1],trainingPairs=len(y),positiveNovelty=int(y.sum()),leaves=leaves,reg=reg)
                    search.append(row)
                    if best is None or (round(q,12),-alpha,-leaves,reg)>(round(best['developmentQuality'],12),-best['alpha'],-best['leaves'],best['reg']):best={**row,'leaves':leaves,'reg':reg}
        profiles[path]=best;print('NOVELTY SELECTED',json.dumps(best),flush=True)
    write(OUT/'novelty-selection.json',dict(createdAt=datetime.now(timezone.utc).isoformat(),profiles=profiles,search=search,models=models,testOpened=False,trainingOOF=True,codeSha256=sha(__file__)))

def method_profiles():
    return read(OUT/('method-selection.json' if (OUT/'method-selection.json').exists() else 'novelty-selection.json'))['profiles']

def safe_select():
    import joblib
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    if (OUT/'method-selection.json').exists() or (OUT/'test-pair-features.npz').exists():raise RuntimeError('Method selection sealed')
    selection=read(OUT/'novelty-selection.json');lock=read(OUT/'support-lock.json');es=views('development');audit=[];profiles={}
    mixtures=[lock['weights'],[.25,.25,.25,.25],[.25,0,.5,.25],[0,.25,.25,.5],[0,0,.25,.75],[0,0,.5,.5],[0,.5,0,.5],[.5,0,.5,0]]
    mixtures=list({tuple(w):w for w in mixtures}.values())
    for path,key in [('RAG','X'),('no-RAG','noRAGX')]:
        weights=mixtures if path=='RAG' else [[0,0,a,1-a] for a in [0,.25,.5,.75,1]]
        x=np.concatenate([e[key].reshape(-1,e[key].shape[-1]) for e in es]);ends=np.cumsum([e[key].shape[0]*e[key].shape[1] for e in es])[:-1];candidates=[]
        for filename in sorted({r['model'] for r in selection['search'] if r['path']==path}):
            m=joblib.load(OUT/filename);pp=np.split(m.predict_proba(x)[:,1],ends);factors=[p.reshape(e['base'].shape) if e['w'] else np.ones(e['base'].shape) for e,p in zip(es,pp)]
            for w in weights:
                bases=[e['components']@np.asarray(w) for e in es];C=[b.mean(1) for b in bases];M=[(b*f).mean(1) for b,f in zip(bases,factors)]
                for r in [r for r in selection['search'] if r['path']==path and r['model']==filename]:
                    alpha=r['alpha'];ss=[alpha*c+(1-alpha)*mm for c,mm in zip(C,M)];q=float(old.parent_quality(ss,es).mean());diff={v:float(np.mean([old.quality(s,e['gain'])-old.quality(c,e['gain']) for s,c,e in zip(ss,C,es) if e['view']==v])) for v in ['rich','owned-rich']}
                    good=all(d>=-1e-12 for d in diff.values()) and alpha<1;row={**r,'supportWeights':w,'developmentQuality':q,'developmentRichDifferences':diff,'eligible':good};audit.append(row)
                    if good:candidates.append(row)
        if not candidates:raise RuntimeError('No nontrivial development-safe conditional method: '+path)
        chosen=max(candidates,key=lambda r:(round(r['developmentQuality'],12),-r['alpha'],sum(r['supportWeights'][:2]),-r['leaves']));profiles[path]=chosen;print('FINAL METHOD',json.dumps(chosen),flush=True)
    write(OUT/'method-selection.json',dict(createdAt=datetime.now(timezone.utc).isoformat(),profiles=profiles,search=audit,originalSelectionSha256=sha(OUT/'novelty-selection.json'),protocolSha256=sha(OUT/'protocol.json'),testOpened=False,codeSha256=sha(__file__)))

def predict(stage='test'):
    import joblib
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    lock=read(OUT/'final-plan.json')
    for name,value in lock['files'].items():assert sha(OUT/name)==value,name
    for name,value in lock['implementationHashes'].items():assert sha(Path(__file__).parent/name)==value,name
    if (OUT/f'{stage}-probabilities.npz').exists():raise RuntimeError('Final probabilities already materialized')
    X=np.load(OUT/f'{stage}-pair-features.npz')['X'];ps=[]
    for name in ['retrieval-linear.joblib','retrieval-boosted.joblib','text-linear.joblib','text-boosted.joblib']:
        x=X if name.startswith('retrieval') else X[...,9:];m=joblib.load(OUT/name)
        ps.append(m.predict_proba(x.reshape(-1,x.shape[-1]))[:,1].reshape(x.shape[:-1]).astype(np.float32))
    np.savez_compressed(OUT/f'{stage}-probabilities.npz',P=np.stack(ps,axis=-1))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['features','fit-support','train-novelty','safe-select','predict']);p.add_argument('--split',choices=['training','development','test','strict-test']);a=p.parse_args()
    if a.stage=='features':features(a.split)
    elif a.stage=='predict':predict(a.split or 'test')
    else:globals()[a.stage.replace('-','_')]()
