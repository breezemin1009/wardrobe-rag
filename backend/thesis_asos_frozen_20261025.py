"""One prespecified zero-shot ASOS test of the frozen POG methods.

No ASOS observations enter retrieval, PCA, TF-IDF, model fitting, calibration,
weight selection or thresholds. The graph is only the problem constructor and
evaluation target. Whole master garments and source outfits are disjoint
between parent units. Original POG failures and adoption decisions are retained.
"""
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
import argparse, json, hashlib, time
import numpy as np
from backend.thesis_study_20261023 import read,write,sha,h
from backend import thesis_adaptive_policy_20261024 as metric

ROOT=Path(__file__).parent/'data/menswear-expansion-v3'
ASOS=ROOT/'asos'; OUT=ASOS/'frozen-transfer-full-source-20261025'
METHOD=ROOT/'pog/joint-reserved-20261025'
SERVICE=ROOT/'pog/prospective-upgrade-20261024'
SEED=2026102547
METHODS=['ensemble','ensemble-C','conditional','product-residual','old-kernel-CM',
         'direct-ranker','neural-direct-ranker','no-runtime-retrieval','service-M','service-C']
PRIMARY=[('ensemble','ensemble-C'),('ensemble','old-kernel-CM'),('service-M','service-C')]

def mappings(folder):
    items=read(folder/'items.json');cats=defaultdict(Counter)
    for r in read(folder/'reference.json'):
        for s in ['top','bottom']:
            x=items[r[s]];cats[x['side'],x['kind']][x['sourceCategory']]+=1
    return {k:v.most_common(1)[0][0] for k,v in cats.items()}

def prepare():
    if OUT.exists():raise RuntimeError('ASOS cohort already reserved')
    items=read(ASOS/'projected/items.json');edges=read(ASOS/'projected/edges.json')
    targetmap=mappings(METHOD);servicemap=mappings(SERVICE)
    supported={k for k in targetmap if k in servicemap};items={p:x for p,x in items.items() if (x['side'],x['kind']) in supported}
    g=defaultdict(set);rev=defaultdict(set);io={p:set(v) for p,v in read(ASOS/'full-source-incidence.json').items()}
    for r in edges:
        if r['top'] not in items or r['bottom'] not in items:continue
        q,b=r['top'],r['bottom'];g[q].add(b);rev[b].add(q)
    pools=defaultdict(list);degreepool=defaultdict(list)
    key=lambda p:(len(io[p]),h(f'{SEED}:{p}'))
    for b in rev:pools[items[b]['kind']].append(b);degreepool[items[b]['kind'],len(rev[b])].append(b)
    for v in pools.values():v.sort(key=key)
    for v in degreepool.values():v.sort(key=key)
    seen=set();occupied=set();frames=[];rejected=Counter()
    def okay(p,local):return p not in seen and p not in local and not io[p]&occupied
    for q0 in sorted(g,key=key):
        if not okay(q0,set()):continue
        choices=defaultdict(list)
        for b in g[q0]:
            if okay(b,{q0}):choices[items[b]['kind']].append(b)
        cats=[k for k,v in choices.items() if len(v)>=2]
        if not cats:continue
        kind=min(cats,key=lambda x:h(f'{SEED}:{q0}:{x}'));bs=sorted(choices[kind],key=key)
        owned=[bs[0]];candidates=[bs[1]];contexts=[q0];local={q0,*owned,*candidates}
        for b in pools[kind]:
            if not okay(b,local):continue
            qs=[q for q in rev[b] if okay(q,local)]
            if not qs:continue
            q=min(qs,key=key);candidates.append(b);contexts.append(q);local.update([q,b])
            if len(contexts)==3:break
        if len(contexts)!=3:rejected['three-contexts']+=1;continue
        for degree in [len(rev[b]) for b in candidates]:
            controls=[b for b in degreepool[kind,degree] if okay(b,local)]
            if not controls:break
            b=controls[0];candidates.append(b);local.add(b)
        if len(candidates)!=6:rejected['degree-controls']+=1;continue
        rng=np.random.default_rng(int(h(f'{SEED}:order:{len(frames)}')[:12],16));candidates=rng.permutation(candidates).tolist();contexts=rng.permutation(contexts).tolist();bottoms=candidates+owned
        covered=set().union(*(rev[b] for b in candidates[4:]+owned));rich=set().union(*(rev[b] for b in candidates[:4]))-covered-set(contexts)
        rich=sorted([q for q in rich if okay(q,local)],key=key)[:4]
        if len(rich)!=4:rejected['rich-capacity']+=1;continue
        extras=sorted([q for q in g if q not in set(contexts+rich) and okay(q,local)],key=lambda p:(len(io[p]),h(f'{SEED}:extra:{len(frames)}:{p}')))[:4]
        if len(extras)!=4:rejected['extra-capacity']+=1;continue
        master=contexts+rich+extras;ix={q:i for i,q in enumerate(master)}
        views={n:dict(q=[ix[q] for q in qq],c=cc,w=ww) for n,qq,cc,ww in [
            ('basic',contexts,list(range(6)),[6]),('owned-basic',contexts,list(range(4)),[4,5,6]),
            ('no-owned',contexts,list(range(6)),[]),('rich',contexts+rich,list(range(6)),[6]),
            ('owned-rich',contexts+rich,list(range(4)),[4,5,6]),('low-frequency-contexts',contexts+extras,list(range(6)),[6]) ]}
        obs=np.asarray([[int(b in g[q]) for q in master] for b in bottoms]);v=views['owned-rich'];G=(obs[np.ix_(v['c'],v['q'])]*(1-obs[np.ix_(v['w'],v['q'])].max(0))).sum(1)
        if G.max()<2:rejected['multiple-gain']+=1;continue
        ids=set(master+bottoms);oo=set().union(*(io[p] for p in ids));assert not seen&ids and not occupied&oo
        frames.append(dict(id=f'asos-{len(frames):04}',contexts=master,bottoms=bottoms,views=views,observed=obs.tolist(),kind=kind,sourceOutfits=sorted(oo)))
        seen.update(ids);occupied.update(oo)
        if len(frames)>=500:break
    assert frames
    # Ensure frozen category mappings can find a retrieval pool in both engines.
    for folder,mapping in [(METHOD,targetmap),(SERVICE,servicemap)]:
        olditems=read(folder/'items.json');pools0={(olditems[r['top']]['sourceCategory'],olditems[r['top']]['kind'],olditems[r['bottom']]['sourceCategory'],olditems[r['bottom']]['kind']) for r in read(folder/'reference.json')}
        for c in frames:
            for q in c['contexts']:assert (mapping['top',items[q]['kind']],items[q]['kind'],mapping['bottom',c['kind']],c['kind']) in pools0
        assert not seen&set(olditems)
    OUT.mkdir();write(OUT/'frames.json',frames);write(OUT/'items.json',{p:items[p] for p in sorted(seen)})
    plan=dict(createdAt=datetime.now(timezone.utc).isoformat(),seed=SEED,
        source='https://osf.io/cg6eq/',sourceManifestSha256=sha(ASOS/'raw/manifest.json'),
        projectionHashes={n:sha(ASOS/'projected'/n) for n in ['items.json','edges.json','outfits.json','audit.json']},
        framesSha256=sha(OUT/'frames.json'),itemsSha256=sha(OUT/'items.json'),
        sourceModelFinalPlanSha256=sha(METHOD/'final-plan.json'),
        fullIncidenceSha256=sha(ASOS/'full-source-incidence.json'),sourceIncidenceAuditSha256=sha(ASOS/'source-incidence-audit.json'),
        sourceServiceHashes={n:sha(SERVICE/n) for n in ['calibration-lock.json','no-rag-selection.json','support-mixture-selection.json','training-pair-feature-lock.json','reference-pca.joblib','reference.json','encoder.json','items.json','vectors.npy','itemIds.json','pair-linear-model.joblib','pair-boosted-model.joblib','no-rag-pair-linear.joblib','no-rag-pair-boosted.joblib']},
        categoryMappings={label:{':'.join(k):v for k,v in m.items()} for label,m in [('experimental',targetmap),('service',servicemap)]},
        counts=dict(parents=len(frames),uniqueMasterGarments=len(seen),uniqueSourceOutfits=len(occupied),rejected=dict(rejected)),
        construction='Same six 3/7-top and6/4-candidate views as latest POG study. Three observed candidate seeds plus three exact connected-top-degree controls. Rich views permit at least two added observed links; added contexts chosen by low source frequency and fixed hash. Up to500 sequential greedy master-garment and whole-source-outfit-disjoint units. Not a random user wardrobe sample.',
        methods=METHODS,primaryContrasts=PRIMARY,
        selection='All POG weights, models, PCA, TFIDF and reference outfits frozen. NO ASOS fitting, calibration, development split, weight/threshold selection or retrieval reference. Only external product name encoded with the same fixed multilingual encoder; category mapping uses most common POG source category per fixed kind.',
        statistics='Parent-average NG across applicable views. 10000 paired parent bootstraps;50000 two-sided sign flips;Holm3 across prespecified contrasts. Whole-source disjoint parents support inference conditional on constructed cohort ifN>=20; otherwise descriptive only. Subgroups descriptive. Uniform random expectation contextual only.',
        decision='This is one new-source frozen-method replication. It does not overwrite any previous POG failure or adoption decision and cannot establish human aesthetic or purchase satisfaction. No retuning after predictions.',
        implementationSha256=sha(Path(__file__)))
    write(OUT/'protocol.json',plan);print('ASOS RESERVED',json.dumps(plan['counts']),flush=True)

def attach(e,label):
    import joblib
    from scipy.sparse import vstack
    from backend.pog_retrieval import text
    plan=read(OUT/'protocol.json');items=read(OUT/'items.json');ids=sorted(items);vectors=np.load(OUT/'external-vectors.npy')
    for p,x in items.items():x['sourceCategory']=plan['categoryMappings'][label][x['side']+':'+x['kind']]
    assert not set(ids)&set(e.ids)
    e.lex=vstack([e.lex,e.vectorizer.transform([text(items[p]) for p in ids])]).tocsr();e.vectors=np.vstack([e.vectors,vectors]);e.ids+=ids;e.ix={p:i for i,p in enumerate(e.ids)};e.items.update(items)
    folder=METHOD if label=='experimental' else SERVICE
    e.projection=joblib.load(folder/'reference-pca.joblib').transform(e.vectors)
    return e

def encode():
    if (OUT/'encoder-lock.json').exists():return
    import torch
    from sentence_transformers import SentenceTransformer
    from backend.pog_retrieval import text,MODEL,REVISION
    items=read(OUT/'items.json');ids=sorted(items);torch.set_num_threads(2);torch.manual_seed(SEED)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    m=SentenceTransformer(MODEL,revision=REVISION,device=device,local_files_only=True,trust_remote_code=False)
    v=m.encode([text(items[p]) for p in ids],normalize_embeddings=True,batch_size=128,show_progress_bar=True,convert_to_numpy=True)
    assert np.isfinite(v).all();np.save(OUT/'external-vectors.npy',v)
    write(OUT/'encoder-lock.json',dict(ids=ids,model=MODEL,revision=REVISION,dimension=v.shape[1],vectorsSha256=sha(OUT/'external-vectors.npy'),fields='Publisher name only; no description, IDs, images or ASOS outfit information.',refit=False))

def predict():
    if (OUT/'predictions.json').exists():raise RuntimeError('ASOS outcomes already opened')
    import joblib,torch
    from threadpoolctl import threadpool_limits
    from backend.conditional_case_utility import ConditionalRetriever
    from backend.learned_case_utility import LearnedSupportRetriever
    from backend.thesis_conditional_models_20261025 import novelty_inputs,marginal_estimates
    from backend.thesis_neural_pair_20261025 import PairNet
    from backend.thesis_neural_selection_20261025 import inputs as neural_inputs
    from backend.thesis_ensemble_20261025 import rankx
    threadpool_limits(1);torch.set_num_threads(2)
    plan=read(OUT/'protocol.json');assert sha(Path(__file__))==plan['implementationSha256']
    for n,k in [('frames.json','framesSha256'),('items.json','itemsSha256')]:assert sha(OUT/n)==plan[k]
    assert sha(ASOS/'raw/manifest.json')==plan['sourceManifestSha256']
    assert sha(ASOS/'full-source-incidence.json')==plan['fullIncidenceSha256']
    for n,v in plan['projectionHashes'].items():assert sha(ASOS/'projected'/n)==v,n
    oldplan=read(METHOD/'final-plan.json');assert sha(METHOD/'final-plan.json')==plan['sourceModelFinalPlanSha256']
    for n,v in oldplan['files'].items():assert sha(METHOD/n)==v,n
    for n,v in oldplan['implementationHashes'].items():assert sha(Path(__file__).parent/n)==v,n
    for n,v in plan['sourceServiceHashes'].items():assert sha(SERVICE/n)==v,n
    encode();enc=read(OUT/'encoder-lock.json');assert sha(OUT/'external-vectors.npy')==enc['vectorsSha256']
    e=attach(ConditionalRetriever(METHOD),'experimental');sv=attach(LearnedSupportRetriever(SERVICE),'service')
    selection=read(METHOD/'ensemble-selection.json');original=selection['componentProfiles']['RAG']['original'];residual=selection['componentProfiles']['RAG']['residual'];nr=selection['componentProfiles']['no-runtime-retrieval']['original']
    gmodel=joblib.load(METHOD/original['model']);nmodel=joblib.load(METHOD/nr['model']);control=read(METHOD/'control-selection.json')['profiles'];ranker=joblib.load(METHOD/control['learned-ranker']['config']['path']);nranker=joblib.load(METHOD/selection['neuralRanker']['model'])
    nnsel=read(METHOD/'neural-calibration-lock.json')['selected']['RAG'];ck=torch.load(METHOD/nnsel['neural'],map_location='cpu',weights_only=True);nn=PairNet(ck['kind']);nn.load_state_dict(ck['state']);nn.eval();calibrator=joblib.load(METHOD/nnsel['calibrator']);nw=read(METHOD/'neural-method-selection.json')['profiles']['RAG']['weights']
    parity_frames=read(METHOD/'test-frames.json')[:5];parity_n=len(parity_frames)
    expected={(r['parent'],r['view']):r for r in read(METHOD/'ensemble-test-predictions.json') if r['parent']<parity_n}
    aliases={'ensemble':'proposed','ensemble-C':'same-support-C','conditional':'conditional-original','product-residual':'residual-product','old-kernel-CM':'old-CM','direct-ranker':'learned-ranker','neural-direct-ranker':'neural-learned-ranker','no-runtime-retrieval':'no-runtime-retrieval'}
    parity_checks=0;parity_error=0.;rows=[];start=time.monotonic()
    for i,c in enumerate(parity_frames+read(OUT/'frames.json')):
        X,P=e.pair_features(c['contexts'],c['bottoms']);ids=[e.ix[p] for p in c['bottoms']];qs=[e.ix[p] for p in c['contexts']];lex=np.clip((e.lex[ids]@e.lex[ids].T).toarray(),0,1);dense=np.clip(e.vectors[ids]@e.vectors[ids].T,0,1)
        qv=torch.as_tensor(np.broadcast_to(e.vectors[qs][None,:,:],(len(ids),len(qs),384)).copy().reshape(-1,384));bv=torch.as_tensor(np.broadcast_to(e.vectors[ids][:,None,:],(len(ids),len(qs),384)).copy().reshape(-1,384))
        with torch.inference_mode():Z=nn(qv,bv).numpy().reshape(len(ids),len(qs))
        xx=np.concatenate([X,Z[...,None]],axis=-1);new=calibrator.predict_proba(xx.reshape(-1,140))[:,1].astype(np.float32).reshape(Z.shape);NP=np.concatenate([P,new[...,None]],axis=-1)
        service=sv.matrix(c['contexts'],c['bottoms']);obs=np.asarray(c['observed'])
        for view,v in c['views'].items():
            ci,wi,q=v['c'],v['w'],v['q'];prior=obs[np.ix_(wi,q)].max(0) if wi else np.zeros(len(q));G=(obs[np.ix_(ci,q)]*(1-prior)).sum(1)
            if not G.max():continue
            cg,mg,_=marginal_estimates(gmodel,X,P,ci,wi,q,lex[np.ix_(ci,wi)],dense[np.ix_(ci,wi)],original['supportWeights'])
            pr=P[ci][:,q]@np.asarray(residual['config']['weights']);op=P[wi][:,q]@np.asarray(residual['config']['weights']) if wi else np.zeros((0,len(q)));cp=pr.mean(1);mp=(pr*np.prod(1-op,axis=0)).mean(1)
            C=.25*cg+.75*cp;M=.25*mg+.75*mp
            cn,mn,_=marginal_estimates(nmodel,X,P,ci,wi,q,lex[np.ix_(ci,wi)],dense[np.ix_(ci,wi)],nr['supportWeights'],True)
            old=control['old-CM']['config'];k=old['kernel'];rk=X[ci][:,q,k];ok=X[wi][:,q,k].max(0) if wi else np.zeros(len(q));oldscore=old['alpha']*rk.mean(1)+(1-old['alpha'])*np.maximum(rk-ok,0).mean(1)
            ex=dict(X=novelty_inputs(X,P,ci,wi,q,lex[np.ix_(ci,wi)],dense[np.ix_(ci,wi)]),components=P[ci][:,q],ownedComponents=P[wi][:,q] if wi else np.zeros((0,len(q),4)))
            nx=dict(X=neural_inputs(X,Z,NP,ci,wi,q,lex,dense),components=NP[ci][:,q],ownedComponents=NP[wi][:,q] if wi else np.zeros((0,len(q),5)))
            sc=service[np.ix_(ci,q)];sp=service[np.ix_(wi,q)].max(0) if wi else np.zeros(len(q))
            scores={'ensemble':.3*C+.7*M,'ensemble-C':C,'conditional':original['alpha']*cg+(1-original['alpha'])*mg,'product-residual':mp,'old-kernel-CM':oldscore,'direct-ranker':ranker.predict(rankx(ex,original['supportWeights'])),'neural-direct-ranker':nranker.predict(rankx(nx,nw)),'no-runtime-retrieval':nr['alpha']*cn+(1-nr['alpha'])*mn,'service-M':np.maximum(sc-sp,0).mean(1),'service-C':sc.mean(1)}
            assert set(scores)==set(METHODS)
            if i<parity_n:
                oldrow=expected[i,view];assert G.tolist()==oldrow['gain']
                for name,oldname in aliases.items():
                    err=float(np.max(np.abs(scores[name]-oldrow['scores'][oldname])));parity_error=max(parity_error,err)
                    assert err<1e-6,(i,view,name,err)
                    assert abs(metric.quality(scores[name],G)-oldrow['NG'][oldname])<1e-12,(i,view,name,'selection mismatch')
                    parity_checks+=len(G)
            else:
                rows.append(dict(parent=i-parity_n,id=c['id'],view=view,gain=G.tolist(),C=C.tolist(),M=M.tolist(),scores={k:z.tolist() for k,z in scores.items()},NG={k:metric.quality(z,G) for k,z in scores.items()}))
        e.cache.clear();sv.cache.clear()
        if i==parity_n-1:
            write(OUT/'implementation-parity-before-ASOS.json',dict(allPassed=True,POGParents=parity_n,scalarScoreChecks=parity_checks,maxAbsoluteError=parity_error,selectionAndNGIdentical=True,ASOSPredictionsOpened=False))
            print('Pre-ASOS implementation parity',parity_checks,parity_error,flush=True)
        if i>=parity_n and (i-parity_n)%10==0:print('ASOS frozen predict',i-parity_n+1,'elapsed',round(time.monotonic()-start,1),flush=True)
    write(OUT/'predictions.json',rows);write(OUT/'prediction-lock.json',dict(protocolSha256=sha(OUT/'protocol.json'),predictionsSha256=sha(OUT/'predictions.json'),createdAt=datetime.now(timezone.utc).isoformat()))

def report():
    from backend.thesis_conditional_final_20261025 import interval
    rows=read(OUT/'predictions.json');parents=defaultdict(list)
    for r in rows:parents[r['parent']].append(r)
    values={k:np.asarray([np.mean([r['NG'][k] for r in rr]) for rr in parents.values()]) for k in METHODS};contrasts={}
    for a,b in PRIMARY:
        d=values[a]-values[b];rr=dict(mean=float(d.mean()),better=int((d>1e-12).sum()),same=int((np.abs(d)<=1e-12).sum()),worse=int((d< -1e-12).sum()))
        if len(d)>=20:
            rr.update(interval(d));rng=np.random.default_rng(SEED);count=0
            for _ in range(500):count+=np.count_nonzero(np.abs((d*rng.choice([-1.,1.],(100,len(d)))).mean(1))>=abs(d.mean())-1e-14)
            rr['p']=(count+1)/50001
        contrasts[a+' vs '+b]=rr
    if len(parents)>=20:
        last=0
        for j,k in enumerate(sorted(contrasts,key=lambda k:contrasts[k]['p'])):last=max(last,min(1.,contrasts[k]['p']*(3-j)));contrasts[k]['holmP']=last
    expected=float(np.mean([np.mean([np.mean(r['gain'])/max(r['gain']) for r in rr]) for rr in parents.values()]))
    result=dict(parents=len(parents),views=len(rows),rates={k:interval(v) for k,v in values.items()},contrasts=contrasts,
        uniformSelectionExpectedNG=expected,
        byView={v:dict(n=sum(r['view']==v for r in rows),rates={k:float(np.mean([r['NG'][k] for r in rows if r['view']==v])) for k in METHODS}) for v in sorted({r['view'] for r in rows})},
        inferenceAvailable=len(parents)>=20,protocolSha256=sha(OUT/'protocol.json'),predictionsSha256=sha(OUT/'predictions.json'),
        priorPOGAdoptionDecisionUnchanged=True,ASOSModelFitting=False,
        scope='One prespecified whole-source-disjoint constructed-cohort zero-shot ASOS replication. No ASOS co-outfit data in retrieval/model/selection. No human aesthetic or buying satisfaction outcome.')
    write(OUT/'report.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','predict','report']);a=p.parse_args();globals()[a.stage]()
