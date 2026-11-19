"""Fresh cold-candidate test and mixed-context development for C/M weights.

The v2 single-gain test succeeded, but the frozen multi-gain diagnostic showed
that M-only hurts when several added links are available. This new procedure
learns a bounded C/M decision weight on training/development variants, and
reserves NEW candidate/owned garments for its final test. Old results remain.
"""
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import argparse,json,os,time
import numpy as np
from backend.thesis_study_20261023 import ROOT,PARENT,source,read,write,sha,h
from backend.thesis_upgrade_20261024 import OUT as OLD

OUT=ROOT/'adaptive-policy-20261024';SEED=2026102405

def frame(c,items,g,rev,io,pools,extra_filter=lambda q:True):
    cat=c['category'];original=list(c['contexts']);bottoms=c['candidates']+c['owned'];covered=set().union(*(rev[b] for b in c['owned']+c['candidates'][6:]))
    rich=set().union(*(rev[b] for b in c['candidates'][:6]))-set(original)-covered
    valid=lambda q:(items[q]['sourceCategory'],items[q]['kind'],*cat) in pools and extra_filter(q)
    rich=sorted([q for q in rich if valid(q)],key=lambda q:(len(io[q]),h(f'{SEED}:rich:{c["id"]}:{q}')))[:4]
    random_pool=sorted([q for q in items if items[q]['side']=='top' and q not in set(original+rich) and valid(q)],
        key=lambda q:(len(io.get(q,())),h(f'{SEED}:random:{c["id"]}:{q}')))
    random_extra=random_pool[:4];master=original+rich+random_extra
    rng=np.random.default_rng(int(h(f'{SEED}:context-order:{c["id"]}')[:12],16));master=rng.permutation(master).tolist()
    index={q:i for i,q in enumerate(master)};views={
        'basic':dict(q=[index[q] for q in original],c=list(range(8)),w=[8]),
        'owned-basic':dict(q=[index[q] for q in original],c=list(range(6)),w=[6,7,8]),
        'no-owned':dict(q=[index[q] for q in original],c=list(range(8)),w=[])}
    if len(rich)==4:
        views['rich']=dict(q=[index[q] for q in original+rich],c=list(range(8)),w=[8])
        views['owned-rich']=dict(q=[index[q] for q in original+rich],c=list(range(6)),w=[6,7,8])
    if len(random_extra)==4:views['random-contexts']=dict(q=[index[q] for q in original+random_extra],c=list(range(8)),w=[8])
    obs=np.array([[int(b in g[q]) for q in master] for b in bottoms])
    return dict(id=c['id'],contexts=master,bottoms=bottoms,category=cat,views=views,observed=obs.tolist(),
                originalContextIndices=[index[q] for q in original],degrees=[len(rev[b]) for b in c['candidates']])

def prepare():
    if OUT.exists():raise RuntimeError('Immutable adaptive study already exists')
    items,edges,g,rev,_,_=source();io={q:set(oo) for q,oo in read(ROOT/'raw-outfit-incidence-audit-20261023/item-outfits.json').items()}
    eo={tuple(k.split(':')):set(oo) for k,oo in read(ROOT/'adversarial-audit-20261023/source-edge-outfits.json').items()}
    oldinputs=sum([read(OLD/f'{stage}-inputs.json') for stage in ['training','development','test']],[])
    oldused={p for c in oldinputs for p in c['contexts']+c['candidates']+c['owned']};oldoutfits=set().union(*(io[p] for p in oldused))
    oldref=read(OLD/'reference.json');pools={(items[r['top']]['sourceCategory'],items[r['top']]['kind'],items[r['bottom']]['sourceCategory'],items[r['bottom']]['kind']) for r in oldref}
    pool=defaultdict(list);key=lambda p:(len(io[p]),h(f'{SEED}:{p}'))
    for b in rev:
        if b not in oldused and not io[b]&oldoutfits:pool[(items[b]['sourceCategory'],items[b]['kind'])].append(b)
    for cat in pool:pool[cat].sort(key=key)
    testused=set();occupied=set();cases=[];rejects=Counter()
    def bottom_ok(p,local):return p not in oldused and p not in testused and p not in local and not io[p]&oldoutfits and not io[p]&occupied
    def top_ok(p,local):return p not in oldused and p not in testused and p not in local and not io[p]&occupied
    for q0 in sorted(g,key=key):
        if len(cases)>=400:break
        if not top_ok(q0,set()):continue
        cats=defaultdict(list)
        for b in g[q0]:
            if bottom_ok(b,{q0}):cats[(items[b]['sourceCategory'],items[b]['kind'])].append(b)
        cats=[cat for cat,bs in cats.items() if len(bs)>=2]
        if not cats:continue
        cat=min(cats,key=lambda cat:h(f'{SEED}:{q0}:{cat}'));bs=sorted([b for b in g[q0] if (items[b]['sourceCategory'],items[b]['kind'])==cat and bottom_ok(b,{q0})],key=key)
        owned=[bs[0]];candidates=[bs[1]];contexts=[q0];local={q0,*owned,*candidates}
        for b in pool[cat]:
            if not bottom_ok(b,local):continue
            qs=[q for q in rev[b] if top_ok(q,local) and (items[q]['sourceCategory'],items[q]['kind'],*cat) in pools]
            if not qs:continue
            q=min(qs,key=key);candidates.append(b);contexts.append(q);local.update((q,b))
            if len(contexts)==4:break
        if len(contexts)!=4:rejects['four-contexts']+=1;continue
        degrees=[len(rev[b]) for b in candidates]
        for degree in degrees:
            controls=[b for b in pool[cat] if len(rev[b])==degree and bottom_ok(b,local)]
            if not controls:break
            b=min(controls,key=lambda b:h(f'{SEED}:control:{len(cases)}:{len(candidates)}:{b}'));candidates.append(b);local.add(b)
        if len(candidates)!=8:rejects['degree-control']+=1;continue
        rng=np.random.default_rng(int(h(f'{SEED}:candidate-order:{len(cases)}')[:12],16));candidates=rng.permutation(candidates).tolist();contexts=rng.permutation(contexts).tolist()
        raw=dict(id=f'fresh-{len(cases):04}',candidates=candidates,owned=owned,contexts=contexts,category=list(cat))
        f=frame(raw,items,g,rev,io,pools,extra_filter=lambda q:top_ok(q,local))
        if len(f['views'])!=6:rejects['mixed-context-capacity']+=1;continue
        v=f['views']['owned-rich'];obs=np.asarray(f['observed']);gain=(obs[np.ix_(v['c'],v['q'])]*(1-obs[np.ix_(v['w'],v['q'])].max(0))).sum(1)
        if gain.max()<2:rejects['multi-gain-eligibility']+=1;continue
        allids=set(f['bottoms']+f['contexts']);oo=set().union(*(io[p] for p in allids));assert not occupied&oo
        cases.append(f);testused.update(allids);occupied.update(oo)
    testbottoms={p for c in cases for p in c['bottoms']};testbottomout=set().union(*(io[p] for p in testbottoms)) if testbottoms else set()
    refs=[r for r in oldref if r['top'] not in testbottoms and r['bottom'] not in testbottoms and not eo[(r['top'],r['bottom'])]&testbottomout]
    newpools={(items[r['top']]['sourceCategory'],items[r['top']]['kind'],items[r['bottom']]['sourceCategory'],items[r['bottom']]['kind']) for r in refs}
    eligible=[c for c in cases if all((items[q]['sourceCategory'],items[q]['kind'],*c['category']) in newpools for q in c['contexts'])]
    OUT.mkdir()
    for name in ['items.json','vectors.npy','itemIds.json','encoder.json']:os.link(PARENT/name,OUT/name)
    write(OUT/'reference.json',refs);write(OUT/'test-frames.json',eligible)
    for stage in ['training','development']:
        inputs=read(OLD/f'{stage}-inputs.json');frames=[frame(c,items,g,rev,io,newpools) for c in inputs]
        write(OUT/f'{stage}-frames.json',frames)
    protocol=dict(createdAt=datetime.now(timezone.utc).isoformat(),seed=SEED,files={n:sha(OUT/n) for n in ['items.json','reference.json']},
        counts=dict(training=457,development=229,testConstructed=len(cases),testEligible=len(eligible),referencePairs=len(refs),freshBottomPool=sum(map(len,pool.values()))),
        rejects=dict(rejects),frameHashes={stage:sha(OUT/f'{stage}-frames.json') for stage in ['training','development','test']},
        sourceBoundary='NEW final candidate and owned bottom IDs and all their source outfits are absent from every older cohort and from reference. Context tops may be known in reference through other outfits; full master input source outfits are disjoint between final cases.',
        development='Six named views per training/development parent when structurally available. Original four contexts, added source-related contexts, random added contexts, one/three/zero existing bottoms. No view name/labels/IDs enter policy features.',
        supervision='Refit original fixed pair models on 457 original training wardrobes after removing fresh candidates from reference. Generate training support using three-fold out-of-fold models. Fit gate only on training outcomes; select on development parent-average normalized gain.',
        primary='Paired parent-average normalized additional gain across six fixed views; no-gain views counted separately, not turned into artificial successes. Bootstrap parents, never individual related views.',
        modelSelection='Convex C/M gate using wardrobe support summaries; regularized logistic gate or small boosted cost-sensitive gate, and fixed-alpha baselines. Selection budget recorded before final evaluation.',
        finalOpening='Write policy-selection.json and final-plan.json before reading final outcome metrics. One final evaluation; no change based on that evaluation.',
        previousFindings='V2 single-gain 458 improved, but rich 315 showed M-only below C. Those are diagnostic research history and are never called untouched validation of this revision.',codeSha256=sha(__file__))
    write(OUT/'protocol.json',protocol);(OUT/'frozen-adaptive-preparation.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(protocol['counts']),flush=True)

def features(stage):
    import joblib
    from sklearn.decomposition import PCA
    from threadpoolctl import threadpool_limits
    from backend.pog_refined_retrieval import Retriever
    from backend.learned_case_utility import CONFIGS
    threadpool_limits(1)
    if stage=='test' and not (OUT/'final-plan.json').exists():raise RuntimeError('Fresh final test is sealed')
    dest=OUT/f'{stage}-pair-features.npz'
    if dest.exists():raise RuntimeError('Pair features already exist')
    engine=Retriever(OUT);pp=OUT/'reference-pca.joblib'
    if not pp.exists():
        if stage!='training':raise RuntimeError('Fit projection on reference first')
        refids=sorted({r[s] for r in engine.records for s in ['top','bottom']})
        pca=PCA(n_components=32,svd_solver='randomized',random_state=SEED)
        pca.fit(engine.vectors[[engine.ix[p] for p in refids]]);joblib.dump(pca,pp)
    projection=joblib.load(pp).transform(engine.vectors);rows=read(OUT/f'{stage}-frames.json');data=[];excluded=[];start=time.monotonic()
    for i,c in enumerate(rows):
        if any((engine.items[q]['sourceCategory'],engine.items[q]['kind'],*c['category']) not in engine.pools for q in c['contexts']):
            excluded.append(dict(id=c['id'],reason='Unsupported reference category, determined before predictions'))
            data.append(np.zeros((9,12,139),dtype=np.float32));continue
        contexts=c['contexts'];bottoms=c['bottoms'];qs=[engine.ix[p] for p in contexts];cs=[engine.ix[p] for p in bottoms];nq=len(qs)
        matrices=[engine.matrix(contexts,bottoms,cfg) for cfg in CONFIGS]
        lex=(engine.lex[cs]@engine.lex[qs].T).toarray();dense=engine.vectors[cs]@engine.vectors[qs].T
        q=np.broadcast_to(projection[qs][None,:,:],(9,nq,32));b=np.broadcast_to(projection[cs][:,None,:],(9,nq,32))
        x=np.concatenate([np.stack(matrices+[lex,dense],axis=-1),q,b,np.abs(q-b),q*b],axis=-1)
        padded=np.zeros((9,12,139),dtype=np.float32);padded[:,:nq]=x;data.append(padded);engine.cache.clear()
        if i%50==0 or i+1==len(rows):print('features',stage,i+1,'/',len(rows),'seconds',round(time.monotonic()-start,1),flush=True)
    np.savez_compressed(dest,X=np.asarray(data))
    write(OUT/f'{stage}-pair-feature-lock.json',dict(featureSha256=sha(dest),pcaSha256=sha(pp),excluded=excluded,codeSha256=sha(__file__)))

def fit_support():
    import joblib
    from backend.thesis_calibrated_upgrade_20261024 import new_pair_model
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    if (OUT/'support-lock.json').exists() or (OUT/'test-pair-features.npz').exists():raise RuntimeError('Fit stage is sealed')
    rows=read(OUT/'training-frames.json');dev=read(OUT/'development-frames.json')
    tr=np.load(OUT/'training-pair-features.npz')['X'];dv=np.load(OUT/'development-pair-features.npz')['X']
    train=np.asarray([tr[i][:,c['originalContextIndices']] for i,c in enumerate(rows)])
    y=np.asarray([np.asarray(c['observed'])[:,c['originalContextIndices']] for c in rows])
    fold=np.array([int(h(f'{SEED}:pair-fold:{c["id"]}')[:12],16)%3 for c in rows]);trainpred=[];devpred=[];models={}
    for source,sl in [('retrieval',slice(None)),('text',slice(9,None))]:
        for family in ['linear','boosted']:
            oof=np.zeros(tr.shape[:-1],dtype=np.float32);z=train[...,sl];x=tr[...,sl];dx=dv[...,sl]
            for f in range(3):
                model=new_pair_model(family);keep=fold!=f
                model.fit(z[keep].reshape(-1,z.shape[-1]),y[keep].ravel())
                oof[~keep]=model.predict_proba(x[~keep].reshape(-1,x.shape[-1]))[:,1].reshape(x[~keep].shape[:-1])
                print('OOF',source,family,f,int((~keep).sum()),flush=True)
            model=new_pair_model(family);model.fit(z.reshape(-1,z.shape[-1]),y.ravel())
            name=f'{source}-{family}.joblib';joblib.dump(model,OUT/name);models[name]=sha(OUT/name)
            trainpred.append(oof);devpred.append(model.predict_proba(dx.reshape(-1,dx.shape[-1]))[:,1].reshape(dx.shape[:-1]).astype(np.float32))
    for stage,ps in [('training',trainpred),('development',devpred)]:np.savez_compressed(OUT/f'{stage}-probabilities.npz',P=np.stack(ps,axis=-1))
    write(OUT/'support-lock.json',dict(createdAt=datetime.now(timezone.utc).isoformat(),models=models,folds=fold.tolist(),trainingOOF=True,
        weights=[.25,0,.50,.25],features='139: nine retrieved support kernels, direct lexical/dense title similarities, reference-only PCA32 of top/bottom/absolute difference/product. Text-only130 omits nine kernels.',
        fitPairs=int(y.size),fitObservedLinks=int(y.sum()),codeSha256=sha(__file__)))

def policy_features(p,owned_count):
    """Permutation-invariant input-only summaries; also used by serving."""
    n=len(p)-owned_count;s=p[:n];old=p[n:].max(0) if owned_count else np.zeros(p.shape[1]);d=np.maximum(s-old,0)
    C=s.mean(1);M=d.mean(1);R=(s*(1-old)).mean(1);eps=1e-8
    vals=[p.shape[1],owned_count,n]
    for z in [old,C,M,M/np.maximum(C,eps),s.max(0),s.std(0),d.max(0)]:
        vals.extend([z.mean(),z.std(),z.min(),z.max()])
    ci=np.isclose(C,C.max(),atol=1e-12,rtol=0);mi=np.isclose(M,M.max(),atol=1e-12,rtol=0)
    vals.extend([float(np.any(ci&mi)),C[mi].mean(),M[ci].mean(),C.max()-C[mi].mean(),M.max()-M[ci].mean(),
                 float(np.corrcoef(C,M)[0,1]) if C.std()>eps and M.std()>eps else 0,
                 np.sort(C)[-1]-np.sort(C)[-2] if n>1 else 0,np.sort(M)[-1]-np.sort(M)[-2] if n>1 else 0,
                 (s>old).mean(),np.mean(s*(1-s))])
    return np.nan_to_num(np.asarray(vals,dtype=np.float64)),C,M,R

def examples(stage,P=None):
    if P is None:P=np.load(OUT/f'{stage}-probabilities.npz')['P']
    rows=read(OUT/f'{stage}-frames.json');excluded={r['id'] for r in read(OUT/f'{stage}-pair-feature-lock.json')['excluded']};result=[]
    for i,c in enumerate(rows):
        if c['id'] in excluded:continue
        obs=np.asarray(c['observed'])
        for name,v in c['views'].items():
            ix=v['c']+v['w'];prob=P[i][ix][:,v['q']];s=prob@np.array([.25,0,.5,.25]);z,C,M,R=policy_features(s,len(v['w']))
            history=obs[np.ix_(v['w'],v['q'])].max(0) if v['w'] else np.zeros(len(v['q']))
            gains=(obs[np.ix_(v['c'],v['q'])]*(1-history)).sum(1)
            if not gains.max():continue
            result.append(dict(parent=i,id=c['id'],view=name,z=z,C=C,M=M,R=R,gain=gains,support=prob,rawFeaturesIndex=(ix,v['q'])))
    return result

def quality(s,g):
    best=np.isclose(s,np.max(s),atol=1e-12,rtol=0)
    return float(np.mean(g[best])/np.max(g))

def parent_quality(scores,es):
    groups=defaultdict(list)
    for s,e in zip(scores,es):groups[e['parent']].append(quality(s,e['gain']))
    return np.asarray([np.mean(groups[k]) for k in sorted(groups)])

def gate_alpha(model,kind,z):
    if kind=='constant':return np.full(len(z),float(model))
    if kind=='logistic' or kind=='boosted':return model.predict_proba(z)[:,1]
    if kind=='hard-logistic' or kind=='hard-boosted':return (model.predict_proba(z)[:,1]>=.5).astype(float)
    if kind=='pairwise':
        from scipy.special import expit
        return expit(model['scaler'].transform(z)@model['beta']+model['bias'])
    raise ValueError(kind)

def develop():
    import joblib
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    if (OUT/'policy-selection.json').exists() or (OUT/'test-pair-features.npz').exists():raise RuntimeError('Policy search is sealed')
    tr=examples('training');dv=examples('development');tz=np.asarray([e['z'] for e in tr]);dz=np.asarray([e['z'] for e in dv]);candidates=[];search=[]
    def record(name,kind,model):
        alpha=gate_alpha(model,kind,dz);scores=[a*e['C']+(1-a)*e['M'] for a,e in zip(alpha,dv)]
        pq=parent_quality(scores,dv);byview={v:float(np.mean([quality(s,e['gain']) for s,e in zip(scores,dv) if e['view']==v])) for v in sorted({e['view'] for e in dv})}
        row=dict(name=name,kind=kind,developmentQuality=float(pq.mean()),byView=byview,alphaMean=float(alpha.mean()))
        search.append(row);candidates.append((row,model));print('policy',json.dumps(row),flush=True)
    for alpha in [0,.05,.1,.2,.3,.4,.5,.6,.7,.8,.9,.95,1]:record(f'constant-{alpha}','constant',alpha)
    yc=np.array([quality(e['C'],e['gain']) for e in tr]);ym=np.array([quality(e['M'],e['gain']) for e in tr]);use=np.abs(yc-ym)>1e-12
    parentcounts=Counter(e['parent'] for e in tr);weight=np.array([abs(a-b)/parentcounts[e['parent']] for a,b,e in zip(yc,ym,tr)])
    for reg in [.01,.1,1,10]:
        m=make_pipeline(StandardScaler(),LogisticRegression(C=reg,solver='lbfgs',max_iter=1000,random_state=SEED))
        m.fit(tz[use],(yc[use]>ym[use]).astype(int),logisticregression__sample_weight=weight[use]*6)
        for kind in ['logistic','hard-logistic']:record(f'{kind}-{reg}',kind,m)
    for leaves in [3,7]:
        for reg in [1,10]:
            m=HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=leaves,min_samples_leaf=30,l2_regularization=reg,learning_rate=.04,early_stopping=False,random_state=SEED)
            m.fit(tz[use],(yc[use]>ym[use]).astype(int),sample_weight=weight[use]*6)
            for kind in ['boosted','hard-boosted']:record(f'{kind}-leaf{leaves}-reg{reg}',kind,m)
    # Directly optimize the convex score, rather than only C-vs-M endpoint labels.
    from scipy.optimize import minimize
    from scipy.special import expit
    scaler=StandardScaler().fit(tz);zz=scaler.transform(tz);zz=np.column_stack([zz,np.ones(len(zz))]);pairs=[]
    for i,e in enumerate(tr):
        g=e['gain']/max(e['gain']);C=e['C'];M=e['M']
        for a in range(len(g)):
            for b in range(len(g)):
                if g[a]>g[b]:pairs.append((i,C[a]-C[b],M[a]-M[b],(g[a]-g[b])/parentcounts[e['parent']]))
    idx=np.array([a[0] for a in pairs]);dc=np.array([a[1] for a in pairs]);dm=np.array([a[2] for a in pairs]);pw=np.array([a[3] for a in pairs]);pw/=pw.sum()
    for reg in [.001,.01,.1,1]:
        def loss(beta):
            a=expit(zz@beta);diff=(a[idx]*dc+(1-a[idx])*dm)/.03
            val=(pw*np.logaddexp(0,-diff)).sum()+reg*np.dot(beta[:-1],beta[:-1])/2
            err=-pw*expit(-diff)*(dc-dm)/.03*a[idx]*(1-a[idx]);gr=zz.T@np.bincount(idx,weights=err,minlength=len(zz));gr[:-1]+=reg*beta[:-1]
            return val,gr
        fit=minimize(loss,np.zeros(zz.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':500})
        m=dict(scaler=scaler,beta=fit.x[:-1],bias=fit.x[-1]);record(f'pairwise-{reg}','pairwise',m)
    row,model=max(candidates,key=lambda r:(round(r[0]['developmentQuality'],12),r[0]['kind']=='constant',-len(r[0]['name'])))
    constant=max((r for r in candidates if r[0]['kind']=='constant'),key=lambda r:r[0]['developmentQuality'])
    joblib.dump(model,OUT/'policy-model.joblib')
    write(OUT/'policy-selection.json',dict(createdAt=datetime.now(timezone.utc).isoformat(),selected=row,bestConstant=constant[0],search=search,
        modelSha256=sha(OUT/'policy-model.joblib'),supportLockSha256=sha(OUT/'support-lock.json'),trainingParents=len({e['parent'] for e in tr}),developmentParents=len({e['parent'] for e in dv}),
        trainingViews=len(tr),developmentViews=len(dv),inputFeatures=41,selection='Maximize development mean of per-parent mean normalized gain; equal parent weight. Gate uses input support summaries only, never source labels/view names/IDs/degrees.',codeSha256=sha(__file__)))
    print('SELECTED',json.dumps(row),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','features','fit-support','develop']);p.add_argument('--split',choices=['training','development','test']);a=p.parse_args()
    if a.stage=='features':features(a.split)
    else:globals()[a.stage.replace('-','_')]()
