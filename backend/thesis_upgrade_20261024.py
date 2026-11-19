"""Prospective, immutable development/test protocol for the upgraded ranker.

This is a new internal POG study, not an external replication. Previously read
POG labels are acknowledged. Source-outfit-disjoint units are constructed
before any predictions. Development may guide engineering; the sealed final
test may be opened only after a model-selection lock has been written.
"""
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
import argparse, json, os, time
import numpy as np
from backend.thesis_study_20261023 import ROOT, PARENT, read, write, sha, h, source

OUT=ROOT/'prospective-upgrade-20261024'
SEED=2026102401
CONFIGS=[dict(mode=m,k=k) for m in ['char-density','hybrid-density','hybrid-conditional'] for k in [8,32,96]]

def prepare():
    if OUT.exists(): raise RuntimeError('Immutable protocol already exists')
    items,edges,g,rev,_,_=source()
    incidence={p:set(oo) for p,oo in read(ROOT/'raw-outfit-incidence-audit-20261023/item-outfits.json').items()}
    eo={tuple(k.split(':')):set(v) for k,v in read(ROOT/'adversarial-audit-20261023/source-edge-outfits.json').items()}
    pools=defaultdict(list)
    for b in rev: pools[(items[b]['sourceCategory'],items[b]['kind'])].append(b)
    # Structural packing reduces dependence, without consulting any model or
    # method difference. Low-incidence sampling is explicitly part of scope.
    key=lambda p:(len(incidence.get(p,())),h(f'{SEED}:{p}'))
    for cat in pools:pools[cat].sort(key=key)
    used=set();occupied=set();blocks=[];reject=Counter()
    def available(p,local):return p not in used and p not in local and not (incidence.get(p,set()) & occupied)
    anchors=sorted(g,key=key)
    for t0 in anchors:
        if not available(t0,set()):continue
        cats=defaultdict(list)
        for b in g[t0]:
            if available(b,{t0}):cats[(items[b]['sourceCategory'],items[b]['kind'])].append(b)
        cats=[cat for cat,bs in cats.items() if len(bs)>=2]
        if not cats:continue
        cat=min(cats,key=lambda c:h(f'{SEED}:{t0}:{c}'))
        bs=sorted([b for b in g[t0] if (items[b]['sourceCategory'],items[b]['kind'])==cat and available(b,{t0})],key=key)
        contexts=[t0];owned=[bs[0]];candidates=[bs[1]];local={t0,*owned,*candidates}
        for b in pools[cat]:
            if not available(b,local):continue
            ts=[t for t in rev[b] if available(t,local)]
            if not ts:continue
            t=min(ts,key=key);contexts.append(t);candidates.append(b);local.update((t,b))
            if len(contexts)==4:break
        if len(contexts)!=4:reject['anchor-unavailable']+=1;continue
        anchor_degrees=[len(rev[b]) for b in candidates]
        for degree in anchor_degrees:
            controls=[b for b in pools[cat] if len(rev[b])==degree and available(b,local)]
            if not controls:break
            b=min(controls,key=lambda p:h(f'{SEED}:control:{len(blocks)}:{p}'))
            candidates.append(b);local.add(b)
        if len(candidates)!=8:reject['exact-degree-control-unavailable']+=1;continue
        rng=np.random.default_rng(int(h(f'{SEED}:order:{len(blocks)}')[:12],16))
        contexts=rng.permutation(contexts).tolist();candidates=rng.permutation(candidates).tolist()
        obs=np.array([[int((t,b) in edges) for t in contexts] for b in candidates])
        hist=np.array([[int((t,b) in edges) for t in contexts] for b in owned])
        gain=(obs*(1-hist.max(0))).sum(1)
        if gain.max()==0:reject['zero-additional-observation']+=1;continue
        oo=set().union(*(incidence[p] for p in local))
        assert not occupied&oo
        blocks.append(dict(id=f'packed-{len(blocks):04}',contexts=contexts,owned=owned,candidates=candidates,
                           observed=obs.tolist(),history=hist.tolist(),category=list(cat),degrees=[len(rev[b]) for b in candidates]))
        occupied.update(oo);used.update(local)
    refs=[dict(top=t,bottom=b) for t,b in sorted(edges) if t not in used and b not in used and not eo[(t,b)]&occupied]
    validpools={(items[r['top']]['sourceCategory'],items[r['top']]['kind'],items[r['bottom']]['sourceCategory'],items[r['bottom']]['kind']) for r in refs}
    eligible=[]
    for c in blocks:
        cat=c['category']
        if all((items[t]['sourceCategory'],items[t]['kind'],*cat) in validpools for t in c['contexts']):eligible.append(c)
        else:reject['unsupported-reference-pool']+=1
    # Randomized split is fixed after structural eligibility, before scoring.
    order=sorted(eligible,key=lambda c:h(f'{SEED}:split:{c["id"]}'))
    n=len(order);a=int(.4*n);b=int(.6*n)
    cohorts={'training':order[:a],'development':order[a:b],'test':order[b:]}
    OUT.mkdir()
    for name in ['items.json','vectors.npy','itemIds.json','encoder.json']:os.link(PARENT/name,OUT/name)
    write(OUT/'reference.json',refs)
    for stage,rows in cohorts.items():
        write(OUT/f'{stage}-inputs.json',[{k:c[k] for k in ['id','contexts','owned','candidates','category']} for c in rows])
        write(OUT/f'{stage}-targets.json',[{k:c[k] for k in ['id','observed','history','degrees']} for c in rows])
    protocol=dict(createdAt=datetime.now(timezone.utc).isoformat(),seed=SEED,files={n:sha(OUT/n) for n in ['items.json','reference.json']},
                  cohortHashes={p.name:sha(p) for p in OUT.glob('*-*.json')},rawIncidenceSha256=sha(ROOT/'raw-outfit-incidence-audit-20261023/item-outfits.json'),
                  sourceScope='New internal post-audit experiment on previously inspected POG. No external, newly collected or untouched-source claim.',
                  sampling='Greedy low-incidence structural packing, four tops, one existing bottom, eight candidate bottoms. Four linked anchors and four exact-full-degree matched controls; all positions shuffled. Cohorts and inference units share no raw source outfit.',
                  counts={k:len(v) for k,v in cohorts.items()},allPacked=len(blocks),eligible=n,referencePairs=len(refs),rejects=dict(reject),
                  learning='Training labels fit a small feature-based reranker. Development selects models and simple baseline profiles. Test features/targets unopened until selection.json exists. All data retrieval excludes all cohort garments and incident source outfits.',
                  primary='Normalized additional observed-link gain at top-1; ties integrated exactly. Paired 10000 bootstrap and 50000 sign flips; Holm across all predeclared primary contrasts.',
                  primaryComparators=['same-C','old-M','char-novelty','dense-novelty','residual','learned-without-M'],
                  secondary='Top-1 maximizing gain, tied NDCG, no-owned M=C, duplicate/ordering invariance. Fixed post-audit old benchmark retained as diagnostic, never pooled.',
                  decision='One final test opening, no changing sample/filter/model after test. Every locked contrast reported. No stopping when p becomes significant.',codeSha256=sha(__file__))
    write(OUT/'protocol.json',protocol);(OUT/'frozen-preparation.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:protocol[k] for k in ['counts','allPacked','eligible','referencePairs','rejects']},ensure_ascii=False),flush=True)

def targets(stage):
    rows=read(OUT/f'{stage}-targets.json');gains=[]
    for c in rows:
        o=np.asarray(c['observed']);hist=np.asarray(c['history']);g=(o*(1-hist.max(0))).sum(1)
        assert g.max()>0;gains.append(g)
    return np.array(gains)

def features(stage):
    from backend.pog_refined_retrieval import Retriever
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    if stage=='test' and not (OUT/'final-plan.json').exists():raise RuntimeError('Final test remains sealed')
    path=OUT/f'{stage}-features.npz'
    if path.exists():raise RuntimeError('Frozen features exist')
    rows=read(OUT/f'{stage}-inputs.json');engine=Retriever(OUT);start=time.monotonic()
    names=[];features=[];baseline=[];without=[]
    for i,c in enumerate(rows):
        X=[];bs=[];wi=[]
        ci=[engine.ix[p] for p in c['candidates']];oi=[engine.ix[p] for p in c['owned']]
        lex=np.clip((engine.lex[ci]@engine.lex[oi].T).toarray().max(1),0,1)
        dense=np.clip((engine.vectors[ci]@engine.vectors[oi].T).max(1),0,1)
        for cfg in CONFIGS:
            a=engine.matrix(c['contexts'],c['candidates']+c['owned'],cfg)
            s=a[:8];prior=a[8:].max(0);d=s-prior;C=s.mean(1);M=np.maximum(d,0).mean(1)
            R=(s*(1-prior)).mean(1)
            values={'C':C,'M':M,'residual':R,'char-novelty':C*(1-lex),'dense-novelty':C*(1-dense),
                    'maxSupport':s.max(1),'minSupport':s.min(1),'stdSupport':s.std(1),
                    'positiveFraction':(d>0).mean(1),'maxIncrease':np.maximum(d,0).max(1),'relativeM':M/np.maximum(C,1e-8)}
            for key,value in values.items():
                X.append(value)
                if i==0:names.append(f'{cfg["mode"]}-K{cfg["k"]}/{key}')
                if key not in ['M','residual','positiveFraction','maxIncrease','relativeM']:wi.append(len(X)-1)
            bs.append(np.stack([C,M,C*(1-lex),C*(1-dense),R],axis=1))
        X.extend([lex,dense])
        if i==0:names+=['max-owned-char-similarity','max-owned-dense-similarity']
        wi.extend([len(X)-2,len(X)-1]);features.append(np.stack(X,axis=1));baseline.append(np.stack(bs,axis=1));without=wi
        if i%50==0 or i+1==len(rows):print('features',stage,i+1,'/',len(rows),'seconds',round(time.monotonic()-start,1),flush=True)
        engine.cache.clear()
    X=np.array(features,dtype=np.float64);B=np.array(baseline,dtype=np.float64)
    assert np.isfinite(X).all()
    np.savez_compressed(path,X=X,baselines=B)
    meta=dict(names=names,withoutM=without,configs=CONFIGS,inputsSha256=sha(OUT/f'{stage}-inputs.json'),
              protocolSha256=sha(OUT/'protocol.json'),featureSha256=sha(path),codeSha256=sha(__file__),seconds=time.monotonic()-start)
    write(OUT/f'{stage}-features-lock.json',meta)

def normalize(X):
    # A candidate's feature scale is relative to its actual comparison list.
    # Means/stds use no target, candidate position or construction role.
    return (X-X.mean(1,keepdims=True))/np.maximum(X.std(1,keepdims=True),1e-6)

def score_quality(scores,g):
    best=np.isclose(scores,scores.max(1,keepdims=True),atol=1e-10,rtol=0)
    return ((g/g.max(1,keepdims=True)*best).sum(1)/best.sum(1))

def fit_models(X,g,columns):
    from sklearn.linear_model import LogisticRegression,Ridge
    from sklearn.ensemble import HistGradientBoostingRegressor
    z=normalize(X)[:,:,columns];x=z.reshape(-1,len(columns));y=(g/g.max(1,keepdims=True)).ravel()
    diffs=[];labels=[];weights=[]
    for xx,gg in zip(z,g):
        dd=[];yy=[];ww=[]
        for a in range(8):
            for b in range(a+1,8):
                if gg[a]==gg[b]:continue
                dd.extend([xx[a]-xx[b],xx[b]-xx[a]]);yy.extend([int(gg[a]>gg[b]),int(gg[b]>gg[a])]);ww.extend([abs(gg[a]-gg[b])]*2)
        w=np.array(ww,dtype=float);w/=w.sum()
        diffs.extend(dd);labels.extend(yy);weights.extend(w)
    diffs=np.asarray(diffs);labels=np.asarray(labels);weights=np.asarray(weights);weights*=len(weights)/weights.sum()
    result=[]
    for strength in [.01,.1,1,10]:
        model=LogisticRegression(C=strength,fit_intercept=False,solver='liblinear',max_iter=5000,tol=1e-7,random_state=SEED)
        model.fit(diffs,labels,sample_weight=weights);result.append((f'pairwise-logistic-C{strength}',model,'normalized'))
    for strength in [.1,1,10,100]:
        model=Ridge(alpha=strength);model.fit(x,y);result.append((f'ridge-L2{strength}',model,'normalized'))
    for leaves in [7,15]:
        for penalty in [1,10]:
            model=HistGradientBoostingRegressor(max_iter=150,max_leaf_nodes=leaves,min_samples_leaf=25,l2_regularization=penalty,
                    learning_rate=.04,early_stopping=False,random_state=SEED)
            model.fit(x,y);result.append((f'boosted-leaves{leaves}-L2{penalty}',model,'normalized'))
    return result

def predict_model(model,kind,X,columns):
    z=normalize(X)[:,:,columns];xx=z.reshape(-1,len(columns))
    if hasattr(model,'decision_function'):y=model.decision_function(xx)
    else:y=model.predict(xx)
    return y.reshape(len(X),X.shape[1])

def select(explore=False):
    import joblib
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    if (OUT/'selection.json').exists():raise RuntimeError('Selection is already frozen')
    if (OUT/'test-features.npz').exists():raise RuntimeError('Test was opened too early')
    meta=read(OUT/'training-features-lock.json');tr=np.load(OUT/'training-features.npz');dv=np.load(OUT/'development-features.npz')
    tg=targets('training');dg=targets('development');allcols=list(range(tr['X'].shape[-1]));search=[];chosen={}
    for family,cols in [('upgraded',allcols),('learned-without-M',meta['withoutM'])]:
        fitted=fit_models(tr['X'],tg,cols);candidates=[]
        for order,(name,model,kind) in enumerate(fitted):
            s=predict_model(model,kind,dv['X'],cols);q=float(score_quality(s,dg).mean())
            row=dict(family=family,model=name,developmentQuality=q,priority=-order,columns=cols)
            candidates.append((round(q,12),-order,row,model,kind));search.append(row)
        _,_,profile,model,kind=max(candidates,key=lambda r:(r[0],r[1]))
        if not explore:joblib.dump(model,OUT/(family+'-model.joblib'))
        chosen[family]={**profile,'kind':kind,**({} if explore else {'modelSha256':sha(OUT/(family+'-model.joblib'))})}
        print('selected',family,profile['model'],profile['developmentQuality'],flush=True)
    for j,name in enumerate(['same-C','old-M','char-novelty','dense-novelty','residual']):
        scores=[float(score_quality(dv['baselines'][:,:,k,j],dg).mean()) for k in range(len(CONFIGS))]
        k=max(range(len(scores)),key=lambda k:(round(scores[k],12),-k));chosen[name]=dict(config=CONFIGS[k],index=k,column=j,developmentQuality=scores[k])
    if explore:
        write(OUT/'development-exploration-v1.json',dict(profiles=chosen,search=search,testUnopened=True,codeSha256=sha(__file__)))
        print(json.dumps({n:{k:v for k,v in p.items() if k!='columns'} for n,p in chosen.items()},ensure_ascii=False),flush=True)
        return
    write(OUT/'selection-search.json',search)
    write(OUT/'selection.json',dict(createdAt=datetime.now(timezone.utc).isoformat(),profiles=chosen,
        protocolSha256=sha(OUT/'protocol.json'),trainingFeaturesSha256=sha(OUT/'training-features.npz'),
        developmentFeaturesSha256=sha(OUT/'development-features.npz'),testFeaturesOpened=False,testLabelsOpened=False,
        codeSha256=sha(__file__),selectionRule='Maximum development normalized gain; deterministic simpler-first tie break; no refitting after selection.'))
    (OUT/'frozen-model-selection.py').write_bytes(Path(__file__).read_bytes())

def develop():select(explore=True)

def evaluate():
    import joblib
    from backend.thesis_evaluate_20261023 import metric
    from backend.thesis_analysis_20261023 import holm
    from backend.thesis_adversarial_audit_20261023 import gain_metric,exact_tied_dcg
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    if (OUT/'report.json').exists():raise RuntimeError('Final test already reported')
    lock=read(OUT/'selection.json');data=np.load(OUT/'test-features.npz');g=targets('test');scores={}
    for name,profile in lock['profiles'].items():
        if name in ['upgraded','learned-without-M']:
            path=OUT/(name+'-model.joblib');assert sha(path)==profile['modelSha256']
            scores[name]=predict_model(joblib.load(path),profile['kind'],data['X'],profile['columns'])
        else:scores[name]=data['baselines'][:,:,profile['index'],profile['column']]
    rates={name:score_quality(s,g) for name,s in scores.items()}
    def interval(a):
        a=np.asarray(a);rng=np.random.default_rng(SEED);boots=[]
        for _ in range(100):boots.extend(a[rng.integers(0,len(a),(100,len(a)))].mean(1))
        return dict(mean=float(a.mean()),ci95=np.quantile(boots,[.025,.975]).tolist())
    comparisons={};rng=np.random.default_rng(SEED)
    for name,v in rates.items():
        if name=='upgraded':continue
        d=rates['upgraded']-v;pcount=0
        for _ in range(500):pcount+=np.count_nonzero(np.abs((d*rng.choice([-1.,1.],(100,len(d)))).mean(1))>=abs(d.mean())-1e-14)
        comparisons[name]={**interval(d),'pPermutation':(pcount+1)/50001}
        comparisons[name]['p']=comparisons[name]['pPermutation']
    # Local correction is independent of the older comparison-family helpers.
    order=sorted(comparisons,key=lambda n:comparisons[n]['p']);last=0
    for rank,name in enumerate(order):
        last=max(last,min(1.,comparisons[name]['p']*(len(order)-rank)));comparisons[name]['holmP']=last
    inputs=read(OUT/'test-inputs.json');rows=[];maxerror=0.
    for i,c in enumerate(inputs):
        ms={}
        for name,s in scores.items():
            value=gain_metric(s[i],g[i]);maxerror=max(maxerror,abs(value-rates[name][i]))
            ms[name]=dict(normalizedGain=value,ndcg=exact_tied_dcg(s[i],g[i]))
        rows.append(dict(id=c['id'],gain=g[i].tolist(),scores={n:s[i].tolist() for n,s in scores.items()},metrics=ms))
    assert maxerror<1e-12
    write(OUT/'predictions.json',rows)
    report=dict(createdAt=datetime.now(timezone.utc).isoformat(),n=len(g),rates={n:interval(v) for n,v in rates.items()},comparisons=comparisons,
                checks=dict(independentMetricChecks=len(rows)*len(scores),maximumError=maxerror),
                selectionSha256=sha(OUT/'selection.json'),protocolSha256=sha(OUT/'protocol.json'),
                scope='Frozen internal final test of low-incidence original-outfit-disjoint constructed wardrobes. No external or real purchase satisfaction claim.',
                criterionMet=all(c['holmP']<.05 and c['mean']>0 and c['ci95'][0]>0 for c in comparisons.values()))
    write(OUT/'report.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','features','develop','select','evaluate']);p.add_argument('--split',choices=['training','development','test']);a=p.parse_args()
    features(a.split) if a.stage=='features' else globals()[a.stage]()
