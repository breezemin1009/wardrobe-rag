"""Common-reference learned support and one sealed, diverse new ASOS test.

All historical input pairs are development. Both their targets and every new
cold candidate are absent from the common retrieval reference before fitting.
No new test features/predictions may be computed before the selection binding.
"""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,json,hashlib,time,shutil
import numpy as np
from threadpoolctl import threadpool_limits
from backend import thesis_repair_models_20261026 as base

DATA=base.DATA;ROOT=DATA/'asos';OUT=DATA/'research-repair-20261026/common-reference-study'
read=base.read;sha=base.sha;write=base.write;stamp=base.stamp
SEED=2026102631
def key(x):return hashlib.sha256((str(SEED)+':'+str(x)).encode()).hexdigest()
def oid(x):return x.rsplit(':',1)[-1]

def reserve():
    if OUT.exists():raise RuntimeError('Study already reserved')
    items=read(ROOT/'expanded-projected-20261026/items.json');edges=read(ROOT/'expanded-projected-20261026/edges.json')
    io={p:{oid(o) for o in os} for p,os in read(ROOT/'expanded-full-source-incidence-20261026.json').items()}
    prior=read(DATA/'research-repair-20261026/prior-exposure.json');used=set(prior['ids']);occupied={oid(o) for o in prior['outfits']}
    eligible={p for p in items if p not in used and not io[p]&occupied}
    dev=read(base.OUT/'frames.json');oldc={p for c in dev for p in c['bottoms'][:4]}
    cold={p for p in eligible if items[p]['side']=='bottom' and int(key('cold:'+p)[:8],16)%10<4}
    removed=oldc|cold;removed_outfits={o for p in removed for o in io[p]}
    ref=[e for e in edges if e['bottom'] not in removed and not {oid(o) for o in e['outfits']}&removed_outfits]
    g=defaultdict(set);rev=defaultdict(set);rg=defaultdict(set);rr=defaultdict(set)
    for e in edges:g[e['top']].add(e['bottom']);rev[e['bottom']].add(e['top'])
    for e in ref:rg[e['top']].add(e['bottom']);rr[e['bottom']].add(e['top'])
    assert not {r['bottom'] for r in ref}&removed
    candidates=defaultdict(list);degree=defaultdict(list)
    order=lambda p:(len(io[p]),key(p))
    for p in cold:
        if p in rev:candidates[items[p]['kind']].append(p);degree[items[p]['kind'],len(rev[p])].append(p)
    for pool in [candidates,degree]:
        for ps in pool.values():ps.sort(key=order)
    frames=[];reject=Counter();seen=set(used);taken=set(occupied)
    okay=lambda p,local:p in eligible and p not in seen and p not in local and not io[p]&taken
    for a in sorted(cold&set(rev),key=order):
        if not okay(a,set()):continue
        kind=items[a]['kind'];nq=[3,5,8][int(key('q:'+a)[:8],16)%3];nw=[0,1,3][int(key('w:'+a)[:8],16)%3]
        local={a};cs=[a];b=None
        for p in sorted(candidates[kind],key=lambda p:(abs(len(rev[p])-len(rev[a])),order(p))):
            if okay(p,local) and any(okay(q,local|{p}) for q in rev[p]):b=p;break
        if b is None:reject['second-seed']+=1;continue
        cs.append(b);local.add(b)
        for seed in [a,b]:
            for p in degree[kind,len(rev[seed])]:
                if okay(p,local):cs.append(p);local.add(p);break
        if len(cs)<4:reject['degree-controls']+=1;continue
        qs=[]
        for seed in [a,b]:
            for q in sorted(rev[seed],key=order):
                if okay(q,local):qs.append(q);local.add(q);break
        if len(qs)<2:reject['seed-contexts']+=1;continue
        extras=sorted(set().union(*(rev[p] for p in cs))-set(qs),key=lambda p:key('qextra:'+p))
        for q in extras:
            if okay(q,local):qs.append(q);local.add(q)
            if len(qs)==nq:break
        if len(qs)<nq:
            for q in sorted(rg,key=lambda p:key('qfill:'+p)):
                if okay(q,local):qs.append(q);local.add(q)
                if len(qs)==nq:break
        if len(qs)<nq:reject['top-count']+=1;continue
        ws=[];coverage=set()
        pool={w for q in qs for w in rg[q] if items[w]['kind']==kind}
        for _ in range(nw):
            options=[w for w in pool if okay(w,local) and all((w in g[q])==(w in rg[q]) for q in qs)]
            if not options:break
            # Only available reference history, no hidden candidate gain.
            w=min(options,key=lambda w:(-len((rr[w]&set(qs))-coverage),order(w)))
            ws.append(w);local.add(w);coverage.update(rr[w]&set(qs))
        if len(ws)<nw:reject['owned-count']+=1;continue
        rng=np.random.default_rng(int(key('shuffle:'+a)[:10],16));cs=rng.permutation(cs).tolist()
        ids=set(qs+cs+ws);os=set().union(*(io[p] for p in ids));assert not ids&seen and not os&taken
        observed=np.asarray([[int(p in g[q]) for q in qs] for p in cs+ws],dtype=int)
        history=np.asarray([[int(p in rg[q]) for q in qs] for p in cs+ws],dtype=int)
        assert not history[:4].any() and np.array_equal(history[4:],observed[4:])
        gains=(observed[:4]*(1-history[4:].max(0) if ws else 1)).sum(1)
        frames.append(dict(id=key('test:'+':'.join(sorted(cs)))[:20],stage='test',contexts=qs,bottoms=cs+ws,kind=kind,
          observed=observed.tolist(),knownHistory=history.tolist(),sourceOutfits=sorted(os),
          views={'primary':dict(q=list(range(nq)),c=list(range(4)),w=list(range(4,4+nw)))},
          construction=dict(topCount=nq,ownedCount=nw,nonzeroGain=bool(gains.max()),knownCoverage=int(history[4:].max(0).sum()) if ws else 0)))
        seen.update(ids);taken.update(os)
        if len(frames)>=1200:break
    assert frames,'No new frames; do not silently substitute previously used inputs'
    OUT.mkdir(parents=True)
    # Existing development gains remain the historical observations, but H
    # is rebuilt only from this common allowed reference.
    for c in dev:
        c['knownHistory']=[[int(p in rg[q]) for q in c['contexts']] for p in c['bottoms']]
        assert not np.asarray(c['knownHistory'])[:4].any()
    for name,value in [('items.json',items),('reference.json',ref),('development-frames.json',dev),('test-frames.json',frames)]:write(OUT/name,value)
    first=read(base.OUT/'plan.json')
    plan={**first,'createdAt':stamp(),'stage':'Final test reserved before new support fitting; no test score computed',
      'commonReference':'Remove ALL original source outfits of all445development candidate targets and hash40%new candidate pool. Thus no new candidate interaction appears in fitted RAG features or test retrieval. Known query/owned reference interactions are allowed, identically for RAG and noRAG paths.',
      'construction':'One view per source-disjoint parent; Q3/5/8 and W0/1/3 from independent item hash. Two degree-matched observed seeds and two degree-matched controls. No forced candidate overlap-versus-new conflict. W chosen greedily using visible history only. Keep all zero-gain requests; NG=0 there for all methods. Not a sample of human wardrobes.',
      'primaryMetric':'Mean parent NG, with NG=0 if max gain0. Secondary actual new-link fraction Gselected/|Q| and nonzero-task NG are descriptive, not extra selected hypothesis tests.',
      'statisticsSeeds':dict(bootstrap=2026102632,signflip=2026102633),
      'counts':dict(developmentParents=len(dev),testParents=len(frames),referenceEdges=len(ref),kinds=dict(Counter(c['kind'] for c in frames)),topCounts=dict(Counter(len(c['contexts']) for c in frames)),ownedCounts=dict(Counter(len(c['bottoms'])-4 for c in frames)),positiveTaskParents=sum(c['construction']['nonzeroGain'] for c in frames),rejections=dict(reject)),
      'files':{n:sha(OUT/n) for n in ['items.json','reference.json','development-frames.json','test-frames.json']},
      'repairCodeSha256':sha(__file__),'priorExposureSha256':sha(DATA/'research-repair-20261026/prior-exposure.json'),
      'sourceFiles':{n:sha(ROOT/n) for n in ['expanded-projected-20261026/items.json','expanded-projected-20261026/edges.json','expanded-full-source-incidence-20261026.json']}}
    write(OUT/'protocol.json',plan);write(OUT/'plan.json',plan)
    print('RESERVED',json.dumps(plan['counts']),flush=True)

def verify():
    p=read(OUT/'protocol.json');assert p['repairCodeSha256']==sha(__file__)
    for n,h in p['files'].items():assert sha(OUT/n)==h,n
    return p

def encode():
    from backend.pog_retrieval import MODEL,REVISION,text
    from sentence_transformers import SentenceTransformer
    import torch
    verify()
    if (OUT/'vectors.npy').exists():raise RuntimeError('Vectors already sealed')
    items=read(OUT/'items.json');ref=read(OUT/'reference.json');frames=read(OUT/'development-frames.json')+read(OUT/'test-frames.json')
    needed={r[s] for r in ref for s in ['top','bottom']}|{p for c in frames for p in c['contexts']+c['bottoms']}
    cache={}
    for path in base.COHORTS:
        p=DATA/path;ids=read(p/'itemIds.json');V=np.load(p/'vectors.npy')
        for i,pid in enumerate(ids):
            if pid in needed:cache[pid]=V[i]
    missing=sorted(needed-set(cache));start=time.monotonic()
    if missing:
        torch.set_num_threads(4);m=SentenceTransformer(MODEL,revision=REVISION,device='cpu',local_files_only=True,trust_remote_code=False)
        values=m.encode([text(items[p]) for p in missing],batch_size=192,normalize_embeddings=True,show_progress_bar=True,convert_to_numpy=True)
        cache.update(zip(missing,values))
    ids=sorted(needed);np.save(OUT/'vectors.npy',np.asarray([cache[p] for p in ids]));write(OUT/'itemIds.json',ids)
    write(OUT/'encoder.json',dict(model=MODEL,revision=REVISION,items=len(ids),newTitles=len(missing),seconds=time.monotonic()-start,vectorsSha256=sha(OUT/'vectors.npy'),idsSha256=sha(OUT/'itemIds.json')))
    print('ENCODED',len(ids),'new',len(missing),flush=True)

def extract(stage):
    from backend.pog_refined_retrieval import Retriever
    verify();threadpool_limits(1)
    if stage=='test' and not (OUT/'selection-binding.json').exists():raise RuntimeError('New test features still sealed')
    dest=OUT/('features.npz' if stage=='development' else 'test-features.npz')
    if dest.exists():raise RuntimeError('Features already sealed')
    e=Retriever(OUT);frames=read(OUT/(stage+'-frames.json'));xx=[];ff=[];rr=[];start=time.monotonic()
    configs=[dict(mode=mode,k=k) for mode in ['dense-conditional','density','hybrid-conditional','hybrid-density'] for k in [1,8,32]]
    for i,c in enumerate(frames):
        xx.append(base.base_features(e,c));rr.append(base.reverse_features(e,c));ff.append(np.stack([e.matrix(c['contexts'],c['bottoms'][:4],cfg) for cfg in configs],axis=-1));e.cache.clear()
        if i%25==0:print(stage,'COMMON FEATURES',i+1,'/',len(frames),'seconds',round(time.monotonic()-start,1),flush=True)
    # New test Q lengths vary; save them individually rather than pad scores.
    if stage=='test':
        np.savez_compressed(dest,**{f'{part}{i}':v for part,values in [('X',xx),('F',ff),('R',rr)] for i,v in enumerate(values)})
    else:
        np.savez_compressed(dest,X=np.asarray(xx),F=np.asarray(ff),R=np.asarray(rr));shutil.copyfile(OUT/'development-frames.json',OUT/'frames.json')
        write(OUT/'feature-lock.json',dict(createdAt=stamp(),parents=len(frames),featuresSha256=sha(dest),framesSha256=sha(OUT/'frames.json'),planSha256=sha(OUT/'plan.json')))
    write(OUT/(stage+'-feature-binding.json'),dict(createdAt=stamp(),featuresSha256=sha(dest),protocolSha256=sha(OUT/'protocol.json'),codeSha256=sha(__file__)))

def develop():
    verify();assert not (OUT/'test-features.npz').exists();base.OUT=OUT;base.develop()
    write(OUT/'selection-binding.json',dict(createdAt=stamp(),selectionSha256=sha(OUT/'selection-lock.json'),protocolSha256=sha(OUT/'protocol.json'),codeSha256=sha(__file__),newTestOpened=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['reserve','encode','extract-development','develop','extract-test']);a=p.parse_args()
    if a.stage.startswith('extract-'):extract(a.stage.split('-',1)[1])
    else:globals()[a.stage]()
