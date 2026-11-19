"""Explicit follow-up setting: known wardrobe queries, newly held-out candidates.

The two failed fully-cold repair rounds remain frozen. This is a separate,
post-result study, not a third attempt relabelled as its untouched primary test.
The setting, changed information boundary and all outcomes must be disclosed.
"""
from collections import defaultdict,Counter
from pathlib import Path
import argparse,shutil,time
import numpy as np
from backend import thesis_repair_models_20261026 as b

ROOT=b.DATA/'asos'; REPAIR=b.DATA/'research-repair-20261026'
OUT=REPAIR/'known-query-corrected-20261027'
SEED=2026102757
key=lambda p:b.stable(str(SEED)+':'+str(p))
oid=lambda o:o.rsplit(':',1)[-1]

def prepare():
    assert not (OUT/'protocol.json').exists()
    items=b.read(ROOT/'expanded-projected-20261026/items.json')
    edges=b.read(ROOT/'expanded-projected-20261026/edges.json')
    io={p:{oid(o) for o in v} for p,v in b.read(ROOT/'expanded-full-source-incidence-20261026.json').items()}
    previous=set(b.read(OUT/'prior-ids.json')); cold=set(b.read(OUT/'cold-candidate-ids.json'))
    # Development candidates are explicitly previously exposed. Their source
    # labels cannot subsequently be called independent new evidence.
    devcold={p for p in previous if p in items and items[p]['side']=='bottom' and int(key('dev:'+p)[:8],16)%5==0}
    removed=set().union(*(io[p] for p in cold|devcold))
    ref=[r for r in edges if not {oid(o) for o in r['outfits']}&removed]
    g=defaultdict(set);rev=defaultdict(set);rg=defaultdict(set);rr=defaultdict(set)
    for r in edges:g[r['top']].add(r['bottom']);rev[r['bottom']].add(r['top'])
    for r in ref:rg[r['top']].add(r['bottom']);rr[r['bottom']].add(r['top'])
    allframes={};counter={}
    for stage,pool,limit in [('development',devcold,600),('test',cold,350)]:
        pc=defaultdict(list);pd=defaultdict(list);tq=defaultdict(set)
        for c in pool&set(rev):
            kind=items[c]['kind'];pc[kind].append(c);pd[kind,len(rev[c])].append(c)
            for q in rev[c]&set(rg):tq[kind].add(q)
        for lists in [pc,pd]:
            for v in lists.values():v.sort(key=key)
        used_c=set();used_q=set();used_w=set();taken=set();frames=[];reject=Counter()
        history_options={k:sorted((w for w in rr if rr[w]&tq[k]),key=key) for k in pc}
        query_options={k:sorted(tq[k],key=key) for k in pc}
        history_order=sorted(((k,w) for k,ws in history_options.items() for w in ws),key=lambda p:key(stage+':history:'+p[0]+p[1]))
        for kind,w in history_order:
            if w in used_w:continue
            anchors=(rr[w]&tq[kind])-used_q
            if not anchors:continue
            q0=min(anchors,key=key);qs=[q0];ws=[w]
            nw=[2,3][int(key(stage+':w:'+w)[:8],16)%2]
            nq=[5,8][int(key(stage+':q:'+w)[:8],16)%2]
            # Every wardrobe has 2/3 existing bottoms, potentially of different
            # kinds. Each contributes a visible outfit with one selected top.
            # Existing jeans may therefore cover a query for new trousers.
            options=history_options[kind];start=int(key(stage+':more-history:'+w)[:8],16)%len(options)
            for x in options[start:]+options[:start]:
                aa=(rr[x]&tq[kind])-used_q-set(qs)
                if x in used_w or x in ws or not aa:continue
                ws.append(x);qs.append(min(aa,key=key))
                if len(ws)==nw:break
            if len(ws)<nw:reject['owned-bottom-count']+=1;continue
            # History first. Remaining queries chosen by hash rotation, without
            # model scores, candidate gain or forced C/M disagreement.
            options=query_options[kind];start=int(key(stage+':fill:'+w)[:8],16)%len(options)
            for q in options[start:]+options[:start]:
                if q in used_q or q in qs:continue
                qs.append(q)
                if len(qs)==nq:break
            if len(qs)<nq:reject['query-count']+=1;continue
            if any((x in g[q])!=(x in rg[q]) for x in ws for q in qs):reject['incomplete-visible-history']+=1;continue
            available=lambda c:c not in used_c and not io[c]&taken
            seeds=sorted((set().union(*(g[q] for q in qs))&set(pc[kind]))-{w},key=lambda c:key(stage+':seed:'+w+c))
            cs=[]
            for c in seeds:
                if available(c):cs.append(c)
                if len(cs)==2:break
            if len(cs)<2:reject['two-recorded-seeds']+=1;continue
            for seed in list(cs):
                opts=pd[kind,len(rev[seed])]
                for c in sorted(opts,key=lambda c:key(stage+':control:'+w+seed+c)):
                    if available(c) and c not in cs and not any(c in g[q] for q in qs):cs.append(c);break
            if len(cs)<4:reject['degree-controls']+=1;continue
            assert all(items[x]['kind']==kind for x in cs)
            cs=np.random.default_rng(int(key(stage+':order:'+w)[:8],16)).permutation(cs).tolist()
            history=[[int(x in rg[q]) for q in qs] for x in ws]
            actual=[[int(x in g[q]) for q in qs] for x in ws]
            assert history==actual and np.max(history,axis=0).sum()>=nw
            observed=[[int(c in g[q]) for q in qs] for c in cs]+actual
            hist=[[0]*nq for _ in cs]+history
            origins=set().union(*(io[c] for c in cs));assert not origins&taken
            frames.append(dict(id=key(stage+':'+':'.join(sorted(cs)))[:20],stage=stage,contexts=qs,bottoms=cs+ws,
              kind=kind,observed=observed,knownHistory=hist,sourceOutfits=sorted(origins),
              views={'primary':dict(q=list(range(nq)),c=list(range(4)),w=list(range(4,4+nw)))},
              construction=dict(topCount=nq,ownedCount=nw,knownCoverage=int(np.max(history,axis=0).sum()))))
            used_c.update(cs);used_q.update(qs);used_w.update(ws);taken.update(origins)
            if len(frames)>=limit:break
        allframes[stage]=frames;counter[stage]=dict(n=len(frames),kinds=dict(Counter(c['kind'] for c in frames)),
          queries=dict(Counter(len(c['contexts']) for c in frames)),rejections=dict(reject))
    assert len(allframes['development'])>=60 and len(allframes['test'])>=40,counter
    # Candidate families and their complete source outfits are separated across
    # development/test. Queries and visible histories may be known and shared
    # across stages by design; never describe this as all-item-disjoint.
    dc={p for c in allframes['development'] for p in c['bottoms'][:4]}
    tc={p for c in allframes['test'] for p in c['bottoms'][:4]}
    do=set().union(*(io[p] for p in dc));to=set().union(*(io[p] for p in tc))
    assert not dc&tc and not do&to and not tc&previous
    assert not any({oid(o) for o in r['outfits']}&(do|to) for r in ref)
    shutil.copyfile(ROOT/'expanded-projected-20261026/items.json',OUT/'items.json')
    b.write(OUT/'reference.json',ref)
    for stage,frames in allframes.items():b.write(OUT/(stage+'-frames.json'),frames)
    protocol=dict(createdAt=b.stamp(),seed=SEED,codeSha256=b.sha(__file__),phase='Explicit post-result follow-up under a different known-query / cold-candidate information boundary.',
      previousResults={n:b.sha(REPAIR/n/'test-results.json') for n in ['common-reference-study','expanded-learning-final']},
      previousFailureRetained=True,maximumEvaluationsInThisSetting=1,notExternalPreregistration=True,
      motivation='Sparse owned-history inputs did not represent the intended existing-wardrobe purchase task. Here every constructed wardrobe contains two/three visible existing bottoms and five/eight reference-known tops. New candidate IDs and their full source outfits remain cold. Existing bottoms can be of a different kind from candidate bottoms.',
      construction='History first, hash-selected queries, then two uniformly hash-selected recorded candidates from all queries, plus two full-degree-matched unobserved controls. No forced ordering of gain or C/M conflict; zero-gain inputs retained.',
      counts=counter,referenceEdges=len(ref),developmentColdIDs=len(devcold),testColdIDs=len(cold),
      information='Allow query itself in retrieval when it has legitimate remaining visible outfits. Entire candidate source outfits removed from common reference before any learning. Near-candidate exclusions remain unchanged. Titles only; no target IDs/degrees in model features.',
      models=[dict(view=v,family=f) for v in ['compact','wide'] for f in ['HGB_leaf7','HGB_leaf15']],
      alphas=[0,.1,.25,.5,1.],folds=5,foldSeed=SEED,
      choice='Mean parent OOF NG, then weighted pair logloss, then smaller alpha, then fixed model enumeration. Same four architectures and weights per RAG/noRAG path. Report both selected-method and exact-architecture ablations.',
      primaryComparisons=['noRAG','direct','sameSupportC','oldRAG'],
      statistics='10000 paired bootstrap,50000 sign flips,Holm4; report exploratory status and retrospective factor3 sensitivity including both earlier repair rounds. One opening; no claiming the prior cold-query hypotheses have been proved.',
      files={n:b.sha(OUT/n) for n in ['items.json','reference.json','development-frames.json','test-frames.json']},
      candidateIndependence=dict(developmentTestIDs=0,developmentTestFullOutfits=0,testPriorIDs=0,withinStageCandidateFullOutfitsDisjoint=True),
      sourceInventorySha256=b.sha(OUT/'inventory.json'))
    b.write(OUT/'protocol.json',protocol)
    print('FOLLOW-UP PREPARED',counter,'reference',len(ref),flush=True)

def encode():
    assert not (OUT/'vectors.npy').exists();p=b.read(OUT/'protocol.json')
    for n,s in p['files'].items():assert b.sha(OUT/n)==s
    old=REPAIR/'expanded-learning-final';items=b.read(OUT/'items.json');ref=b.read(OUT/'reference.json')
    needed={r[k] for r in ref for k in ['top','bottom']}
    for stage in ['development','test']:
        needed.update(p for c in b.read(OUT/(stage+'-frames.json')) for p in c['contexts']+c['bottoms'])
    ids=b.read(old/'itemIds.json');vec=np.load(old/'vectors.npy');cache={p:vec[i] for i,p in enumerate(ids) if p in needed}
    missing=sorted(needed-set(cache))
    if missing:
        import torch
        from sentence_transformers import SentenceTransformer
        from backend.pog_retrieval import MODEL,REVISION,text
        torch.set_num_threads(4);m=SentenceTransformer(MODEL,revision=REVISION,device='cpu',local_files_only=True,trust_remote_code=False)
        v=m.encode([text(items[p]) for p in missing],batch_size=192,normalize_embeddings=True,show_progress_bar=True,convert_to_numpy=True)
        cache.update(zip(missing,v))
    ids=sorted(needed);np.save(OUT/'vectors.npy',np.asarray([cache[p] for p in ids]));b.write(OUT/'itemIds.json',ids)
    b.write(OUT/'encoder.json',dict(createdAt=b.stamp(),newTitles=len(missing),items=len(ids),vectorsSha256=b.sha(OUT/'vectors.npy'),idsSha256=b.sha(OUT/'itemIds.json')))
    print('FOLLOW-UP ENCODED',len(ids),'new',len(missing),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','encode']);globals()[ap.parse_args().stage]()
