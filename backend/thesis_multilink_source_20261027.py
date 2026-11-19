"""Source-rich wardrobe construction, fixed before any new model fitting.

The earlier semantic-neighbourhood pilot is preserved, never scored. Here
candidate outfit tops explicitly populate a constructed wardrobe; this is a
controlled record-reconstruction task, not naturally sampled user wardrobes.
"""
from collections import defaultdict,Counter
import argparse,shutil,time
import numpy as np
from threadpoolctl import threadpool_limits
from backend import thesis_multilink_inventory_20261027 as inv
from backend import thesis_repair_models_20261026 as b
ROOT=inv.ROOT;REPAIR=inv.REPAIR;OLD=inv.OUT;OUT=REPAIR/'multilink-source-20261027';SEED=2026102807
key=lambda s:b.stable(str(SEED)+':'+str(s))
VIEWS=[f'T{q}B{w}' for q in [3,4,6,8] for w in [2,3,4,5]]

def encode():
    assert not OUT.exists();OUT.mkdir()
    for n in ['items.json','reference.json','candidate-pools.json','source-lock.json']:shutil.copyfile(OLD/n,OUT/n)
    items=b.read(OUT/'items.json');ids=b.read(OLD/'itemIds.json');v=np.load(OLD/'vectors.npy');cache={p:v[i] for i,p in enumerate(ids)};missing=sorted(set(items)-set(ids))
    from backend.pog_retrieval import MODEL,REVISION,text
    if missing:
        import torch
        from sentence_transformers import SentenceTransformer
        torch.set_num_threads(4);m=SentenceTransformer(MODEL,revision=REVISION,device='cpu',local_files_only=True,trust_remote_code=False)
        a=m.encode([text(items[p]) for p in missing],batch_size=192,normalize_embeddings=True,show_progress_bar=True,convert_to_numpy=True);cache.update(zip(missing,a))
    ids=sorted(cache);np.save(OUT/'vectors.npy',np.asarray([cache[p] for p in ids]));b.write(OUT/'itemIds.json',ids)
    b.write(OUT/'encoder.json',dict(createdAt=b.stamp(),model=MODEL,revision=REVISION,newTitles=len(missing),idsSha256=b.sha(OUT/'itemIds.json'),vectorsSha256=b.sha(OUT/'vectors.npy'),priorSourceIndexSha256=b.sha(OLD/'reference.json')))
    print('SOURCE RICH ENCODER',len(ids),'new',len(missing),flush=True)

def prepare():
    assert not (OUT/'protocol.json').exists()
    from backend.pog_retrieval import text
    threadpool_limits(1);items=b.read(OUT/'items.json');ids=b.read(OUT/'itemIds.json');ix={p:i for i,p in enumerate(ids)};V=np.load(OUT/'vectors.npy')
    edges=b.read(ROOT/'expanded-projected-20261026/edges.json');ref=b.read(OUT/'reference.json');pools=b.read(OUT/'candidate-pools.json')
    io={p:{o.rsplit(':',1)[-1] for o in v} for p,v in b.read(ROOT/'expanded-full-source-incidence-20261026.json').items()}
    g=defaultdict(set);rev=defaultdict(set);rg=defaultdict(set);visibleW=set()
    for r in edges:g[r['top']].add(r['bottom']);rev[r['bottom']].add(r['top'])
    for r in ref:rg[r['top']].add(r['bottom']);visibleW.add(r['bottom'])
    norm={p:text(x).casefold() for p,x in items.items()};topopts=defaultdict(list)
    for q in rg:topopts[items[q]['kind']].append(q)
    topopts={k:sorted(v,key=key) for k,v in topopts.items()};topmatrix={k:V[[ix[p] for p in v]] for k,v in topopts.items()}
    bottomopts={k:sorted((w for w in visibleW if items[w]['kind']==k),key=key) for k in ['jeans','trousers','shorts']}
    bm={k:V[[ix[p] for p in opts]] for k,opts in bottomopts.items()}
    usedQ=set();usedW=set();usedC=set();usedTitle=set();usedOut=set();allframes={};reports={};start=time.monotonic()
    for stage in ['development','test']:
        pp=defaultdict(list)
        for c in pools[stage]:pp[items[c]['kind'],len(rev[c])].append(c)
        for opts in pp.values():opts.sort(key=key)
        frames=[];reject=Counter()
        for c1 in sorted(pools[stage],key=lambda c:key(stage+':first:'+c)):
            if c1 in usedC or norm[c1] in usedTitle or io[c1]&usedOut:continue
            kind=items[c1]['kind'];avail1=sorted((q for q in rev[c1] if q not in usedQ and norm[q] not in usedTitle),key=lambda q:key(stage+':source1:'+c1+q))
            if len(avail1)<2:reject['first-candidate-two-unused-tops']+=1;continue
            chosen=None
            for c2 in pp[kind,len(rev[c1])]:
                if c2==c1 or c2 in usedC or norm[c2] in usedTitle or norm[c2]==norm[c1] or io[c2]&(usedOut|io[c1]):continue
                initial=avail1[:2];avail2=sorted((q for q in rev[c2] if q not in usedQ and q not in initial and norm[q] not in usedTitle and norm[q] not in {norm[p] for p in initial}),key=lambda q:key(stage+':source2:'+c2+q))
                if len(avail2)<2:continue
                chosen=(c2,initial+avail2[:2]);break
            if chosen is None:reject['second-degree-matched-candidate-two-tops']+=1;continue
            c2,qs=chosen;cs=[c1,c2]
            # More source tops, then known semantic neighbours. Neither H nor
            # candidate gains are consulted while choosing these query tops.
            extra=sorted((rev[c1]|rev[c2])-set(qs)-usedQ,key=lambda q:key(stage+':source-extra:'+c1+q))
            for q in extra:
                if norm[q] in usedTitle or norm[q] in {norm[x] for x in qs}:continue
                qs.append(q)
                if len(qs)==6:break
            anchor=qs[0];qkind=items[anchor]['kind'];opts=topopts[qkind];scores=V[ix[anchor]]@topmatrix[qkind].T
            for j in np.argsort(-scores,kind='stable'):
                q=opts[int(j)]
                if q in usedQ or q in qs or norm[q] in usedTitle or norm[q] in {norm[x] for x in qs}:continue
                qs.append(q)
                if len(qs)==8:break
            if len(qs)!=8:reject['eight-distinct-tops']+=1;continue
            roles={norm[p] for p in qs+cs};opts=sorted(set().union(*(rg[q] for q in qs))-usedW-usedC,key=lambda w:key(stage+':existing:'+c1+w));ws=[]
            for w in opts:
                if norm[w] in usedTitle or norm[w] in roles or norm[w] in {norm[x] for x in ws}:continue
                ws.append(w)
                if len(ws)==5:break
            # Physical ownership does not require a recorded outfit with every
            # selected top. Fill sparse visible histories by title neighbours.
            if len(ws)<5:
                opts=bottomopts[kind];anchorW=ws[0] if ws else c1;scores=V[ix[anchorW]]@bm[kind].T
                for j in np.argsort(-scores,kind='stable'):
                    w=opts[int(j)]
                    if w in usedW or w in usedC or w in ws or norm[w] in usedTitle or norm[w] in roles or norm[w] in {norm[x] for x in ws}:continue
                    ws.append(w)
                    if len(ws)==5:break
            if len(ws)!=5:reject['five-distinct-existing-bottoms']+=1;continue
            if any((w in g[q])!=(w in rg[q]) for w in ws for q in qs):reject['incomplete-existing-records']+=1;continue
            qs=sorted(qs,key=lambda q:key(stage+':query-order:'+c1+q));ws=sorted(ws,key=lambda w:key(stage+':owned-order:'+c1+w));cs=np.random.default_rng(int(key(stage+':order:'+c1)[:8],16)).permutation(cs).tolist()
            observed=[[int(c in g[q]) for q in qs] for c in cs+ws];history=[[int(c in rg[q]) for q in qs] for c in cs+ws];origins=io[c1]|io[c2]
            assert not any(any(row) for row in history[:2]) and observed[2:]==history[2:]
            views={f'T{q}B{w}':dict(q=list(range(q)),c=[0,1],w=list(range(2,2+w))) for q in [3,4,6,8] for w in [2,3,4,5]}
            frames.append(dict(id=key(stage+':'+':'.join(cs))[:20],stage=stage,kind=kind,contexts=qs,bottoms=cs+ws,observed=observed,knownHistory=history,sourceOutfits=sorted(origins),sourceDegree=len(rev[c1]),views=views))
            usedQ.update(qs);usedW.update(ws);usedC.update(cs);usedTitle.update(norm[p] for p in qs+ws+cs);usedOut.update(origins)
            if len(frames)%100==0:print('SOURCE RICH PROFILES',stage,len(frames),'seconds',round(time.monotonic()-start,1),flush=True)
            if len(frames)>=600:break
        allframes[stage]=frames;gg=[]
        for c in frames:
            for q in [3,4,6,8]:
                for w in [2,3,4,5]:
                    H=np.asarray(c['knownHistory'])[2:2+w,:q].max(0);gg.append((np.asarray(c['observed'])[:2,:q]*(1-H)).sum(1))
        gg=np.asarray(gg);reports[stage]=dict(parents=len(frames),kinds=dict(Counter(c['kind'] for c in frames)),multiGainConditions=int((gg.max(1)>1).sum()),differentGainConditions=int((gg[:,0]!=gg[:,1]).sum()),positiveOpportunityConditions=int((gg.max(1)>0).sum()),maxGain=int(gg.max()),rejections=dict(reject))
    assert min(len(v) for v in allframes.values())>=60,reports
    for stage,frames in allframes.items():b.write(OUT/(stage+'-wardrobes.json'),frames)
    p=dict(createdAt=b.stamp(),codeSha256=b.sha(__file__),seed=SEED,candidateCount=2,referenceEdges=len(ref),counts=reports,
      task='Two equal-global-degree,distinct-title candidates. Include two distinct source tops per candidate,up to six total source tops,then fill to eight with known semantic neighbours. These are explicitly source-constructed candidate-context wardrobes,not natural user samples. Source membership selects query context,never forces existing overlap,gain ordering or a C/M disagreement.',
      ownership='Five distinct existing bottoms selected by hash from visible neighbours; sparse histories filled with semantic-neighbour bottoms. Physical ownership and public recorded coverage are different. All16conditions contain2-5existing bottoms,even if a selected top lacks a known existing record.',
      separation='No query,existing-bottom,candidate IDs or normalized titles reused between retained profiles/stages. Complete candidate source outfits disjoint and excluded from common reference. Near-semantic/style correlation remains possible. Some source records were visible in earlier indices; all models retrained from scratch on this new quarantine.',
      pilot='The earlier semantic-neighbourhood-only pilot226/159was inspected for raw coverage before fitting; it had only10/13multi-gain conditions and was not scored. This source-rich construction answers a narrower controlled multiple-link question.',
      sourceExposure='Previous four study outcomes remain historical; this is a separate exploratory prospective model comparison,not a new label for the earlier reserved one-opening study.',
      metrics='Parent-average16condition NG main; observed gain,regret,positive-opportunity,unequal-gain and multi-gain strata also reported. No filtering on these labels or predictions after construction.',
      encoderSha256=b.sha(OUT/'encoder.json'),pilotProtocolSha256=b.sha(OLD/'protocol.json'),
      previousResultSha256=b.sha(REPAIR/'rank-objective-20261027/test-results.json'),
      files={n:b.sha(OUT/n) for n in ['items.json','reference.json','development-wardrobes.json','test-wardrobes.json']})
    b.write(OUT/'protocol.json',p);print('SOURCE RICH PROFILES SEALED',reports,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['encode','prepare']);globals()[ap.parse_args().stage]()
