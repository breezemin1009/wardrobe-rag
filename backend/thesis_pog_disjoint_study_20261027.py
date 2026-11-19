"""Prospective POG replication with fully disjoint input source neighbourhoods.

Earlier ASOS failures and POG forced-conflict results remain historical. This
study changes source and strengthens sampling independence before any model.
"""
from collections import defaultdict,Counter
from pathlib import Path
import argparse,os,time
import numpy as np
from threadpoolctl import threadpool_limits
from backend import thesis_pog_known_inventory_20261027 as inventory
from backend import thesis_repair_models_20261026 as b
OUT=b.DATA/'research-repair-20261026/pog-disjoint-20261027';SEED=2026102847
key=lambda x:b.stable(str(SEED)+':'+str(x))

def reserve():
    assert not OUT.exists();OUT.mkdir();src=inventory.OUT;items=b.read(src/'items.json');edges=b.read(src/'all-edges.json');io={p:set(v) for p,v in b.read(src/'item-outfits.json').items()}
    fresh=set(b.read(src/'eligible-candidate-ids.json'));g=defaultdict(set);rev=defaultdict(set)
    for r in edges:g[r['top']].add(r['bottom']);rev[r['bottom']].add(r['top'])
    dev=sorted([p for p,x in items.items() if x['side']=='bottom' and p not in fresh and len(rev[p])>=2 and x['kind'] in ['jeans','trousers','shorts']],key=lambda p:key('development-pool:'+p))[:600]
    pools={'test':sorted(fresh),'development':dev};cold=set(fresh)|set(dev);excluded=set().union(*(io[p] for p in cold));ref=[r for r in edges if not set(r['outfits'])&excluded]
    # All facts used for H are retained original relations, not predicted links.
    rg=defaultdict(set);visibleW=set()
    for r in ref:rg[r['top']].add(r['bottom']);visibleW.add(r['bottom'])
    os.link(src/'items.json',OUT/'items.json');b.write(OUT/'reference.json',ref);b.write(OUT/'candidate-pools.json',pools)
    # Reuse only exactly matching title/model vectors from the previous POG
    # serving release. No compatibility model/score is copied from that study.
    previous=inventory.ROOT/'evidence-history-serving-20261026';ids=b.read(previous/'itemIds.json');vectors=np.load(previous/'vectors.npy');cache={p:vectors[i] for i,p in enumerate(ids)};missing=sorted(set(items)-set(cache))
    from backend.pog_retrieval import MODEL,REVISION,text
    if missing:
        from sentence_transformers import SentenceTransformer
        import torch
        torch.set_num_threads(4);model=SentenceTransformer(MODEL,revision=REVISION,device='cpu',local_files_only=True,trust_remote_code=False);values=model.encode([text(items[p]) for p in missing],normalize_embeddings=True,batch_size=192,convert_to_numpy=True,show_progress_bar=True);cache.update(zip(missing,values))
    ids=sorted(items);V=np.asarray([cache[p] for p in ids]);ix={p:i for i,p in enumerate(ids)};np.save(OUT/'vectors.npy',V);b.write(OUT/'itemIds.json',ids)
    b.write(OUT/'encoder.json',dict(model=MODEL,revision=REVISION,newTitles=len(missing),vectorsSha256=b.sha(OUT/'vectors.npy'),idsSha256=b.sha(OUT/'itemIds.json'),priorCacheSha256=b.sha(previous/'vectors.npy')))
    threadpool_limits(1);norm={p:text(x).casefold() for p,x in items.items()};topopts=defaultdict(list)
    for q in rg:topopts[items[q]['kind']].append(q)
    topopts={k:sorted(v,key=key) for k,v in topopts.items()};tm={k:V[[ix[p] for p in v]] for k,v in topopts.items()};bottomopts={k:sorted([p for p in visibleW if items[p]['kind']==k],key=key) for k in ['jeans','trousers','shorts']};bm={k:V[[ix[p] for p in v]] for k,v in bottomopts.items()}
    used=set();titles=set();occupied=set();frames={};counts={};start=time.monotonic()
    def okay(p,local):return p not in used and p not in local and norm[p] not in titles and not io[p]&occupied
    # Reserve the scarce fresh-candidate evaluation first, without predictions.
    for stage in ['test','development']:
        bins=defaultdict(list)
        for c in pools[stage]:bins[items[c]['kind'],len(rev[c])].append(c)
        for v in bins.values():v.sort(key=key)
        rows=[];reject=Counter()
        for c1 in sorted(pools[stage],key=lambda p:key(stage+':first:'+p)):
            if not okay(c1,set()):continue
            aa=sorted([q for q in rev[c1] if okay(q,{c1})],key=lambda q:key(stage+':source:'+c1+q))
            # Ensure actual identity and cleaned-title uniqueness inside Q too.
            q1=[]
            for q in aa:
                if norm[q] not in {norm[x] for x in q1}:q1.append(q)
                if len(q1)==2:break
            if len(q1)<2:reject['two-first-source-tops']+=1;continue
            choice=None
            for c2 in bins[items[c1]['kind'],len(rev[c1])]:
                if not okay(c2,{c1}) or norm[c2]==norm[c1] or io[c2]&io[c1]:continue
                local=set(q1+[c1,c2]);qq=[]
                for q in sorted(rev[c2],key=lambda p:key(stage+':source2:'+c2+p)):
                    if not okay(q,local) or norm[q] in {norm[x] for x in q1+qq}:continue
                    qq.append(q)
                    if len(qq)==2:break
                if len(qq)==2:choice=c2,q1+qq;break
            if choice is None:reject['degree-matched-second-candidate']+=1;continue
            c2,qs=choice;cs=[c1,c2];local=set(cs+qs)
            for q in sorted((rev[c1]|rev[c2])-set(qs),key=lambda q:key(stage+':extra:'+c1+q)):
                if not okay(q,local) or norm[q] in {norm[x] for x in local}:continue
                qs.append(q);local.add(q)
                if len(qs)==6:break
            qkind=items[qs[0]]['kind'];opts=topopts[qkind];sim=V[ix[qs[0]]]@tm[qkind].T
            for j in np.argsort(-sim,kind='stable'):
                q=opts[int(j)]
                if not okay(q,local) or norm[q] in {norm[x] for x in local}:continue
                qs.append(q);local.add(q)
                if len(qs)==8:break
            if len(qs)!=8:reject['eight-source-independent-tops']+=1;continue
            ws=[];options=sorted(set().union(*(rg[q] for q in qs)),key=lambda w:key(stage+':owned:'+c1+w))
            def append_owned(w):
                if not okay(w,local) or norm[w] in {norm[x] for x in local}:return
                if any((w in g[q])!=(w in rg[q]) for q in qs):return
                ws.append(w);local.add(w)
            for w in options:
                append_owned(w)
                if len(ws)==5:break
            if len(ws)<5:
                kind=items[c1]['kind'];opts=bottomopts[kind];sim=V[ix[ws[0] if ws else c1]]@bm[kind].T
                for j in np.argsort(-sim,kind='stable'):
                    append_owned(opts[int(j)])
                    if len(ws)==5:break
            if len(ws)!=5:reject['five-observed-history-bottoms']+=1;continue
            qs=sorted(qs,key=lambda q:key(stage+':query-order:'+c1+q));ws=sorted(ws,key=lambda w:key(stage+':owned-order:'+c1+w));cs=np.random.default_rng(int(key(stage+':candidate-order:'+c1)[:8],16)).permutation(cs).tolist();role=set(qs+cs+ws);origins=set().union(*(io[p] for p in role));assert not origins&occupied
            obs=[[int(c in g[q]) for q in qs] for c in cs+ws];hist=[[int(c in rg[q]) for q in qs] for c in cs+ws];assert not any(any(r) for r in hist[:2]) and obs[2:]==hist[2:]
            views={f'T{q}B{w}':dict(q=list(range(q)),c=[0,1],w=list(range(2,2+w))) for q in [3,4,6,8] for w in [2,3,4,5]}
            rows.append(dict(id=key(stage+':'+':'.join(cs))[:20],stage=stage,kind=items[c1]['kind'],contexts=qs,bottoms=cs+ws,observed=obs,knownHistory=hist,sourceDegree=len(rev[c1]),candidateOutfits=sorted(io[c1]|io[c2]),sourceOutfits=sorted(origins),views=views))
            used.update(role);titles.update(norm[p] for p in role);occupied.update(origins)
            if len(rows)%25==0:print('POG FULL SOURCE DISJOINT',stage,len(rows),'seconds',round(time.monotonic()-start,1),flush=True)
            if len(rows)>=300:break
        frames[stage]=rows;counts[stage]=dict(parents=len(rows),kinds=dict(Counter(c['kind'] for c in rows)),rejections=dict(reject))
    for stage,rows in frames.items():b.write(OUT/(stage+'-wardrobes.json'),rows)
    protocol=dict(createdAt=b.stamp(),codeSha256=b.sha(__file__),seed=SEED,counts=counts,referenceEdges=len(ref),testPool=len(fresh),developmentPool=len(dev),
      scope='POG replication,known public Q/W permitted from earlier project,strict fresh candidate IDs/fulloutfits relative to prior bottom inputs. Entire current role original-source neighbourhoods disjoint acrossalltestanddevelopmentprofiles;cleanedtitles/IDs also disjoint. Explicit originaltitleadultmale scope unchanged.',
      construction='Two samekind/exactglobaldegree candidates;two source tops per candidate,up to6sourcetops,fill8knownsemanticneighbours;5ownedbottomshashselectedthensemanticfill. No forced gain ordering,C/Mconflict,Hcoverage,or exclusionofzero/equalgainconditions.',
      availabilityPassed=all(v['parents']>=60 for v in counts.values()),
      researchHistory='Separate source replication after all earlier ASOS/POG outcomes;not relabelledaspreregisteredconfirmation. Earlier ASOS auxiliary/directgain development failures remain reportable. Sampling/sourceindependence specifiedbeforemodel fitting,notchosenbyscores.',
      inventorySha256=b.sha(src/'inventory.json'),encoderSha256=b.sha(OUT/'encoder.json'),files={name:b.sha(OUT/name) for name in ['items.json','reference.json','development-wardrobes.json','test-wardrobes.json']})
    b.write(OUT/'protocol.json',protocol);print('POG DISJOINT SOURCE RESERVATION',counts,'AVAILABLE',protocol['availabilityPassed'],flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['reserve']);globals()[ap.parse_args().stage]()
