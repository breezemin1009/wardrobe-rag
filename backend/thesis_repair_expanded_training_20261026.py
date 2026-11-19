"""One further fixed development round with a fresh, source-disjoint test.

The previous 1200-case result is preserved and becomes development here.
Additional supervised pairs come only from source/reference records disjoint
from every core development and new final-test source neighbourhood.
"""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,time,shutil,json
import numpy as np
from threadpoolctl import threadpool_limits
from backend import thesis_repair_study_20261026 as first
from backend import thesis_repair_models_20261026 as base
from backend import thesis_repair_pair_kernel_20261026 as kernel

OLD=first.OUT;OUT=OLD.parent/'expanded-learning-final'
read=base.read;write=base.write;sha=base.sha
def key(s):return base.stable('2026102641:'+str(s))

def reserve():
    first.verify();assert read(OLD/'result-audit.json')['allPassed']
    if OUT.exists():raise RuntimeError('Second study already reserved')
    items=read(OLD/'items.json');edges=read(first.ROOT/'expanded-projected-20261026/edges.json')
    io={p:{first.oid(o) for o in os} for p,os in read(first.ROOT/'expanded-full-source-incidence-20261026.json').items()}
    prior=read(OLD.parent/'prior-exposure.json');core=read(OLD/'frames.json')+read(OLD/'test-frames.json')
    used=set(prior['ids'])|{p for c in core for p in c['contexts']+c['bottoms']}
    occupied=set().union(*(io[p] for p in used));eligible={p for p in items if p not in used and not io[p]&occupied}
    # The earlier cold pool has been largely consumed. Reserve additional cold
    # targets from the remaining original source neighbourhoods, before any
    # fitting, and rebuild one common reference for ALL compared methods.
    cold={p for p in eligible if items[p]['side']=='bottom' and int(key('cold:'+p)[:8],16)%10<7}
    coldout=set().union(*(io[p] for p in cold))
    ref=[r for r in read(OLD/'reference.json') if r['bottom'] not in cold and not {first.oid(o) for o in r['outfits']}&coldout]
    g=defaultdict(set);rev=defaultdict(set);rg=defaultdict(set);rr=defaultdict(set)
    for r in edges:g[r['top']].add(r['bottom']);rev[r['bottom']].add(r['top'])
    for r in ref:rg[r['top']].add(r['bottom']);rr[r['bottom']].add(r['top'])
    assert not set(r['bottom'] for r in ref)&cold
    candidates=defaultdict(list);degree=defaultdict(list);order=lambda p:(len(io[p]),key(p))
    for p in cold&set(rev):candidates[items[p]['kind']].append(p);degree[items[p]['kind'],len(rev[p])].append(p)
    for pools in [candidates,degree]:
        for ps in pools.values():ps.sort(key=order)
    seen=set(used);taken=set(occupied);frames=[];reject=Counter();second_order={}
    fill_tops=sorted(rg,key=lambda p:key('qfill:'+p))
    okay=lambda p,local:p in eligible and p not in seen and p not in local and not io[p]&taken
    for a in sorted(cold&set(rev),key=order):
        if not okay(a,set()):continue
        kind=items[a]['kind'];nq=[3,5,8][int(key('q:'+a)[:8],16)%3];nw=[0,1,3][int(key('w:'+a)[:8],16)%3]
        local={a};cs=[a]
        sk=(kind,len(rev[a]))
        if sk not in second_order:second_order[sk]=sorted(candidates[kind],key=lambda p:(abs(len(rev[p])-len(rev[a])),order(p)))
        for p in second_order[sk]:
            if okay(p,local) and any(okay(q,local|{p}) for q in rev[p]):cs.append(p);local.add(p);break
        if len(cs)!=2:reject['second-seed']+=1;continue
        for seed in list(cs):
            for p in degree[kind,len(rev[seed])]:
                if okay(p,local):cs.append(p);local.add(p);break
        if len(cs)!=4:reject['degree-controls']+=1;continue
        qs=[]
        for seed in cs[:2]:
            for q in sorted(rev[seed],key=order):
                if okay(q,local):qs.append(q);local.add(q);break
        if len(qs)!=2:reject['seed-contexts']+=1;continue
        for q in sorted(set().union(*(rev[p] for p in cs))-set(qs),key=lambda p:key('qextra:'+p)):
            if okay(q,local):qs.append(q);local.add(q)
            if len(qs)==nq:break
        if len(qs)<nq:
            for q in fill_tops:
                if okay(q,local):qs.append(q);local.add(q)
                if len(qs)==nq:break
        if len(qs)<nq:reject['top-count']+=1;continue
        ws=[];coverage=set();pool={w for q in qs for w in rg[q] if items[w]['kind']==kind}
        for _ in range(nw):
            options=[w for w in pool if okay(w,local) and all((w in g[q])==(w in rg[q]) for q in qs)]
            if not options:break
            w=min(options,key=lambda w:(-len((rr[w]&set(qs))-coverage),order(w)))
            ws.append(w);local.add(w);coverage.update(rr[w]&set(qs))
        if len(ws)<nw:reject['owned-count']+=1;continue
        cs=np.random.default_rng(int(key('shuffle:'+a)[:10],16)).permutation(cs).tolist()
        ids=set(qs+cs+ws);os=set().union(*(io[p] for p in ids));assert not ids&seen and not os&taken
        obs=[[int(p in g[q]) for q in qs] for p in cs+ws];hist=[[int(p in rg[q]) for q in qs] for p in cs+ws]
        assert not np.asarray(hist)[:4].any() and hist[4:]==obs[4:]
        frames.append(dict(id=key('test:'+':'.join(sorted(cs)))[:20],stage='test',contexts=qs,bottoms=cs+ws,kind=kind,
            observed=obs,knownHistory=hist,sourceOutfits=sorted(os),views={'primary':dict(q=list(range(nq)),c=list(range(4)),w=list(range(4,4+nw)))},
            construction=dict(topCount=nq,ownedCount=nw,knownCoverage=int(np.max(hist[4:],axis=0).sum()) if ws else 0)))
        seen.update(ids);taken.update(os)
        if len(frames)>=1600:break
    assert len(frames)>=500,('Insufficient independent input; never substitute an old scored input',len(frames))
    # Source-pair learning is separated from both parent development folds and
    # the newly reserved final test. Original source neighbourhoods, not only
    # exact item identifiers, are excluded.
    forbidden=taken;pairpool=defaultdict(list)
    for r in ref:
        q=r['top'];c=r['bottom'];kind=items[c]['kind']
        if q not in seen and c not in seen and not (io[q]|io[c])&forbidden:
            pairpool[kind].append(r)
    extra=[];extra_ids=set();extra_origins=set();topcount=Counter();bottomcount=Counter()
    allowedBottoms=defaultdict(list);negativeByDegree=defaultdict(list)
    for c in rr:
        if c not in seen and not io[c]&forbidden:allowedBottoms[items[c]['kind']].append(c)
    for ps in allowedBottoms.values():ps.sort(key=key)
    for kind,ps in allowedBottoms.items():
        for b in ps:negativeByDegree[kind,len(rr[b])].append(b)
    for kind in ['trousers','jeans','shorts']:
        n=0
        for r in sorted(pairpool[kind],key=lambda r:key('positive:'+r['top']+r['bottom'])):
            q=r['top'];c=r['bottom']
            if topcount[q]>=1 or bottomcount[c]>=3:continue
            b=None
            # Nearest degree, then a query-hash rotation of a hash-sorted pool.
            ds=sorted({d for k,d in negativeByDegree if k==kind},key=lambda d:(abs(d-len(rr[c])),d))
            for d in ds:
                ps=negativeByDegree[kind,d];start=int(key('negative:'+q)[:8],16)%len(ps)
                for offset in range(len(ps)):
                    option=ps[(start+offset)%len(ps)]
                    if option not in g[q]:b=option;break
                if b is not None:break
            if b is None:continue
            extra.extend([dict(q=q,c=c,y=1,pairGroup=key(q+c)[:20]),dict(q=q,c=b,y=0,pairGroup=key(q+c)[:20])])
            extra_ids.update([q,c,b]);extra_origins.update(io[q]|io[c]|io[b]);topcount[q]+=1;bottomcount[c]+=1;n+=1
            if n>=1500:break
    assert not extra_ids&seen and not extra_origins&taken and len(extra)>=3000,(len(frames),len(extra),len(extra_ids&seen),len(extra_origins&taken),{k:len(v) for k,v in pairpool.items()})
    OUT.mkdir(parents=True)
    shutil.copyfile(OLD/'items.json',OUT/'items.json');write(OUT/'reference.json',ref)
    pairs={(r['top'],r['bottom']) for r in ref}
    for c in core:c['knownHistory']=[[int((q,b) in pairs) for q in c['contexts']] for b in c['bottoms']]
    write(OUT/'core-frames.json',core);write(OUT/'test-frames.json',frames);write(OUT/'extra-pairs.json',extra)
    p=dict(createdAt=base.stamp(),purpose='Previous445+1200parents now development; one additional fixed comparison on new source-disjoint inputs. Prior test result preserved, never called independent final evidence here.',
        codeSha256=sha(__file__),kernelSha256=sha(kernel.__file__),
        previousReferencePath=str(OLD),previousReferenceSha256=sha(OLD/'reference.json'),
        previousResultSha256=sha(OLD/'test-results.json'),previousAuditSha256=sha(OLD/'result-audit.json'),
        priorExposureSha256=sha(OLD.parent/'prior-exposure.json'),
        files={n:sha(OUT/n) for n in ['items.json','reference.json','core-frames.json','test-frames.json','extra-pairs.json']},
        sourceFiles={n:sha(first.ROOT/n) for n in ['expanded-projected-20261026/items.json','expanded-projected-20261026/edges.json','expanded-full-source-incidence-20261026.json']},
        finalCounts=dict(parents=len(frames),kinds=dict(Counter(c['kind'] for c in frames)),topCounts=dict(Counter(len(c['contexts']) for c in frames)),ownedCounts=dict(Counter(len(c['bottoms'])-4 for c in frames)),rejections=dict(reject)),
        extraCounts=dict(rows=len(extra),positive=sum(x['y'] for x in extra),distinctIDs=len(extra_ids),completeSourceOutfits=len(extra_origins)),
        sourcePairAvailability='Requested up to1500positive pairs per kind. Full neighbourhood disjointness leaves fewer eligible source pairs; retain the resulting1643positives and matched1643unobserved pairs. This availability adjustment preceded all second-round model fitting and test predictions.',
        referenceEdges=len(ref),coreParents=len(core),
        referenceRule='Additional70%hash pool among fully unused source-neighbourhood candidates is cold; remove its complete original outfits from the earlier reference. Recompute every core and final retrieval feature on this single new reference.',
        trainingWeights='Each core parent total20 (each pair5/Qcount); extra-pair corpus total equals core corpus total, preventing row-count dominance. No outcome-tuned weights.',
        models=[dict(view=view,family=fam) for view in ['compact','wide'] for fam in ['LR_C.1','LR_C1','HGB_leaf7','HGB_leaf15','HGB_leaf31']],
        features='compact:noRAG dense/char cos; RAG these2+24bidirectional kernels. wide:noRAG514title features; RAG538title+kernel features. Same10models per path.',
        HGB='200iterations,learningrate.05,L2=10,minleaf20,noearlystop,seed2026102641',
        folds='5 source-disjoint core parent folds hash2026102641; extra corpus disjoint from all core/fullsource outfits and included in every fitting fold; one NG contribution per parent; best OOF NG,then logloss,then smalleralpha',
        alphas=[0,.1,.25,.5],directModels=['dense','char','hybrid'],
        primaryComparisons=['noRAG','direct','oldRAG','sameSupportC'],
        statistics='Paired bootstrap10000 and signflip50000; Holm4 followed by factor2 for two development/test rounds. Final gate: positive difference, lower95%CI>0 and min(1,2*HolmP)<.05. No new round will be treated as if the earlier failure did not occur.',
        finalConstruction='Same four-candidate record-selection task without forced overlap/new conflict; Q3/5/8,W0/1/3,zero-gain retained; one input per parent. Degree-matched controls, greedy existing history from visible reference only. Not human wardrobes.',
        maximumFinalEvaluationRoundsInThisRepair=2,noExternalPreregistration=True)
    write(OUT/'protocol.json',p)
    print('SECOND STUDY RESERVED',json.dumps(dict(final=p['finalCounts'],extra=p['extraCounts']),ensure_ascii=False),flush=True)

def verify():
    p=read(OUT/'protocol.json');assert p['codeSha256']==sha(__file__) and p['kernelSha256']==sha(kernel.__file__)
    for n,v in p['files'].items():assert sha(OUT/n)==v
    assert sha(OLD/'reference.json')==p['previousReferenceSha256']
    return p

def encode():
    from backend.pog_retrieval import MODEL,REVISION,text
    from sentence_transformers import SentenceTransformer
    import torch
    verify();assert not (OUT/'vectors.npy').exists()
    items=read(OUT/'items.json');frames=read(OUT/'core-frames.json')+read(OUT/'test-frames.json');ref=read(OUT/'reference.json');extra=read(OUT/'extra-pairs.json')
    needed={r[s] for r in ref for s in ['top','bottom']}|{p for c in frames for p in c['contexts']+c['bottoms']}|{r[s] for r in extra for s in ['q','c']}
    oldids=read(OLD/'itemIds.json');oldV=np.load(OLD/'vectors.npy');cache={p:oldV[i] for i,p in enumerate(oldids) if p in needed};missing=sorted(needed-set(cache))
    if missing:
        torch.set_num_threads(4);m=SentenceTransformer(MODEL,revision=REVISION,device='cpu',local_files_only=True,trust_remote_code=False)
        v=m.encode([text(items[p]) for p in missing],batch_size=192,normalize_embeddings=True,show_progress_bar=True,convert_to_numpy=True);cache.update(zip(missing,v))
    ids=sorted(needed);np.save(OUT/'vectors.npy',np.asarray([cache[p] for p in ids]));write(OUT/'itemIds.json',ids)
    write(OUT/'encoder.json',dict(model=MODEL,revision=REVISION,items=len(ids),newTitles=len(missing),vectorsSha256=sha(OUT/'vectors.npy'),idsSha256=sha(OUT/'itemIds.json')))
    print('SECOND STUDY ENCODED',len(ids),'new',len(missing),flush=True)

def core_features():
    from backend.pog_refined_retrieval import Retriever
    verify();assert not (OUT/'core-features.npz').exists();threadpool_limits(1)
    engine=Retriever(OUT);frames=read(OUT/'core-frames.json');allx={};start=time.monotonic()
    for i,c in enumerate(frames):
        x=kernel.base_features(engine,c['contexts'],c['bottoms'][:4]);f,r,_=kernel.support_features(engine,c['contexts'],c['bottoms'][:4])
        allx.update({f'X{i}':x,f'F{i}':f,f'R{i}':r})
        if i%100==0:print('EXPANDED CORE FEATURES',i,'/',len(frames),'seconds',round(time.monotonic()-start,1),flush=True)
    np.savez_compressed(OUT/'core-features.npz',**allx)
    write(OUT/'core-feature-lock.json',dict(featuresSha256=sha(OUT/'core-features.npz'),protocolSha256=sha(OUT/'protocol.json'),parents=len(frames)))
    print('EXPANDED CORE SEALED',len(frames),flush=True)

def extra_features():
    from backend.pog_refined_retrieval import Retriever
    verify();assert not (OUT/'extra-features.npz').exists();threadpool_limits(1)
    engine=Retriever(OUT);rows=read(OUT/'extra-pairs.json');xx=[];ff=[];rr=[];start=time.monotonic()
    # Two rows with a common query are evaluated together, while each candidate
    # has its own exclusion mask. Labels never enter these features.
    for i in range(0,len(rows),2):
        pair=rows[i:i+2];assert pair[0]['q']==pair[1]['q']
        qs=[pair[0]['q']];cs=[r['c'] for r in pair]
        x=kernel.base_features(engine,qs,cs);f,r,_=kernel.support_features(engine,qs,cs)
        xx.extend(x[:,0]);ff.extend(f[:,0]);rr.extend(r[:,0])
        if i%500==0:print('SOURCE PAIR FEATURES',i,'/',len(rows),'seconds',round(time.monotonic()-start,1),flush=True)
    np.savez_compressed(OUT/'extra-features.npz',X=np.asarray(xx),F=np.asarray(ff),R=np.asarray(rr))
    write(OUT/'extra-feature-lock.json',dict(featuresSha256=sha(OUT/'extra-features.npz'),protocolSha256=sha(OUT/'protocol.json'),rows=len(rows)))
    print('EXTRA SOURCE FEATURES SEALED',len(rows),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['reserve','encode','core_features','extra_features'])
    globals()[parser.parse_args().stage]()
