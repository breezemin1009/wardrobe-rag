"""Expected observed novelty conditioned on existing garments.

V3's failed final outcomes are retained. New targets/model and fresh garment
cohort are frozen separately. No constants or target labels enter inference.
"""
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import argparse,json,os,time
import numpy as np
from backend.thesis_study_20261023 import ROOT,source,read,write,sha,h
from backend import thesis_adaptive_policy_20261024 as prior
OUT=ROOT/'conditional-final-20261025';SEED=2026102501

def prepare():
    if OUT.exists():raise RuntimeError('New study already exists')
    items,edges,g,rev,_,_=source();io={p:set(v) for p,v in read(ROOT/'raw-outfit-incidence-audit-20261023/item-outfits.json').items()}
    eo={tuple(k.split(':')):set(v) for k,v in read(ROOT/'adversarial-audit-20261023/source-edge-outfits.json').items()}
    previous=sum([read(prior.OLD/f'{s}-inputs.json') for s in ['training','development','test']],[])
    used={p for c in previous for p in c['contexts']+c['owned']+c['candidates']}
    used.update(p for c in read(prior.OUT/'test-frames.json') for p in c['contexts']+c['bottoms'])
    # Warm context garments are legitimate inputs: exclude every training/
    # development bottom and its full source outfits from final bottom selection.
    # Previously inspected test outfits are not declared globally untouched.
    trainbottoms={p for stage in ['training','development'] for c in read(prior.OUT/f'{stage}-frames.json') for p in c['bottoms']}
    usedout=set().union(*(io[p] for p in trainbottoms))
    ref=read(prior.OUT/'reference.json');pools={(items[r['top']]['sourceCategory'],items[r['top']]['kind'],items[r['bottom']]['sourceCategory'],items[r['bottom']]['kind']) for r in ref}
    pool=defaultdict(list);key=lambda p:(len(io[p]),h(f'{SEED}:{p}'))
    for b in rev:
        if b not in used and not io[b]&usedout:pool[(items[b]['sourceCategory'],items[b]['kind'])].append(b)
    for cat in pool:pool[cat].sort(key=key)
    occupied=set();seen=set();frames=[];rejects=Counter()
    def bok(b,local):return b not in used and b not in seen and b not in local and not io[b]&usedout and not io[b]&occupied
    def tok(q,local):return q not in seen and q not in local and not io[q]&occupied
    for q0 in sorted(g,key=key):
        if not tok(q0,set()):continue
        cats=defaultdict(list)
        for b in g[q0]:
            if bok(b,{q0}):cats[(items[b]['sourceCategory'],items[b]['kind'])].append(b)
        cats=[cat for cat,bs in cats.items() if len(bs)>=2]
        if not cats:continue
        cat=min(cats,key=lambda c:h(f'{SEED}:{q0}:{c}'));bs=sorted(cats and [b for b in g[q0] if (items[b]['sourceCategory'],items[b]['kind'])==cat and bok(b,{q0})],key=key)
        owned=[bs[0]];candidates=[bs[1]];contexts=[q0];local={q0,*owned,*candidates}
        for b in pool[cat]:
            if not bok(b,local):continue
            qs=[q for q in rev[b] if tok(q,local) and (items[q]['sourceCategory'],items[q]['kind'],*cat) in pools]
            if not qs:continue
            q=min(qs,key=key);candidates.append(b);contexts.append(q);local.update([q,b])
            if len(contexts)==3:break
        if len(contexts)!=3:rejects['three-contexts']+=1;continue
        for degree in [len(rev[b]) for b in candidates]:
            controls=[b for b in pool[cat] if len(rev[b])==degree and bok(b,local)]
            if not controls:break
            b=min(controls,key=lambda p:h(f'{SEED}:control:{len(frames)}:{len(candidates)}:{p}'));candidates.append(b);local.add(b)
        if len(candidates)!=6:rejects['degree-control']+=1;continue
        rng=np.random.default_rng(int(h(f'{SEED}:order:{len(frames)}')[:12],16));candidates=rng.permutation(candidates).tolist();contexts=rng.permutation(contexts).tolist();bottoms=candidates+owned
        covered=set().union(*(rev[b] for b in candidates[4:]+owned));rich=set().union(*(rev[b] for b in candidates[:4]))-covered-set(contexts)
        valid=lambda q:tok(q,local) and (items[q]['sourceCategory'],items[q]['kind'],*cat) in pools
        rich=sorted([q for q in rich if valid(q)],key=key)[:4]
        if len(rich)!=4:rejects['rich-capacity']+=1;continue
        randoms=sorted([q for q in items if items[q]['side']=='top' and q not in set(contexts+rich) and valid(q)],key=lambda p:(len(io.get(p,())),h(f'{SEED}:random:{len(frames)}:{p}')))[:4]
        if len(randoms)!=4:rejects['random-capacity']+=1;continue
        master=rng.permutation(contexts+rich+randoms).tolist();index={q:i for i,q in enumerate(master)}
        views={name:dict(q=[index[q] for q in qs],c=cs,w=ws) for name,qs,cs,ws in [
            ('basic',contexts,list(range(6)),[6]),('owned-basic',contexts,list(range(4)),[4,5,6]),('no-owned',contexts,list(range(6)),[]),
            ('rich',contexts+rich,list(range(6)),[6]),('owned-rich',contexts+rich,list(range(4)),[4,5,6]),('random-contexts',contexts+randoms,list(range(6)),[6]) ]}
        obs=np.array([[int(b in g[q]) for q in master] for b in bottoms]);v=views['owned-rich'];gain=(obs[np.ix_(v['c'],v['q'])]*(1-obs[np.ix_(v['w'],v['q'])].max(0))).sum(1)
        if gain.max()<2:rejects['multiple-gain']+=1;continue
        allids=set(master+bottoms);oo=set().union(*(io[p] for p in allids));assert not occupied&oo
        frames.append(dict(id=f'conditional-fresh-{len(frames):04}',contexts=master,bottoms=bottoms,category=list(cat),views=views,observed=obs.tolist()))
        seen.update(allids);occupied.update(oo)
    finalbottoms={p for c in frames for p in c['bottoms']};fo=set().union(*(io[p] for p in finalbottoms)) if finalbottoms else set()
    refs=[r for r in ref if not eo[(r['top'],r['bottom'])]&fo];newpools={(items[r['top']]['sourceCategory'],items[r['top']]['kind'],items[r['bottom']]['sourceCategory'],items[r['bottom']]['kind']) for r in refs}
    eligible=[c for c in frames if all((items[q]['sourceCategory'],items[q]['kind'],*c['category']) in newpools for q in c['contexts'])]
    OUT.mkdir()
    for n in ['items.json','vectors.npy','itemIds.json','encoder.json']:os.link(prior.OUT/n,OUT/n)
    write(OUT/'reference.json',refs);write(OUT/'test-frames.json',eligible)
    for stage in ['training','development']:
        rows=read(prior.OUT/f'{stage}-frames.json');rows=[c for c in rows if all((items[q]['sourceCategory'],items[q]['kind'],*c['category']) in newpools for q in c['contexts'])]
        write(OUT/f'{stage}-frames.json',rows)
    protocol=dict(createdAt=datetime.now(timezone.utc).isoformat(),seed=SEED,files={n:sha(OUT/n) for n in ['items.json','reference.json']},
        counts=dict(training=len(read(OUT/'training-frames.json')),development=len(read(OUT/'development-frames.json')),testConstructed=len(frames),testEligible=len(eligible),freshBottomPool=sum(map(len,pool.values())),referencePairs=len(refs)),rejects=dict(rejects),
        primary='Per-parent average normalized observed additional-link gain across six applicable views. Fresh final has 3/7 context tops and6/4 candidates; training has4/8 context tops and8/6 candidates. This is a predefined input-size transfer test, not the same cohort as earlier179.',
        method='Learn pair support from every training master context, using parent-cross-fitting for second-stage training. Estimate whether an existing bottom shares the context conditional on a candidate link; multiply by candidate support to estimate added observed links. All variables are titles/retrieved evidence, never IDs/source degrees/labels.',
        constraints='No-owned M*=C; already-owned M*=0; canonical duplication and order invariance; adding owned garments must not raise raw M*. These are structural checks, not accuracy proof.',
        exclusions='Fresh candidate/owned bottom IDs excluded from all1144V2 and179V3 prior primary inputs. Their full source outfits are disjoint from all training/development bottom source outfits and from reference. Final master garments/outfits are disjoint between final parents. Context tops may be known. Previous diagnostic test outfits can recur via different, previously untested bottom IDs; therefore this is an internal fresh-candidate test on an inspected source, never an untouched external corpus.',
        comparison='Same-support C, development-selected C, old kernel C/M, positive-difference M, probabilistic residual, independent novelty controls, learned ranker, no-retrieval ablation. Select using development only; retain all earlier failed final outcomes.',
        freeze='Detailed candidate budget and final acceptance plan must be written before fresh predictions; no threshold changes after opening.',codeSha256=sha(__file__))
    write(OUT/'protocol.json',protocol);(OUT/'frozen-preparation.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(protocol['counts']),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare']);a=p.parse_args();globals()[a.stage]()
