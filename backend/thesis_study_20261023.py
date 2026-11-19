"""Outcome-blind audit and independently reserved extension of the frozen study.

Existing study files are never changed. This module contains a two-stage lock:
prepare freezes cohorts and methods before evaluate computes any predictions.
"""
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
import argparse, json, hashlib, time, random
import numpy as np

ROOT = Path(__file__).parent / 'data/menswear-expansion-v3/pog'
PARENT = ROOT / 'refined-study-v3-scope-audited'
OUT = ROOT / 'thesis-quality-20261023'
SEED = 2026102301

def read(p): return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def h(x): return hashlib.sha256(str(x).encode()).hexdigest()
def write(p, x): Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2), encoding='utf8')

def source():
    from backend.prepare_pog_menswear import classify
    items = read(PARENT/'items.json')
    # The previous correction removed six items from the reference, but retained
    # them in the cached encoder. Exclude them from the new cohort as well.
    items = {k:v for k,v in items.items() if classify(v['title'])[0] is not None}
    aliases = {p:k for k,v in items.items() for p in v['aliases']}
    edges = set()
    for r in read(ROOT/'research-pairs.json'):
        if r['top'] in aliases and r['bottom'] in aliases:
            edges.add((aliases[r['top']], aliases[r['bottom']]))
    graph, rev = defaultdict(set), defaultdict(set)
    for t,b in edges: graph[t].add(b); rev[b].add(t)
    reference = read(PARENT/'reference.json')
    exposed = {r[s] for r in reference for s in ['top','bottom']}
    studies = [ROOT/'cold-study-v2', ROOT/'refined-study-v3']
    for p in studies:
        for rows in read(p/'cohorts.json').values():
            for c in rows: exposed.update(c['contexts']+c['owned']+c['candidates'])
    return items, edges, graph, rev, reference, exposed

def audit():
    items, edges, g, rev, ref, exposed = source()
    reserve = set(items)-exposed
    remaining = {(t,b) for t,b in edges if t in reserve and b in reserve}
    deg = Counter()
    for t,b in remaining: deg[b]+=1
    report = dict(items=len(items), edges=len(edges), reference=len(ref),
        exposed=len(exposed), reserve=len(reserve), remainingEdges=len(remaining),
        remainingBottomDegrees=dict(Counter(deg.values())),
        remainingTopKinds=dict(Counter(items[t]['kind'] for t,b in remaining)),
        remainingBottomKinds=dict(Counter(items[b]['kind'] for t,b in remaining)))
    print(json.dumps(report, ensure_ascii=False), flush=True)

def prepare():
    if OUT.exists(): raise RuntimeError('Immutable study directory already exists')
    items, edges, g, rev, ref, exposed = source()
    reserve = set(items)-exposed
    refpools = {(items[r['top']]['sourceCategory'],items[r['top']]['kind'],items[r['bottom']]['sourceCategory'],items[r['bottom']]['kind']) for r in ref}
    pools=defaultdict(list)
    for b in sorted(reserve):
        if items[b]['side']=='bottom' and rev[b]&reserve:
            pools[(items[b]['sourceCategory'],items[b]['kind'])].append(b)
    used=set();blocks=[];reasons=Counter()
    for cat, pool in sorted(pools.items()):
        pool=sorted(pool,key=lambda p:h(f'{SEED}:bottom:{p}'))
        for seed in pool:
            if seed in used:continue
            local=set();owned=[];contexts=[];anchors=[]
            # Four owned garments with one recorded context each, followed by
            # four additional independent recorded contexts. No model scores
            # enter cohort construction, acceptance or ordering.
            candidates_for_anchor=[seed]+[b for b in pool if b!=seed]
            for b in candidates_for_anchor:
                if b in used or b in local:continue
                ts=[t for t in rev[b]&reserve if t not in used|local and (items[t]['sourceCategory'],items[t]['kind'],*cat) in refpools]
                if not ts:continue
                t=min(ts,key=lambda p:h(f'{SEED}:{b}:{p}'))
                local.update([t,b]);contexts.append(t)
                (owned if len(owned)<4 else anchors).append(b)
                if len(anchors)==4:break
            if len(anchors)<4:reasons['insufficient-independent-anchor']=reasons.get('insufficient-independent-anchor',0)+1;break
            # Candidate records on already covered contexts are admitted when
            # available, without requiring equal total support or a unique answer.
            redundant=[]
            for t in contexts[:4]:
                options=[b for b in g[t]&reserve if b not in used|local and (items[b]['sourceCategory'],items[b]['kind'])==cat]
                if options:
                    b=min(options,key=lambda p:h(f'{SEED}:redundant:{p}'));redundant.append(b);local.add(b)
            candidate_pool=anchors+redundant
            matched=0;draws=0
            while len(candidate_pool)<16:
                degree=len(rev[anchors[draws%4]]);draws+=1
                options=[b for b in pool if b not in used|local and len(rev[b])==degree]
                if options:matched+=1
                else:options=[b for b in pool if b not in used|local]
                if not options:break
                b=min(options,key=lambda p:h(f'{SEED}:control:{len(blocks)}:{draws}:{p}'));candidate_pool.append(b);local.add(b)
            if len(candidate_pool)<16:reasons['insufficient-candidate-pool']+=1;break
            # Keep all four independently selected anchors in every candidate
            # size. Additional candidates follow a fixed outcome-blind order.
            random.Random(SEED+len(blocks)).shuffle(contexts)
            random.Random(SEED+100000+len(blocks)).shuffle(owned)
            random.Random(SEED+200000+len(blocks)).shuffle(anchors)
            extra=[b for b in candidate_pool if b not in anchors]
            random.Random(SEED+300000+len(blocks)).shuffle(extra)
            candidates=anchors+extra
            obs=[[int(b in g[t]) for t in contexts] for b in candidates]
            history=[[int(b in g[t]) for t in contexts] for b in owned]
            blocks.append(dict(id=f'extension-{len(blocks):04}',contexts=contexts,owned=owned,candidates=candidates,
                observed=obs,history=history,category=list(cat),matchedControlDraws=matched,controlDraws=draws))
            used.update(local)
    variants={'primary':dict(q=8,w=2,n=8),'contexts4':dict(q=4,w=2,n=8),'owned1':dict(q=8,w=1,n=8),
        'owned4':dict(q=8,w=4,n=8),'candidates4':dict(q=8,w=2,n=4),'candidates16':dict(q=8,w=2,n=16),
        'noOwned':dict(q=8,w=0,n=8)}
    OUT.mkdir()
    write(OUT/'blocks.json',blocks)
    parent_selection=read(PARENT/'selection.json')
    # Strong simple alternatives are selected on OLD development only; the
    # independent extension remains unopened until this lock is written.
    from backend.pog_refined_analysis import select
    fs=read(PARENT/'development-features.json');dev=read(PARENT/'cohorts.json')['development']
    alternatives={}
    for name in ['char-novelty','dense-novelty','product-residual']:
        reduced={k:[{'C':r[name], 'positive-difference':r[name]} for r in rows] for k,rows in fs.items()}
        prof=select(dev,reduced,True);prof['policy']={'alpha':0,'marginal':name};alternatives[name]=prof
    lock=dict(createdAt=datetime.now(timezone.utc).isoformat(),seed=SEED,
        design='Prospective reserved garment-disjoint constructed wardrobe blocks; 8 tops, 4 owned bottoms, 16 candidate bottoms; nested variants share a block and are never counted as independent observations.',
        sourceBoundary='Original POG recorded links only; unobserved is not incompatible. Candidate anchors make this a recorded-link recovery test, not random consumer wardrobes or full-catalog recommendation.',
        previousExposure='Excludes every reference garment and every development/evaluation scenario garment of cold-study-v2 and refined-study-v3; exact title/image aliases canonicalized upstream.',
        blocks=len(blocks),garments=len(used),sourceActiveItems=len(items),sourceEdges=len(edges),reservedItems=len(reserve),
        cohortRejects=dict(reasons),variants=variants,
        proposed=parent_selection['proposed'],strongestC=parent_selection['strongestC'],alternatives=alternatives,
        ablation='Same frozen K=96 hybrid kernel without density normalization; 3 degree-preserving within-category reference bottom permutations with seeds 0,1,2; M computed with identical contexts/owned/candidates.',
        methods='proposed, same-C, strongest-C, old-2:1, char-novelty, dense-novelty, product-residual, random, no-density, shuffled-reference',
        primaryMetric='Expected recorded additional links recovered by top-1, normalized by the best candidate recorded gain; ties averaged, no-positive-gain cases excluded from this metric and counted explicitly.',
        secondaryMetrics='Top-1 maximizer hit, selected additional-link count, selected total-link count, regret, NDCG@all using linear gains and tie-expectation discounts.',
        inference='6000 paired block bootstrap; 20000 paired sign permutations; Holm over the 9 primary comparisons; all size variants descriptive with paired intervals and no independent-replication claim.',
        audit='Zero overlap against old exposed garments; mutually disjoint blocks; alias zero overlap; source membership only for construction and targets; model inputs titles only; shuffled reference degree preserved; zero-owned M=C and exact-owned M=0.',
        decision='No retuning or model adoption based on these results. Report every locked method and variant; no stopping rule based on performance.',
        parentFiles={n:sha(PARENT/n) for n in ['protocol.json','selection.json','reference.json','encoder.json','vectors.npy','itemIds.json']},
        blocksSha256=sha(OUT/'blocks.json'),codeSha256=sha(__file__))
    write(OUT/'protocol.json',lock)
    (OUT/'frozen-study.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps({k:lock[k] for k in ['blocks','garments','sourceActiveItems','sourceEdges','reservedItems','cohortRejects','alternatives']},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['audit','prepare']);a=p.parse_args()
    globals()[a.stage]()
