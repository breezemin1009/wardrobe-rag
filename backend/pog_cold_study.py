"""Item-family-disjoint menswear case-retrieval and marginal-utility study.

Preparation is outcome-blind: it freezes source associations and deterministic
query cohorts. A missing association is unobserved, not judged ugly. The target
is recovery of held-out recorded outfit links and their additional coverage.
"""
from collections import Counter,defaultdict
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,random,re
import numpy as np
from backend.prepare_pog_menswear import classify
from backend.data_io import write_json_atomic as write

ROOT=Path(__file__).parent/'data/menswear-expansion-v3/pog'
OUT=ROOT/'cold-study-v2'
SEED=2026102119
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def h(s):return hashlib.sha256(str(s).encode()).hexdigest()
def stable(x):return json.dumps(x,sort_keys=True,separators=(',',':'))

def prepare(out):
    if out.exists():raise ValueError('Fresh immutable output directory required')
    items=read(ROOT/'research-items.json');original=read(ROOT/'research-pairs.json')
    audit=read(ROOT/'menswear-audit.json');category_sides={}
    for category,counts in audit['categoryEvidence'].items():
        sides=Counter()
        for key,n in counts.items():sides[key.split('/')[0]]+=n
        side,n=sides.most_common(1)[0]
        if sum(sides.values())>=100 and n/sum(sides.values())>=.995:category_sides[category]=side
    items={k:v for k,v in items.items() if classify(v['title'])[0] and category_sides.get(v['sourceCategory'])==v['side']}
    # Image-URL and exact normalized-title duplicates form transitive families.
    parent={k:k for k in items}
    def find(k):
        while parent[k]!=k:parent[k]=parent[parent[k]];k=parent[k]
        return k
    image={};titles={}
    for k,v in items.items():
        for table,identity in [(image,v['family']),(titles,re.sub(r'\s+','',v['title']).casefold())]:
            if identity in table:parent[find(k)]=find(table[identity])
            else:table[identity]=k
    aliases=defaultdict(list)
    for k in items:aliases[find(k)].append(k)
    canon={p:min(group) for group in aliases.values() for p in group}
    canonical={}
    for k,v in items.items():canonical.setdefault(canon[k],{**items[canon[k]],'family':h(canon[k]),'aliases':aliases[find(k)]})
    pairs={}
    for row in original:
        if any(row[s] not in items for s in ['top','bottom']):continue
        t,b=canon[row['top']],canon[row['bottom']]
        if canonical[t]['side']!='top' or canonical[b]['side']!='bottom':continue
        pairs.setdefault((t,b),dict(top=t,bottom=b,outfits=[]))['outfits'].extend(row['outfits'])
    used={p for pair in pairs for p in pair};canonical={k:v for k,v in canonical.items() if k in used}
    assigned={pid:('reference' if (bucket:=int(h(f'{SEED}:{pid}')[:10],16)%100)<65 else 'development' if bucket<85 else 'evaluation') for pid in canonical}
    stages={s:[r for (t,b),r in pairs.items() if assigned[t]==assigned[b]==s] for s in ['reference','development','evaluation']}
    graph=defaultdict(set);reverse=defaultdict(set)
    for t,b in pairs:graph[t].add(b);reverse[b].add(t)
    cohorts={};stats={}
    for stage in ['development','evaluation']:
        rows=stages[stage];bottoms=sorted({r['bottom'] for r in rows});pools=defaultdict(list)
        for b in bottoms:pools[(canonical[b]['sourceCategory'],canonical[b]['kind'])].append(b)
        # Reference incidence of EVERY query/owned/candidate garment is zero.
        cases=[];seen=set();used_in_scenarios=set();skipped=Counter()
        for q in sorted(rows,key=lambda r:h(f'{SEED}:{r["top"]}:{r["bottom"]}')):
            t,p=q['top'],q['bottom']
            if t in seen or t in used_in_scenarios or p in used_in_scenarios:continue
            category=(canonical[p]['sourceCategory'],canonical[p]['kind']);pool=pools[category]
            # Match the degree in the complete original source graph as well,
            # preventing source popularity from signalling the answer.
            controls=[b for b in pool if b!=p and b not in graph[t] and len(reverse[b])==len(reverse[p]) and b not in used_in_scenarios]
            controls=sorted(controls,key=lambda b:h(f'{SEED}:{t}:{b}'))
            match=None
            for negative in controls[:160]:
                anchors=[a for a in reverse[negative] if assigned[a]==stage and a!=t and a not in used_in_scenarios]
                for a in sorted(anchors,key=h):
                    owned=[b for b in graph[a] if assigned[b]==stage and b not in {p,negative} and b not in graph[t] and b not in used_in_scenarios and (canonical[b]['sourceCategory'],canonical[b]['kind'])==category]
                    if owned:match=negative,a,sorted(owned,key=h)[0];break
                if match:break
            if not match:skipped['no-same-degree-redundancy-control']+=1;continue
            negative,a,e=match;candidates=[p,negative];random.Random(int(h(t)[:10],16)).shuffle(candidates)
            contexts=[t,a];observed=[[int(b in graph[top]) for top in contexts] for b in candidates];existing=[int(e in graph[top]) for top in contexts]
            if existing!=[0,1]:raise ValueError('Incorrect observed wardrobe history')
            cases.append(dict(id=f'{stage}-{len(cases)}',group=t,contexts=contexts,candidates=candidates,owned=[e],observed=observed,existing=existing,category=category,fullSourceDegree=len(reverse[p])))
            seen.add(t)
            used_in_scenarios.update(contexts+candidates+[e])
        cohorts[stage]=cases;stats[stage]=dict(eligiblePairs=len(rows),cases=len(cases),skipped=dict(skipped))
    out.mkdir(parents=True);write(out/'items.json',canonical);write(out/'reference.json',stages['reference']);write(out/'cohorts.json',cohorts)
    protocol=dict(createdAt=datetime.now(timezone.utc).isoformat(),seed=SEED,source='POG author public release https://github.com/wenyuer/POG',
        sourceFiles={n:sha(ROOT/n) for n in ['item_data.txt.zip','outfit_data.txt.zip','research-items.json','research-pairs.json']},
        input='Explicit male adult-apparel original Chinese product titles only; no outfit membership, item IDs, category hashes, URLs or target labels in embeddings',
        dedup='Transitive closure of identical normalized image URL and normalized title; same family only one stage; source pairs deduplicated',
        split='65/20/15 item-family hash split, reference keeps only internal edges; all query candidates, contexts and owned items come from their own disjoint stage',
        primary='Recovered held-out recorded new links@1, in a constructed wardrobe with one covered and one uncovered context; two same-fine-category and exactly full-source-degree-matched candidates',
        primaryBoundary='Known outfit-link recovery under recorded redundancy, not human preference or purchase conversion; missing links are unobserved controls',
        secondary='Outfit completion on the uncovered context; total recorded support of selected candidate',
        selection='All k, retrieval methods and weights chosen on development only; grouped CV by primary query garment; freeze choices before opening evaluation features',
        grid=dict(k=[8,32,96],retrieval=['dense-conditional','char-conditional','hybrid-conditional','density'],alpha=[0,.1,.2,.3,.4,.5,.6,2/3,.7,.8,.9,1.],marginal=['positive-difference','product-residual']),
        uncertainty='Garment-disjoint primary scenarios (all contexts,candidates,owned); paired scenario bootstrap and sign permutation, Holm over four predefined primary controls; reference collection fixed',
        controls=['same_engine_C','strongest_C','old_2_to_1','random'],
        adoption='Held-out proposed gain over all four primary controls: CI lower bound>0 and Holm p<.05; dense/char retrieval must beat chance in uncovered-context completion; combined-history component clusters >=30; grouped development CV positive in at least4/5folds',
        negativeControls='Candidate-order reversal, exact duplicated wardrobe, shuffled targets, degree equality, zero reference incidence, no stage family overlap',
        corpus=dict(items=len(canonical),pairs=len(pairs),referencePairs=len(stages['reference']),stages=stats),
        amendment='v1 cohorts had shared query garments joining almost all cases into 1 component. v2 reserves all scenario garments at construction. No development or evaluation prediction from v1 was computed.',
        files={n:sha(out/n) for n in ['items.json','reference.json','cohorts.json']},codeSha256=sha(__file__),
        productionApproved=False,photos='POG original URLs currently return 420; title source evidence only until actual images independently acquired; no substitution with unrelated catalog photos')
    write(out/'protocol.json',protocol);(out/'frozen-prepare.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(protocol['corpus'],ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare']);parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args();prepare(args.output)
