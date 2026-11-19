"""Prospective refinement: comparable evidence ranks and garment redundancy.

Retains the failed v2 held-out result. The new final set excludes every garment
in any previously scored scenario; source reference cases remain unchanged.
"""
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import json,random,hashlib
from backend.pog_cold_study import OUT as PREVIOUS,ROOT,read,sha,h,SEED
from backend.data_io import write_json_atomic as write

OUT=ROOT/'refined-study-v3'

def main():
    if OUT.exists():raise ValueError('Fresh directory required')
    items=read(PREVIOUS/'items.json');reference=read(PREVIOUS/'reference.json');old=read(PREVIOUS/'cohorts.json')
    aliases={p:k for k,v in items.items() for p in v['aliases']};full={}
    for r in read(ROOT/'research-pairs.json'):
        if r['top'] not in aliases or r['bottom'] not in aliases:continue
        t,b=aliases[r['top']],aliases[r['bottom']];full.setdefault((t,b),dict(top=t,bottom=b))
    reference_ids={r[s] for r in reference for s in ['top','bottom']};exposed={p for rows in old.values() for c in rows for p in c['contexts']+c['candidates']+c['owned']}
    reserved=set(items)-reference_ids-exposed;rows=[r for (t,b),r in full.items() if t in reserved and b in reserved]
    graph=defaultdict(set);reverse=defaultdict(set)
    for t,b in full:graph[t].add(b);reverse[b].add(t)
    pools=defaultdict(list)
    for b in sorted({r['bottom'] for r in rows}):pools[(items[b]['sourceCategory'],items[b]['kind'],len(reverse[b]))].append(b)
    ref_pools={(items[r['top']]['sourceCategory'],items[r['top']]['kind'],items[r['bottom']]['sourceCategory'],items[r['bottom']]['kind']) for r in reference}
    cases=[];used=set()
    for q in sorted(rows,key=lambda r:h(f'reserved-v3:{r["top"]}:{r["bottom"]}')):
        t,p=q['top'],q['bottom']
        if t in used or p in used:continue
        cat=(items[p]['sourceCategory'],items[p]['kind']);match=None
        if (items[t]['sourceCategory'],items[t]['kind'],*cat) not in ref_pools:continue
        for n in sorted(pools[(*cat,len(reverse[p]))],key=lambda n:h(t+n)):
            if n in used or n==p or n in graph[t]:continue
            for a in sorted(reverse[n]&reserved,key=h):
                if a in used or a==t or (items[a]['sourceCategory'],items[a]['kind'],*cat) not in ref_pools:continue
                owned=[e for e in graph[a]&reserved if e not in used and e not in {n,p} and e not in graph[t] and (items[e]['sourceCategory'],items[e]['kind'])==cat]
                if owned:match=n,a,sorted(owned,key=h)[0];break
            if match:break
        if not match:continue
        n,a,e=match;candidates=[p,n];random.Random(int(h(t)[:12],16)).shuffle(candidates);contexts=[t,a]
        cases.append(dict(id=f'reserved-v3-{len(cases)}',group=t,contexts=contexts,candidates=candidates,owned=[e],observed=[[int(b in graph[top]) for top in contexts] for b in candidates],existing=[int(e in graph[top]) for top in contexts],category=cat,fullSourceDegree=len(reverse[p])))
        used.update(contexts+candidates+[e])
    OUT.mkdir()
    for name in ['items.json','reference.json','vectors.npy','itemIds.json','encoder.json','frozen-encoder-source.py']:(OUT/name).write_bytes((PREVIOUS/name).read_bytes())
    write(OUT/'cohorts.json',dict(development=old['development'],evaluation=cases))
    protocol=dict(createdAt=datetime.now(timezone.utc).isoformat(),previousStudy=str(PREVIOUS),previousReportSha256=sha(PREVIOUS/'evaluation-report.json'),previousFailureRetained=True,
        method='Same source-only retrieval kernels; compare conditional source-evidence percentiles across contexts; separately penalise candidate-to-owned resemblance; choose weights on development only',
        input='Same frozen original Chinese title encoder and reference cases as v2; no neural or compatibility-model training',
        referencePairs=len(reference),developmentCases=len(old['development']),evaluationCases=len(cases),eligibleReservedPairs=len(rows),
        finalIndependence=dict(referenceGarmentOverlap=len(used&reference_ids),previousScenarioGarmentOverlap=len(used&exposed),mutuallyDisjoint=True),
        primary='Recovery of recorded additional co-outfit link in two-context redundancy-present constructed wardrobes; missing pairs are unobserved controls, not incompatibility labels',
        retrievalGrid=dict(mode=['char-conditional','char-density','char-percentile','dense-conditional','density','dense-percentile','hybrid-conditional','hybrid-density','hybrid-percentile'],k=[8,32,96],temperature=.1,percentileReference=256),
        policies=dict(alpha=[0,.1,.2,.3,.4,.5,.6,2/3,.7,.8,.9,1.],marginal=['positive-difference','product-residual','char-novelty','dense-novelty']),
        choice='Development 5-fold validation, then all development; final features unopened until selection lock',
        controls=['same_engine_C','strongest_C','old_2_to_1','random'],
        statistics='6000 paired-scenario bootstrap draws; 20000 paired sign-permutation draws; four-comparison Holm correction; cases use mutually disjoint garments',
        adoption='All four primary comparisons positive CI and Holm p<.05; >=4/5 positive development folds; uncovered-context retrieval above chance; no retained adverse support below -0.02',
        files={n:sha(OUT/n) for n in ['items.json','reference.json','cohorts.json']},codeSha256=sha(__file__),productionApproved=False)
    write(OUT/'protocol.json',protocol);(OUT/'frozen-prepare.py').write_bytes(Path(__file__).read_bytes());print(json.dumps({k:protocol[k] for k in ['referencePairs','developmentCases','evaluationCases','eligibleReservedPairs','finalIndependence']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
