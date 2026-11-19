"""Source availability for a separate coherent-wardrobe/multiple-link study.

No scoring models or prediction outcomes are used by this inventory. The
four earlier studies and their failed results remain frozen and reportable.
"""
from collections import defaultdict,Counter
from pathlib import Path
from backend import thesis_repair_models_20261026 as b
ROOT=b.DATA/'asos';REPAIR=b.DATA/'research-repair-20261026';OUT=REPAIR/'multilink-study-20261027'
SEED=2026102801
key=lambda s:b.stable(str(SEED)+':'+str(s))

def main():
    assert not OUT.exists();OUT.mkdir()
    items=b.read(ROOT/'expanded-projected-20261026/items.json');edges=b.read(ROOT/'expanded-projected-20261026/edges.json')
    rev=defaultdict(set)
    for r in edges:rev[r['bottom']].add(r['top'])
    used=set(b.read(REPAIR/'known-query-corrected-20261027/prior-ids.json'));files={}
    for file in sorted(set(REPAIR.rglob('*frames.json'))|set(REPAIR.rglob('*wardrobes.json'))):
        if OUT in file.parents:continue
        rows=b.read(file)
        if not isinstance(rows,list):continue
        changed=False
        for c in rows:
            if isinstance(c,dict) and 'contexts' in c and 'bottoms' in c:used.update(c['contexts']+c['bottoms']);changed=True
        if changed:files[str(file.relative_to(REPAIR))]=b.sha(file)
    available={p for p,v in rev.items() if len(v)>=2 and p not in used and items[p]['kind'] in ['jeans','trousers','shorts']}
    degree=dict(Counter(len(rev[p]) for p in available));kind=dict(Counter(items[p]['kind'] for p in available))
    report=dict(createdAt=b.stamp(),codeSha256=b.sha(__file__),usedSupervisedRoleIDs=len(used),
      eligibleTargetIDs=len(available),degreeCounts=degree,kinds=kind,priorFiles=files,
      information='Eligibility requires at least two distinct observed tops and no previous input role. Some candidate outfits were previously visible as unsupervised retrieval records; all new candidate full source outfits must be removed from a newly built reference, and all models retrained from scratch. Never call this wholly unexposed source data.',
      purpose='Data availability only, not model evaluation. Future study must specify coherent wardrobe sampling without forcing gain, C/M conflict, or excluding zero-opportunity cases.')
    b.write(OUT/'inventory.json',report);b.write(OUT/'eligible-candidate-ids.json',sorted(available));print(report,flush=True)

if __name__=='__main__':main()
