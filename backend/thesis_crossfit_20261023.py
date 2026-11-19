"""Locked three-fold garment-disjoint reanalysis with nontrivial redundancy.

The model/parameters are already published. This is a robustness reanalysis,
not another untouched confirmation set. All results are retained.
"""
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import argparse,json,random
from backend.thesis_study_20261023 import source,ROOT,PARENT,read,write,sha,h,SEED

OUT=ROOT/'thesis-crossfit-20261023'

def prepare():
    if OUT.exists():raise RuntimeError('Fresh immutable directory required')
    it,edges,g,rev,reference,exposed=source()
    # Exclude every old development garment from test. The selected methods
    # and strongest alternatives are fixed before the new split is evaluated.
    dev=read(PARENT/'cohorts.json')['development']
    training_only={p for c in dev for p in c['contexts']+c['owned']+c['candidates']}
    assigned={p:int(h(f'{SEED}:outer:{p}')[:12],16)%3 for p in it}
    OUT.mkdir();counts=[];allused=set()
    variants={'primary':dict(q=8,w=2,n=8),'contexts4':dict(q=4,w=2,n=8),
        'owned1':dict(q=8,w=1,n=8),'owned4':dict(q=8,w=4,n=8),
        'candidates4':dict(q=8,w=2,n=4),'candidates16':dict(q=8,w=2,n=16),
        'noOwned':dict(q=8,w=0,n=8)}
    for fold in range(3):
        dest=OUT/f'fold-{fold}';dest.mkdir()
        reserved={p for p in it if assigned[p]==fold and p not in training_only}
        refs=[dict(top=t,bottom=b) for t,b in sorted(edges) if assigned[t]!=fold and assigned[b]!=fold]
        refpools={(it[r['top']]['sourceCategory'],it[r['top']]['kind'],it[r['bottom']]['sourceCategory'],it[r['bottom']]['kind']) for r in refs}
        pools=defaultdict(list)
        for b in sorted(reserved):
            if it[b]['side']=='bottom':pools[(it[b]['sourceCategory'],it[b]['kind'])].append(b)
        used=set();blocks=[];skips=Counter()
        for a in sorted(reserved,key=lambda p:h(f'{SEED}:anchor:{p}')):
            if a in used or it[a]['side']!='top':continue
            grouped=defaultdict(list)
            for b in g[a]&reserved:
                if b not in used:grouped[(it[b]['sourceCategory'],it[b]['kind'])].append(b)
            valid=[cat for cat,bs in grouped.items() if len(bs)>=2 and (it[a]['sourceCategory'],it[a]['kind'],*cat) in refpools]
            if not valid:continue
            cat=min(valid,key=h);bs=sorted(grouped[cat],key=lambda p:h(f'{a}:{p}'));owned=[bs[0]];anchor_candidates=[bs[1]]
            contexts=[a];local={a,*owned,*anchor_candidates}
            pool=sorted(pools[cat],key=lambda p:h(f'{SEED}:{a}:{p}'))
            # Three more candidate-context pairs. These are not constrained to
            # match candidate total degree or to yield a unique best candidate.
            for b in pool:
                if b in used|local:continue
                tops=[t for t in rev[b]&reserved if t not in used|local and (it[t]['sourceCategory'],it[t]['kind'],*cat) in refpools]
                if not tops:continue
                t=min(tops,key=lambda p:h(f'{a}:{b}:{p}'))
                contexts.append(t);anchor_candidates.append(b);local.update([b,t])
                if len(contexts)==4:break
            if len(contexts)<4:skips['additional-anchors']+=1;continue
            # Add four further tops from the same fold, without selecting them
            # for any candidate relevance. More-than-one observed links remain
            # possible and are measured rather than constructed away.
            extra_tops=[t for t in reserved-used-local if it[t]['side']=='top' and (it[t]['sourceCategory'],it[t]['kind'],*cat) in refpools]
            extra_tops=sorted(extra_tops,key=lambda p:h(f'{SEED}:extra:{a}:{p}'))[:4]
            if len(extra_tops)<4:skips['extra-tops']+=1;continue
            contexts+=extra_tops;local.update(extra_tops)
            # Extra owned alternatives are ordinary same-category held-out
            # garments, chosen without checking their relation to contexts.
            extras=[b for b in pool if b not in used|local][:3]
            if len(extras)<3:skips['owned']+=1;continue
            owned+=extras;local.update(extras)
            controls=[b for b in pool if b not in used|local][:12]
            if len(controls)<12:skips['controls']+=1;continue
            candidates=anchor_candidates+controls;local.update(controls)
            # Preserve the covered/uncovered contrast in Q=4 and all candidate
            # sizes, but randomize order so no method receives its position.
            random.Random(SEED+fold*10000+len(blocks)).shuffle(contexts[:4])
            for k in ['context-first','candidate-first']:
                rng=random.Random(int(h(f'{k}:{a}')[:10],16))
                if k=='context-first':head=contexts[:4];rng.shuffle(head);contexts=head+contexts[4:]
                else:head=candidates[:4];rng.shuffle(head);candidates=head+candidates[4:]
            observed=[[int(b in g[t]) for t in contexts] for b in candidates]
            history=[[int(b in g[t]) for t in contexts] for b in owned]
            blocks.append(dict(id=f'fold-{fold}-{len(blocks):04}',contexts=contexts,owned=owned,candidates=candidates,
                observed=observed,history=history,category=list(cat)))
            assert not allused&local
            used.update(local);allused.update(local)
        # Cache reuse is exact; six inactive entries remain only in the encoder
        # files to preserve upstream indexing, never in cohorts or reference.
        for n in ['items.json','vectors.npy','itemIds.json','encoder.json']:(dest/n).write_bytes((PARENT/n).read_bytes())
        write(dest/'reference.json',refs);write(dest/'blocks.json',blocks)
        parent=read(ROOT/'thesis-quality-20261023/protocol.json')
        protocol={**parent,'createdAt':datetime.now(timezone.utc).isoformat(),'variants':variants,
            'blocks':len(blocks),'blocksSha256':sha(dest/'blocks.json'),
            'files':{n:sha(dest/n) for n in ['items.json','reference.json']},
            'fold':fold,'referencePairs':len(refs),
            'scope':'Threefold robustness reanalysis of the already studied source; no untouched-confirmatory claim. No new choice of hyperparameters.',
            'design':'One covered context with two same-category bottoms plus three candidate-context anchor pairs and four unrelated heldout contexts. Four owned and sixteen candidate garments; nested sizes in one block.',
            'inference':'6000 block bootstraps, 20000 sign flips; Holm over nine primary contrasts. Report fold-specific effects; conditional on fixed reference/encoder. Folds share source through reference, so not independent trials.',
            'audit':dict(referenceTestOverlap=len(used & {r[s] for r in refs for s in ['top','bottom']}),
                oldDevelopmentTestOverlap=len(used&training_only),mutuallyDisjointBlocks=True,skipped=dict(skips))}
        write(dest/'protocol.json',protocol)
        counts.append(dict(fold=fold,blocks=len(blocks),referencePairs=len(refs),garments=len(used),audit=protocol['audit']))
    write(OUT/'protocol.json',dict(createdAt=datetime.now(timezone.utc).isoformat(),seed=SEED,folds=counts,
        garments=len(allused),oldDevelopmentExcluded=len(training_only),codeSha256=sha(__file__),
        scope='Frozen-method crossfitted robustness reanalysis; three folds are not three independent replications',
        methodLockSha256=sha(ROOT/'thesis-quality-20261023/protocol.json')))
    (OUT/'frozen-prepare.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(counts,ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare']);args=parser.parse_args();globals()[args.stage]()
