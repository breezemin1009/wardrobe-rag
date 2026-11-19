"""Candidate-quarantined public-corpus supervision, before any fresh evaluation.

All 721 previously opened profiles are now development. The purpose is to test
whether more *training relations*, rather than further test re-selection, helps.
"""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,inspect,os,time
import numpy as np
from threadpoolctl import threadpool_limits
from backend import thesis_multilink_source_20261027 as previous
from backend import thesis_repair_models_20261026 as b
from backend import thesis_repair_stable_kernel_20261026 as kernel
OUT=previous.REPAIR/'corpus-support-20261027';SEED=2026102821
key=lambda value:b.stable(str(SEED)+':'+str(value))

def reserve():
    assert not OUT.exists();OUT.mkdir();old=previous.OUT
    for name in ['items.json','reference.json','candidate-pools.json','source-lock.json','vectors.npy','itemIds.json','encoder.json']:
        os.link(old/name,OUT/name)
    frames=b.read(old/'development-wardrobes.json')+b.read(old/'test-wardrobes.json');b.write(OUT/'development-wardrobes.json',frames)
    zz=[np.load(old/(stage+'-features.npz')) for stage in ['development','test']]
    np.savez_compressed(OUT/'development-features.npz',**{k:np.concatenate([z[k] for z in zz]) for k in ['X','F','R']})
    # Fixed random sample of real reference positives; hard unobserved controls
    # match source category and kinds and are selected without target labels.
    from backend.pog_refined_retrieval import Retriever
    threadpool_limits(1);e=Retriever(old);ref=b.read(old/'reference.json');pairs={(r['top'],r['bottom']) for r in ref};selected=sorted(ref,key=lambda r:key(r['top']+':'+r['bottom']))[:4096];rows=[];seen=set()
    for record in selected:
        q=record['top'];c=record['bottom'];source=e.items[q];target=e.items[c];cat=(source['sourceCategory'],source['kind'],target['sourceCategory'],target['kind']);pool=e.pools[cat]
        if (q,c) not in seen:rows.append(dict(q=q,c=c,y=1));seen.add((q,c))
        # A title-similar but unrecorded bottom is harder than an arbitrary
        # opposite-kind negative. The absence remains an observed-zero label.
        ci=e.ix[c];sim=e.vectors[ci]@e.vectors[pool['bi']].T;order=np.argsort(-sim,kind='stable');options=[pool['bottoms'][int(j)] for j in order if (q,pool['bottoms'][int(j)]) not in pairs and (q,pool['bottoms'][int(j)]) not in seen][:20]
        for negative in sorted(options,key=lambda p:key(q+':negative:'+p))[:1]:rows.append(dict(q=q,c=negative,y=0));seen.add((q,negative))
    b.write(OUT/'corpus-pairs.json',rows)
    protocol=dict(createdAt=b.stamp(),codeSha256=b.sha(__file__),seed=SEED,parents=len(frames),pairs=len(rows),positive=sum(r['y'] for r in rows),
      development='All earlier397+324profiles are exposed development now; no reuse as a new independent test.',
      source='4096fixedhash sampled positive edges from165795candidate-quarantined reference. One same-kind title-neighbour observed-zero control per positive; original source IDs and absent records are not aesthetic labels.',
      feature='The same immutable title514 and forward/reverse24 kernels; candidate self and near-title references excluded per pair,query self allowed. Compute each pair individually to match API.',
      models='Equal path budget: compact logistic C0.1/1/10,compact HGBleaf7/15,wideHGBleaf7; neuralNN0/32BCE without auxiliary as retained strong baselines. Auxiliary path uses corpus with total weight equal to wardrobe pairs. Also train each base family without corpus. Five parent folds; alpha0/.1/.25/.5/1; OOFNG selection. Same corpus and folds on both RAG/noRAG paths.',
      continuation='Auxiliary supervision must improve its path best non-auxiliary OOFNG by at least0.01 before a new final evaluation is opened. If not, retain failed development comparisons and do not open another test to seek significance.',
      primary='Best RAG versus best noRAG,direct title and identical supportC. Matched-architecture/corpus ablations retained. Any later source-constructed test and dependence plan must be frozen before predictions.',
      priorOutcomeSha256=b.sha(old/'test-results.json'),referenceSha256=b.sha(OUT/'reference.json'),corpusPairsSha256=b.sha(OUT/'corpus-pairs.json'),developmentSha256=b.sha(OUT/'development-wardrobes.json'),featureKernelSha256=b.sha(kernel.__file__))
    b.write(OUT/'protocol.json',protocol);print('CORPUS SUPERVISION RESERVED',len(frames),len(rows),flush=True)

def extract():
    from backend.pog_refined_retrieval import Retriever
    threadpool_limits(1);e=Retriever(OUT);p=b.read(OUT/'protocol.json');assert p['codeSha256']==b.sha(__file__);rows=b.read(OUT/'corpus-pairs.json');parts=OUT/'corpus-feature-parts';parts.mkdir(exist_ok=True);x=[];f=[];r=[];start=time.monotonic()
    for i,rec in enumerate(rows):
        cp=parts/f'{i:05d}.npz'
        if cp.exists():z=np.load(cp);xx,ff,rr=[z[n] for n in ['X','F','R']]
        else:
            xx=kernel.base_features(e,[rec['q']],[rec['c']])[0,0];ff,rr,_=kernel.support_features(e,[rec['q']],[rec['c']],exclude_self=False,exclude_near=True);ff=ff[0,0];rr=rr[0,0]
            np.savez_compressed(cp,X=xx,F=ff,R=rr)
        x.append(xx);f.append(ff);r.append(rr)
        if i%256==0:print('CORPUS SUPPORT FEATURES',i+1,'/',len(rows),'seconds',round(time.monotonic()-start,1),flush=True)
    np.savez_compressed(OUT/'corpus-features.npz',X=np.asarray(x),F=np.asarray(f),R=np.asarray(r))
    b.write(OUT/'corpus-feature-lock.json',dict(createdAt=b.stamp(),allFinite=all(np.isfinite(a).all() for a in [x,f,r]),featuresSha256=b.sha(OUT/'corpus-features.npz'),protocolSha256=b.sha(OUT/'protocol.json'),codeSha256=b.sha(__file__)))
    print('CORPUS FEATURES COMPLETE',len(rows),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['reserve','extract']);globals()[ap.parse_args().stage]()
