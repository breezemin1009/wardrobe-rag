"""Frozen source-rich features; candidate-batch optimization audited pointwise."""
from pathlib import Path
import argparse,time
import numpy as np
from threadpoolctl import threadpool_limits
from backend import thesis_multilink_source_20261027 as s
from backend import thesis_repair_models_20261026 as b
from backend import thesis_repair_stable_kernel_20261026 as k
OUT=s.OUT

def extract(stage):
    assert not (OUT/(stage+'-features.npz')).exists()
    if stage=='test':assert (OUT/'selection-lock.json').exists() and (OUT/'evaluation-lock.json').exists()
    from backend.pog_refined_retrieval import Retriever
    threadpool_limits(1);e=Retriever(OUT);frames=b.read(OUT/(stage+'-wardrobes.json'));parts=OUT/(stage+'-feature-parts');parts.mkdir(exist_ok=True)
    lock=dict(codeSha256=b.sha(__file__),kernelSha256=b.sha(k.__file__),protocolSha256=b.sha(OUT/'protocol.json'),framesSha256=b.sha(OUT/(stage+'-wardrobes.json')))
    if (parts/'binding.json').exists():assert b.read(parts/'binding.json')==lock
    else:b.write(parts/'binding.json',lock)
    xx=[];ff=[];rr=[];start=time.monotonic()
    for i,c in enumerate(frames):
        cp=parts/f'{i:04d}.npz'
        if cp.exists():z=np.load(cp);x,f,r=[z[n] for n in ['X','F','R']]
        else:
            cs=c['bottoms'][:2];qs=c['contexts'];x=np.concatenate([k.base_features(e,[q],cs) for q in qs],axis=1)
            if len({(e.items[p]['sourceCategory'],e.items[p]['kind']) for p in cs})==1:f,r,_=k.support_features(e,qs,cs,exclude_self=False,exclude_near=True)
            else:
                values=[k.support_features(e,qs,[candidate],exclude_self=False,exclude_near=True) for candidate in cs];f=np.concatenate([a[0] for a in values]);r=np.concatenate([a[1] for a in values])
            assert x.shape==(2,8,514) and f.shape==r.shape==(2,8,12) and all(np.isfinite(a).all() for a in [x,f,r])
            with cp.with_suffix('.tmp').open('wb') as handle:np.savez_compressed(handle,X=x,F=f,R=r)
            cp.with_suffix('.tmp').replace(cp)
        xx.append(x);ff.append(f);rr.append(r)
        if i%25==0:print('MULTILINK FEATURES',stage,i+1,'/',len(frames),'seconds',round(time.monotonic()-start,1),flush=True)
    np.savez_compressed(OUT/(stage+'-features.npz'),X=np.asarray(xx),F=np.asarray(ff),R=np.asarray(rr))
    b.write(OUT/(stage+'-feature-lock.json'),dict(createdAt=b.stamp(),**lock,featuresSha256=b.sha(OUT/(stage+'-features.npz'))))
    if stage=='development':
        errors=[]
        for kind in ['jeans','trousers','shorts']:
            i=next(i for i,c in enumerate(frames) if c['kind']==kind);c=frames[i]
            for j,candidate in enumerate(c['bottoms'][:2]):
                for h,q in enumerate(c['contexts']):
                    x=k.base_features(e,[q],[candidate]);f,r,_=k.support_features(e,[q],[candidate],exclude_self=False,exclude_near=True)
                    for a,z in [(xx[i][j,h],x[0,0]),(ff[i][j,h],f[0,0]),(rr[i][j,h],r[0,0])]:errors.append(float(np.max(abs(a-z))))
        assert max(errors)<=1e-7,errors
        b.write(OUT/'numerical-preflight.json',dict(createdAt=b.stamp(),allPassed=True,pairs=48,maxFeatureDifference=max(errors),codeSha256=b.sha(__file__),kernelSha256=b.sha(k.__file__),featuresSha256=b.sha(OUT/'development-features.npz')))
    print('MULTILINK FEATURES SEALED',stage,len(frames),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['development','test']);extract(ap.parse_args().stage)
