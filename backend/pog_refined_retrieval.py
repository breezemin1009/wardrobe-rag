"""Source-grounded evidence rank calibration and direct garment redundancy."""
from pathlib import Path
from collections import defaultdict
import argparse,time,hashlib
import numpy as np
from scipy.special import logsumexp,expit
from backend.pog_retrieval import Retriever as Base,verify
from backend.pog_refined_study import OUT
from backend.pog_cold_study import read,sha,stable,h
from backend.data_io import write_json_atomic as write

MODES=['char-conditional','char-density','char-percentile','dense-conditional','density','dense-percentile','hybrid-conditional','hybrid-density','hybrid-percentile']
CONFIGS=[dict(mode=m,k=k) for m in MODES for k in [8,32,96]]

class Retriever(Base):
    def __init__(self,out):
        super().__init__(out);self.backgrounds={};self.neighbours={}

    def pool_data(self,category,mode):
        key=(category,mode)
        if key in self.backgrounds:return self.backgrounds[key]
        pool=self.pools[category];background=sorted(pool['bottoms'],key=h)[:256];indices=np.array([self.ix[p] for p in background])
        if mode.startswith('char'):
            matrix=(self.lex[indices]@self.lex[pool['bi']].T).toarray()
        else:
            matrix=self.vectors[indices]@self.vectors[pool['bi']].T
            if mode.startswith('hybrid'):matrix=(matrix+(self.lex[indices]@self.lex[pool['bi']].T).toarray())/2
        self.backgrounds[key]=matrix;return matrix

    def matrix(self,contexts,candidates,config):
        results=[]
        for top in dict.fromkeys(contexts):
            cachekey=(top,tuple(candidates),stable(config))
            if cachekey in self.cache:results.append(self.cache[cachekey]);continue
            source=self.items[top];target=self.items[candidates[0]];category=(source['sourceCategory'],source['kind'],target['sourceCategory'],target['kind']);pool=self.pools.get(category)
            if pool is None:raise ValueError('No reference pool')
            qi=self.ix[top];ci=np.array([self.ix[p] for p in candidates]);mode=config['mode'];temp=.1
            if mode.startswith('char'):
                a=(self.lex[qi]@self.lex[pool['ti']].T).toarray().ravel();b=(self.lex[ci]@self.lex[pool['bi']].T).toarray()
            else:
                a=self.vectors[qi]@self.vectors[pool['ti']].T;b=self.vectors[ci]@self.vectors[pool['bi']].T
                if mode.startswith('hybrid'):a=(a+(self.lex[qi]@self.lex[pool['ti']].T).toarray().ravel())/2;b=(b+(self.lex[ci]@self.lex[pool['bi']].T).toarray())/2
            order=np.argsort(-a,kind='stable')[:min(config['k'],len(a))];mask=np.isin(pool['t'],order)
            w=a[pool['t'][mask]]/temp-np.log(pool['counts'][pool['t'][mask]]);w-=logsumexp(w)
            conditional=logsumexp(b[:,pool['b'][mask]]/temp+w,axis=1)
            if mode.endswith('percentile'):
                bg=self.pool_data(category,mode);scores=logsumexp(bg[:,pool['b'][mask]]/temp+w,axis=1)
                value=((conditional[:,None]>scores).sum(1)+.5*np.isclose(conditional[:,None],scores,atol=1e-10,rtol=0).sum(1))/len(scores)
            elif mode.endswith('density'):
                bw=-np.log(pool['counts'][pool['t']])-np.log(len(a));marginal=logsumexp(b[:,pool['b']]/temp+bw,axis=1);value=expit(conditional-marginal)
            else:value=np.clip(conditional*temp,0,1)
            self.cache[cachekey]=value;results.append(value)
        return np.asarray(results).T

    def features(self,case,config):
        f=super().features(case,config);c=case['candidates'];owned=list(dict.fromkeys(case['owned']));ci=[self.ix[p] for p in c];oi=[self.ix[p] for p in owned]
        dense=np.clip((self.vectors[ci]@self.vectors[oi].T).max(1),0,1) if owned else np.zeros(len(c))
        lex=np.clip((self.lex[ci]@self.lex[oi].T).toarray().max(1),0,1) if owned else np.zeros(len(c))
        f['char-novelty']=f['C']*(1-lex);f['dense-novelty']=f['C']*(1-dense)
        return f

def run(stage,out):
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    if stage=='evaluation' and not (out/'selection.json').exists():raise ValueError('Selection must be frozen first')
    cases=read(out/'cohorts.json')[stage];engine=Retriever(out);features={}
    for config in CONFIGS:
        start=time.monotonic();rows=[]
        for case in cases:rows.append({k:np.asarray(v).tolist() for k,v in engine.features(case,config).items()})
        features[stable(config)]=rows;print(stage,config,len(rows),round(time.monotonic()-start,1),flush=True)
    write(out/(stage+'-features.json'),features);write(out/(stage+'-features-lock.json'),dict(codeSha256=sha(__file__),baseCodeSha256=sha(Path(__file__).with_name('pog_retrieval.py')),featuresSha256=sha(out/(stage+'-features.json')),encoderSha256=sha(out/'encoder.json'),protocolSha256=sha(out/'protocol.json')))
    (out/'frozen-refined-retrieval.py').write_bytes(Path(__file__).read_bytes())

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['development','evaluation']);args=parser.parse_args();run(args.stage,OUT)
