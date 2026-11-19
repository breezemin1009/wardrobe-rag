"""Reference-only full-vector pair learning, before a reserved new-ID test.

All earlier failed studies are immutable. This reference excludes every target
outfit of the 31 reserved cases. No reserved predictions are made in this file.
"""
from pathlib import Path
from datetime import datetime,timezone
from collections import defaultdict,Counter
import argparse,json,random,time
import numpy as np
import torch
from torch import nn
from backend.thesis_study_20261023 import ROOT,read,write,sha,h
OUT=ROOT/'joint-reserved-20261025';SEED=2026102523

class PairNet(nn.Module):
    def __init__(self,kind):
        super().__init__();self.kind=kind
        if kind=='bilinear':
            self.top=nn.Linear(384,64,bias=False);self.bottom=nn.Linear(384,64,bias=False);self.bias=nn.Linear(768,1)
        else:self.net=nn.Sequential(nn.Linear(1536,256),nn.ReLU(),nn.Dropout(.15),nn.Linear(256,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,q,b):
        if self.kind=='bilinear':return (self.top(q)*self.bottom(b)).sum(-1)/8+self.bias(torch.cat([q,b],dim=-1)).squeeze(-1)
        return self.net(torch.cat([q,b,torch.abs(q-b),q*b],dim=-1)).squeeze(-1)

def prepare():
    if (OUT/'neural-protocol.json').exists():raise RuntimeError('Neural development already reserved')
    assert not (OUT/'test-pair-features.npz').exists()
    write(OUT/'neural-protocol.json',dict(createdAt=datetime.now(timezone.utc).isoformat(),referenceSha256=sha(OUT/'reference.json'),
        purpose='Improve cold-item pair support using full384-dimensional title vectors ratherthan32-dimensionalPCA alone. Reference-only relation training. No source IDs/degrees/outfits enter inference.',
        models='Two architectures:rank64bilinear withlinear bias; MLP1536-256-64-1 withReLU/dropout.15. AdamW lr.001,weightdecay.0001/.01,12epochs,batch1024. Saveepochs2/4/8/12,16fixedcheckpoints. GPUfixedseed2523.',
        negatives='Per reference positive:two hash-random same sourcecategory/kind reference bottoms, andtwo semihard among64most-title-similar reference bottoms, rejecting all knownreference-positive pairs. Missingpair is unobserved,notaestheticnegative. Query-degree inverse parent weights.',
        calibration='Fixed neural checkpoint logits plus nine reference retrieval features, standardized logisticC=.1 usingtraining450querylabels.3parentfoldcrosspredictionsforconditionalheadtraining. Development225only selects neuralcheckpoint/supportmix/alpha. Reserved31freshIDs neverpredicteduntilfinalseal.',
        history='Previous broad258,strict36 andadditional77historytask have been opened and remainfailedadoption reports. Newlyreserved31IDs are unscored; internal inspectedPOGsource,notuntouchedexternaldata.',
        priorReservationSha256=sha(OUT/'protocol.json'),codeSha256=sha(__file__)))
    from backend.pog_refined_retrieval import Retriever
    e=Retriever(OUT);g=defaultdict(set);pools=defaultdict(set);degree=Counter()
    for r in e.records:
        g[r['top']].add(r['bottom']);b=e.items[r['bottom']];pools[(b['sourceCategory'],b['kind'])].add(r['bottom']);degree[r['top']]+=1
    qs=[];bs=[];ys=[];ws=[]
    for i,r in enumerate(e.records):
        q,b=r['top'],r['bottom'];cat=(e.items[b]['sourceCategory'],e.items[b]['kind']);pool=sorted(pools[cat]);neg=[x for x in pool if x not in g[q]]
        if len(neg)<4:continue
        rng=np.random.default_rng(int(h(f'{SEED}:{q}:{b}')[:12],16));randoms=rng.choice(neg,2,replace=False).tolist();cix=np.asarray([e.ix[x] for x in neg]);sim=e.vectors[cix]@e.vectors[e.ix[b]];hard=[neg[j] for j in np.argsort(-sim,kind='stable')[:64] if neg[j] not in randoms];selected=randoms+rng.choice(hard,2,replace=False).tolist()
        for bb,y in [(b,1)]+[(x,0) for x in selected]:qs.append(e.ix[q]);bs.append(e.ix[bb]);ys.append(y);ws.append(1/degree[q])
        if i%10000==0:print('neural pairs',i,'/',len(e.records),flush=True)
    weights=np.asarray(ws,dtype=np.float32);weights*=len(weights)/weights.sum();np.savez_compressed(OUT/'neural-reference-pairs.npz',q=np.asarray(qs,dtype=np.int32),b=np.asarray(bs,dtype=np.int32),y=np.asarray(ys,dtype=np.float32),w=weights)
    write(OUT/'neural-training-lock.json',dict(pairs=len(ys),positives=int(sum(ys)),referencePairs=len(e.records),artifactSha256=sha(OUT/'neural-reference-pairs.npz'),vectorsSha256=sha(OUT/'vectors.npy')));print('REFERENCE TRAINING',len(ys),flush=True)

def train():
    if (OUT/'neural-models.json').exists():raise RuntimeError('Neural models already fitted')
    torch.set_num_threads(2);torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED);device='cuda' if torch.cuda.is_available() else 'cpu';torch.backends.cuda.matmul.allow_tf32=False
    V=torch.as_tensor(np.load(OUT/'vectors.npy'),device=device);a=np.load(OUT/'neural-reference-pairs.npz');qi=torch.as_tensor(a['q'].astype(np.int64),device=device);bi=torch.as_tensor(a['b'].astype(np.int64),device=device);Y=torch.as_tensor(a['y'],device=device);W=torch.as_tensor(a['w'],device=device);checkpoints={};start=time.monotonic()
    for kind in ['bilinear','mlp']:
        for decay in [.0001,.01]:
            torch.manual_seed(SEED);m=PairNet(kind).to(device);optimizer=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=decay);rng=np.random.default_rng(SEED)
            for epoch in range(1,13):
                m.train();order=torch.as_tensor(rng.permutation(len(Y)),device=device);total=0.
                for ix in order.split(1024):
                    optimizer.zero_grad(set_to_none=True);z=m(V[qi[ix]],V[bi[ix]]);loss=(nn.functional.binary_cross_entropy_with_logits(z,Y[ix],reduction='none')*W[ix]).mean();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),5);optimizer.step();total+=float(loss.detach())*len(ix)
                print('neural',kind,decay,'epoch',epoch,'loss',round(total/len(Y),5),'elapsed',round(time.monotonic()-start,1),flush=True)
                if epoch in [2,4,8,12]:
                    name=f'neural-{kind}-reg{decay}-epoch{epoch}.pt';torch.save(dict(kind=kind,state={k:v.cpu() for k,v in m.state_dict().items()}),OUT/name);checkpoints[name]=sha(OUT/name)
    write(OUT/'neural-models.json',dict(models=checkpoints,protocolSha256=sha(OUT/'neural-protocol.json'),trainingLockSha256=sha(OUT/'neural-training-lock.json'),device=device,codeSha256=sha(__file__)))

def support():
    from backend import thesis_conditional_models_20261025 as old
    old.OUT=OUT
    old.features('training');old.features('development');old.fit_support()

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train','support']);a=p.parse_args();globals()[a.stage]()
