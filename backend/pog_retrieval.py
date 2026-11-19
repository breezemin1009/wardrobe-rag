"""Frozen multilingual + character case retrieval; no compatibility training.

Only source product titles enter encoders. Co-outfit memberships enter the
reference case index and never the query embeddings or parameter fitting.
"""
from pathlib import Path
import argparse,hashlib,json,os,re,site,time
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
site.addsitedir(str(Path(__file__).resolve().parent/'.venv/Lib/site-packages'))
import numpy as np
from scipy.special import logsumexp,expit
from sklearn.feature_extraction.text import TfidfVectorizer
from backend.pog_cold_study import OUT,read,sha,stable
from backend.data_io import write_json_atomic as write

MODEL='sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'
REVISION='e8f8c211226b894fcb81acc59f3b34ba3efd5f42'
CONFIGS=[dict(mode=mode,k=k) for mode in ['dense-conditional','char-conditional','hybrid-conditional','density'] for k in [8,32,96]]

def text(x):
    # Remove years, seller SKU strings and sizes, retained source identifiers
    # never become encoder input. Chinese garment/color/pattern words remain.
    value=re.sub(r'[A-Za-z]*\d[A-Za-z\d_.-]*',' ',x['title'])
    value=re.sub(r'男士|男装|男款|男性|潮男|男生|男|官方|旗舰店|新款|新品|包邮|特价',' ',value)
    return re.sub(r'\s+',' ',value).strip()

def verify(out):
    p=read(out/'protocol.json')
    for name,expected in p['files'].items():
        if sha(out/name)!=expected:raise ValueError('Frozen source/cohort changed: '+name)
    return p

def encode(out,device):
    verify(out);items=read(out/'items.json');ids=sorted(items)
    plan=dict(model=MODEL,revision=REVISION,input='Original Chinese item title only; numeric SKU/year and generic male/retail terms removed',itemsSha256=sha(out/'items.json'),codeSha256=sha(__file__),training=False,device=device)
    if (out/'encoder.json').exists():
        saved=read(out/'encoder.json')
        if any(saved[k]!=v for k,v in plan.items()) or sha(out/'vectors.npy')!=saved['vectorsSha256']:raise ValueError('Encoder cache changed')
        return
    from sentence_transformers import SentenceTransformer
    import torch
    torch.set_num_threads(4);torch.manual_seed(20261021)
    if device=='cuda':
        torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True)
    model=SentenceTransformer(MODEL,revision=REVISION,device=device,local_files_only=True,trust_remote_code=False)
    vectors=model.encode([text(items[p]) for p in ids],normalize_embeddings=True,batch_size=128,show_progress_bar=True,convert_to_numpy=True)
    if not np.isfinite(vectors).all():raise ValueError('Nonfinite title vectors')
    np.save(out/'vectors.npy',vectors);write(out/'itemIds.json',ids)
    write(out/'encoder.json',dict(**plan,dimension=vectors.shape[1],vectorsSha256=sha(out/'vectors.npy'),idsSha256=sha(out/'itemIds.json')))

class Retriever:
    def __init__(self,out):
        verify(out);self.items=read(out/'items.json');self.ids=read(out/'itemIds.json');self.ix={p:i for i,p in enumerate(self.ids)}
        self.vectors=np.load(out/'vectors.npy');self.records=read(out/'reference.json')
        refids=sorted({r[s] for r in self.records for s in ['top','bottom']})
        self.vectorizer=TfidfVectorizer(analyzer='char',ngram_range=(2,4),min_df=2,max_features=100000,sublinear_tf=True,dtype=np.float32)
        self.vectorizer.fit([text(self.items[p]) for p in refids]);self.lex=self.vectorizer.transform([text(self.items[p]) for p in self.ids]).tocsr()
        self.pools={};self.cache={}
        from collections import defaultdict
        grouped=defaultdict(list)
        for r in self.records:
            t,b=self.items[r['top']],self.items[r['bottom']]
            grouped[(t['sourceCategory'],t['kind'],b['sourceCategory'],b['kind'])].append(r)
        for category,rows in grouped.items():
            tops=sorted({r['top'] for r in rows});bottoms=sorted({r['bottom'] for r in rows});ti={p:i for i,p in enumerate(tops)};bi={p:i for i,p in enumerate(bottoms)}
            t=np.array([ti[r['top']] for r in rows]);b=np.array([bi[r['bottom']] for r in rows]);counts=np.bincount(t,minlength=len(tops))
            self.pools[category]=dict(tops=tops,bottoms=bottoms,t=t,b=b,counts=counts,ti=np.array([self.ix[p] for p in tops]),bi=np.array([self.ix[p] for p in bottoms]),rows=rows)

    def matrix(self,contexts,candidates,config):
        contexts=list(dict.fromkeys(contexts));candidates=list(candidates);result=[]
        for top in contexts:
            cachekey=(top,tuple(candidates),stable(config))
            if cachekey in self.cache:result.append(self.cache[cachekey]);continue
            source=self.items[top];target=self.items[candidates[0]];category=(source['sourceCategory'],source['kind'],target['sourceCategory'],target['kind']);pool=self.pools.get(category)
            if pool is None:raise ValueError('No reference cases for source/target category')
            qi=self.ix[top];ci=np.array([self.ix[b] for b in candidates]);mode=config['mode'];temp=.1
            a=self.vectors[qi]@self.vectors[pool['ti']].T
            b=self.vectors[ci]@self.vectors[pool['bi']].T
            if mode in ['char-conditional','hybrid-conditional']:
                la=(self.lex[qi]@self.lex[pool['ti']].T).toarray().ravel()
                lb=(self.lex[ci]@self.lex[pool['bi']].T).toarray()
                if mode=='char-conditional':a,b=la,lb
                else:a,b=(a+la)/2,(b+lb)/2
            order=np.argsort(-a,kind='stable')[:min(config['k'],len(a))]
            ranks=np.empty(len(a),int);ranks.fill(len(a));ranks[order]=np.arange(len(order))
            mask=ranks[pool['t']]<len(order)
            logweights=a[pool['t'][mask]]/temp-np.log(pool['counts'][pool['t'][mask]])
            logweights-=logsumexp(logweights)
            conditional=logsumexp(b[:,pool['b'][mask]]/temp+logweights,axis=1)
            if mode=='density':
                backgroundWeights=-np.log(pool['counts'][pool['t']])-np.log(len(a))
                background=logsumexp(b[:,pool['b']]/temp+backgroundWeights,axis=1)
                value=expit(conditional-background)
            else:
                # Kernel mean transformed back to cosine scale. This is an
                # evidence similarity index, never an aesthetic probability.
                value=np.clip(conditional*temp,0,1)
            self.cache[cachekey]=value;result.append(value)
        return np.asarray(result).T

    def features(self,case,config):
        contexts=list(dict.fromkeys(case['contexts']));owned=list(dict.fromkeys(case['owned']));candidates=case['candidates'];matrix=self.matrix(contexts,candidates+owned,config)
        count=len(candidates);c=matrix[:count];prior=matrix[count:];baseline=prior.max(0) if len(prior) else np.zeros(len(contexts))
        m=np.maximum(c-baseline,0).mean(1);residual=(c*np.prod(1-prior,axis=0) if len(prior) else c).mean(1)
        for i,p in enumerate(candidates):
            if p in owned:m[i]=0;residual[i]=0
        return dict(C=c.mean(1),uncovered=c[:,0],**{'positive-difference':m,'product-residual':residual})

def build_features(out,stage):
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1)
    if stage=='evaluation' and not (out/'selection.json').exists():raise ValueError('Freeze development choices first')
    cohorts=read(out/'cohorts.json');cases=cohorts[stage];engine=Retriever(out);features={}
    for config in CONFIGS:
        name=stable(config);start=time.monotonic();rows=[]
        for n,case in enumerate(cases):
            rows.append({k:np.asarray(v).tolist() for k,v in engine.features(case,config).items()})
        features[name]=rows;print(stage,config,len(rows),round(time.monotonic()-start,2),flush=True)
    write(out/(stage+'-features.json'),features)
    write(out/(stage+'-features-lock.json'),dict(codeSha256=sha(__file__),featuresSha256=sha(out/(stage+'-features.json')),encoderSha256=sha(out/'encoder.json'),protocolSha256=sha(out/'protocol.json')))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['encode','development','evaluation']);parser.add_argument('--output',type=Path,default=OUT);parser.add_argument('--device',default='cuda');args=parser.parse_args()
    if args.stage=='encode':encode(args.output,args.device)
    else:build_features(args.output,args.stage)
