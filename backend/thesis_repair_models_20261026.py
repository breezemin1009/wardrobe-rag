"""Develop an observed-link support model using previously inspected data only.

Historical tests are explicitly development data here. New test predictions
are forbidden until a separate selection lock has been written. No old output
is overwritten and no aesthetic or purchase-probability interpretation is used.
"""
from pathlib import Path
from datetime import datetime,timezone
import argparse,json,hashlib,time
import numpy as np
from scipy.special import logsumexp,expit
from threadpoolctl import threadpool_limits

DATA=Path(__file__).parent/'data/menswear-expansion-v3'
OUT=DATA/'research-repair-20261026/model-development'
COHORTS=['asos/evidence-history-v2-20261026','asos/expanded-history-transfer-20261026','asos/history-title-neighbour-replication-20261026']
read=lambda p:json.loads(Path(p).read_text(encoding='utf8'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def stamp():return datetime.now(timezone.utc).isoformat()
def stable(s):return hashlib.sha256(str(s).encode()).hexdigest()

def reserve():
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'plan.json').exists():raise RuntimeError('Existing development plan')
    write(OUT/'plan.json',dict(createdAt=stamp(),purpose='All445historicalASOSparents are development, including prior tests. No new test read or evaluated.',
      sourceFiles={p:{n:sha(DATA/p/n) for n in ['frames.json','reference.json','itemIds.json','vectors.npy']} for p in COHORTS},
      features='Frozen384MiniLM->fixedGaussian128projection; q/c/product/absdiff plus dense and char cosine. RAG adds12original forward supports and12reverse supports (dense/hybrid x K1,8,32 x conditional/density). No IDs/degrees/target edges in features.',
      folds='Five deterministic parent folds, whole source neighbourhoods verified disjoint before fitting. Selection uses OOF parent-averageNG with one parent weight; tie break lower complexity then smalleralpha.',
      models=['LR_C.1','LR_C1','HGB_leaf7_reg1','HGB_leaf7_reg10','HGB_leaf15_reg1','HGB_leaf15_reg10'],
      sameBudgetNoRAG=True,alphas=[0,.1,.25,.5],directControls='dense,char,halfhybrid title cosine; same alpha grid and sameH; best OOF selected',
      finalDesign='One independent input per new parent; top counts3/5/8,owned0/1/3,candidates4. Two observed seeds plus two degree-matched cold controls, no enforced old-versus-new candidate conflict. W chosen from visible reference only; retain zero-gain inputs. All previous input master IDs/full outfit neighbourhoods excluded.',
      primaryComparisons=['development-selected-noRAG','development-selected-direct-title','frozen-old-RAG','same-support-C'],
      testStatistics='One new test opening after all methods fixed; parent paired bootstrap10000/signflip50000, Holm4. Report observed new-link fraction, NG (zero for all-zero gain),taskcoverage and prespecifiedkind/history strata. No threshold changes on final outcomes.',
      noExternalPreregistration=True,codeSha256=sha(__file__)))
    print('Development plan',sha(OUT/'plan.json'),flush=True)

def reverse_features(e,c):
    """Candidate-bottom neighbourhoods retrieve their actual paired tops."""
    qs=c['contexts'];bs=c['bottoms'][:4];values=[]
    for b in bs:
        target=e.items[b];bi=e.ix[b];row=[]
        for q in qs:
            source=e.items[q];qi=e.ix[q]
            pool=e.pools[(source['sourceCategory'],source['kind'],target['sourceCategory'],target['kind'])]
            ad=e.vectors[bi]@e.vectors[pool['bi']].T
            al=(e.lex[bi]@e.lex[pool['bi']].T).toarray().ravel()
            bd=e.vectors[qi]@e.vectors[pool['ti']].T
            bl=(e.lex[qi]@e.lex[pool['ti']].T).toarray().ravel()
            counts=np.bincount(pool['b'],minlength=len(ad));features=[]
            for hybrid in [False,True]:
                a=(ad+al)/2 if hybrid else ad;bval=(bd+bl)/2 if hybrid else bd
                order=np.argsort(-a,kind='stable')
                bg=logsumexp(bval[pool['t']]/.1-np.log(counts[pool['b']])-np.log(len(a)))
                for k in [1,8,32]:
                    mask=np.isin(pool['b'],order[:min(k,len(a))]);w=a[pool['b'][mask]]/.1-np.log(counts[pool['b'][mask]]);w-=logsumexp(w)
                    con=logsumexp(bval[pool['t'][mask]]/.1+w)
                    features.extend([np.clip(con*.1,0,1),expit(con-bg)])
            row.append(features)
        values.append(row)
    return np.asarray(values,dtype=np.float32)

def base_features(e,c):
    # A fixed seeded projection is an input transform, never fitted to labels.
    rng=np.random.default_rng(2026102621);projection=rng.normal(0,1/np.sqrt(128),(384,128)).astype(np.float32)
    qids=[e.ix[p] for p in c['contexts']];bids=[e.ix[p] for p in c['bottoms'][:4]]
    q=e.vectors[qids]@projection;b=e.vectors[bids]@projection
    qq=np.broadcast_to(q,(4,*q.shape));bb=np.broadcast_to(b[:,None,:],qq.shape)
    dense=e.vectors[bids]@e.vectors[qids].T;lex=(e.lex[bids]@e.lex[qids].T).toarray()
    return np.concatenate([qq,bb,qq*bb,np.abs(qq-bb),dense[...,None],lex[...,None]],axis=-1).astype(np.float32)

def extract():
    from backend.pog_refined_retrieval import Retriever
    plan=read(OUT/'plan.json');assert plan['codeSha256']==sha(__file__)
    if (OUT/'features.npz').exists():raise RuntimeError('Development features already exist')
    rows=[];xx=[];ff=[];rr=[];parents=[];ids=set();origins=set();start=time.monotonic()
    threadpool_limits(1)
    for path in COHORTS:
        p=DATA/path
        for n,v in plan['sourceFiles'][path].items():assert sha(p/n)==v
        e=Retriever(p);frames=read(p/'frames.json')
        arrays={s:np.load(p/f'{s}-features.npz')['S'] for s in ['development','test'] if (p/f'{s}-features.npz').exists()};counts={s:0 for s in arrays}
        for c in frames:
            roleids=set(c['contexts']+c['bottoms']);os={o.removeprefix('ASOS:') for o in c['sourceOutfits']}
            assert not ids&roleids and not origins&os,(path,c['id']);ids.update(roleids);origins.update(os)
            s=c['stage'];i=counts[s];counts[s]+=1
            f=arrays[s][i,:,:4,:].transpose(1,2,0);r=reverse_features(e,c);x=base_features(e,c)
            assert f.shape==(4,5,12) and r.shape==(4,5,12) and x.shape==(4,5,514)
            xx.append(x);ff.append(f);rr.append(r);rows.append(c);parents.append(path)
            if len(rows)%25==0:print('Development features',len(rows),'seconds',round(time.monotonic()-start,1),flush=True)
        del e
    np.savez_compressed(OUT/'features.npz',X=np.asarray(xx),F=np.asarray(ff),R=np.asarray(rr))
    write(OUT/'frames.json',rows);write(OUT/'feature-lock.json',dict(createdAt=stamp(),parents=len(rows),cohorts=parents,
      featuresSha256=sha(OUT/'features.npz'),framesSha256=sha(OUT/'frames.json'),planSha256=sha(OUT/'plan.json'),sourceNeighbourhoodDisjoint=True))
    print('Development features sealed',len(rows),flush=True)

def model(name):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    if name.startswith('LR'):return make_pipeline(StandardScaler(),LogisticRegression(C=float(name[4:]),max_iter=500,random_state=2026102621))
    _,leaf,reg=name.split('_')
    return HistGradientBoostingClassifier(max_iter=160,learning_rate=.05,max_leaf_nodes=int(leaf[4:]),l2_regularization=float(reg[3:]),min_samples_leaf=20,early_stopping=False,random_state=2026102621)

def ng(scores,gains):
    if np.max(gains)==0:return 0.
    return float(np.mean(np.asarray(gains)[np.abs(scores-np.max(scores))<=1e-12])/np.max(gains))

def evaluate_support(S,frames,alpha):
    result=[]
    for support,c in zip(S,frames):
        obs=np.asarray(c['observed']);hist=np.asarray(c['knownHistory']);vs=[]
        for name,v in c['views'].items():
            if name=='no-owned':continue
            qs=v['q'];ws=v['w'];base=support[:,qs]
            old=hist[np.ix_(ws,qs)].max(0) if ws else np.zeros(len(qs))
            trueold=obs[np.ix_(ws,qs)].max(0) if ws else np.zeros(len(qs))
            gains=(obs[:4,qs]*(1-trueold)).sum(1);C=base.mean(1);M=(base*(1-old)).mean(1)
            vs.append(ng(alpha*C+(1-alpha)*M,gains))
        result.append(float(np.mean(vs)))
    return np.asarray(result)

def develop():
    import joblib
    threadpool_limits(1)
    if (OUT/'selection-lock.json').exists():raise RuntimeError('Development already frozen')
    plan=read(OUT/'plan.json');lock=read(OUT/'feature-lock.json');assert plan['codeSha256']==sha(__file__)
    assert lock['featuresSha256']==sha(OUT/'features.npz') and lock['framesSha256']==sha(OUT/'frames.json')
    frames=read(OUT/'frames.json');z=np.load(OUT/'features.npz');n=len(frames)
    folds=np.asarray([int(stable('2026102621:fold:'+c['id'])[:8],16)%5 for c in frames]);y=np.asarray([c['observed'][:4] for c in frames]).reshape(-1)
    search=[];profiles={};start=time.monotonic();oof={}
    for path in ['noRAG','RAG']:
        X=z['X'] if path=='noRAG' else np.concatenate([z['X'],z['F'],z['R']],axis=-1)
        for name in plan['models']:
            preds=np.zeros((n,4,5));logloss=[]
            for fold in range(5):
                tr=np.flatnonzero(folds!=fold);dv=np.flatnonzero(folds==fold);m=model(name)
                tx=X[tr].reshape(-1,X.shape[-1]);ty=y.reshape(n,20)[tr].reshape(-1)
                if name.startswith('LR'):m.fit(tx,ty)
                else:m.fit(tx,ty)
                preds[dv]=m.predict_proba(X[dv].reshape(-1,X.shape[-1]))[:,1].reshape(-1,4,5)
            oof[path+'__'+name]=preds
            loss=float(np.mean(-(y*np.log(np.clip(preds.reshape(-1),1e-7,1-1e-7))+(1-y)*np.log(np.clip(1-preds.reshape(-1),1e-7,1-1e-7)))))
            for alpha in plan['alphas']:
                vals=evaluate_support(preds,frames,alpha);search.append(dict(path=path,model=name,alpha=alpha,OOF_NG=float(vals.mean()),OOF_logloss=loss))
            m=model(name);m.fit(X.reshape(-1,X.shape[-1]),y);fname=path+'__'+name+'.joblib';joblib.dump(m,OUT/fname)
            print('OOF',path,name,'bestNG',max(v['OOF_NG'] for v in search if v['path']==path),'seconds',round(time.monotonic()-start,1),flush=True)
        profiles[path]=max([v for v in search if v['path']==path],key=lambda v:(v['OOF_NG'],-v['OOF_logloss'],-v['alpha']))
    for feature,name in [(512,'dense'),(513,'char'),(None,'hybrid')]:
        S=(z['X'][...,feature]+1)/2 if feature is not None else (1+(z['X'][...,512]+z['X'][...,513])/2)/2
        for alpha in plan['alphas']:
            vals=evaluate_support(S,frames,alpha);search.append(dict(path='direct',model=name,alpha=alpha,OOF_NG=float(vals.mean())))
    profiles['direct']=max([v for v in search if v['path']=='direct'],key=lambda v:(v['OOF_NG'],-v['alpha']))
    np.savez_compressed(OUT/'oof-support.npz',**oof)
    for name,p in profiles.items():
        if name!='direct':p['modelFile']=p['path']+'__'+p['model']+'.joblib';p['modelSha256']=sha(OUT/p['modelFile'])
    result=dict(createdAt=stamp(),profiles=profiles,search=search,folds=folds.tolist(),parents=n,
       planSha256=sha(OUT/'plan.json'),featuresSha256=sha(OUT/'features.npz'),oofSha256=sha(OUT/'oof-support.npz'),codeSha256=sha(__file__),
       warning='OOF selected estimates are development, not unbiased final accuracy. Historical tests reused only as development. No new test predictions yet.')
    write(OUT/'selection-lock.json',result);print('LOCKED',json.dumps(profiles),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['reserve','extract','develop']);a=p.parse_args();globals()[a.stage]()
