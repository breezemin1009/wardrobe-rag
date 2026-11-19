"""OOF query support calibration and augmented conditional overlap learning."""
from pathlib import Path
from datetime import datetime,timezone
from itertools import combinations
from collections import defaultdict,Counter
import argparse,json
import numpy as np
from backend.thesis_neural_pair_20261025 import OUT,SEED,PairNet
from backend.thesis_study_20261023 import read,write,sha,h
from backend import thesis_adaptive_policy_20261024 as metric

def logits(stage):
    import torch
    dest=OUT/f'neural-{stage}-logits.npz'
    if dest.exists():return np.load(dest)['Z']
    torch.set_num_threads(2);device='cuda' if torch.cuda.is_available() else 'cpu';V=torch.as_tensor(np.load(OUT/'vectors.npy'),device=device);ix={p:i for i,p in enumerate(read(OUT/'itemIds.json'))};rows=read(OUT/f'{stage}-frames.json');models=read(OUT/'neural-models.json')['models'];allz=[]
    q=[];b=[];offsets=[];pos=0
    for c in rows:
        for bb in c['bottoms']:
            for qq in c['contexts']:q.append(ix[qq]);b.append(ix[bb])
        n=len(c['bottoms'])*len(c['contexts']);offsets.append((pos,pos+n));pos+=n
    q=torch.as_tensor(q,device=device);b=torch.as_tensor(b,device=device)
    for name,value in models.items():
        assert sha(OUT/name)==value;ckpt=torch.load(OUT/name,map_location='cpu',weights_only=True);m=PairNet(ckpt['kind']).to(device);m.load_state_dict(ckpt['state']);m.eval();zz=[]
        with torch.inference_mode():
            for j in range(0,len(q),4096):zz.append(m(V[q[j:j+4096]],V[b[j:j+4096]]).cpu().numpy())
        z=np.concatenate(zz);a=np.zeros((len(rows),9,12),dtype=np.float32)
        for i,(start,end) in enumerate(offsets):a[i,:len(rows[i]['bottoms']),:len(rows[i]['contexts'])]=z[start:end].reshape(len(rows[i]['bottoms']),len(rows[i]['contexts']))
        allz.append(a)
    Z=np.stack(allz,axis=-1);np.savez_compressed(dest,Z=Z);write(OUT/f'neural-{stage}-logit-lock.json',dict(models=list(models),modelLockSha256=sha(OUT/'neural-models.json'),logitsSha256=sha(dest)))
    return Z

def calibrate():
    import joblib
    from threadpoolctl import threadpool_limits
    from backend.thesis_conditional_models_20261025 import pair_model,valid_pairs
    threadpool_limits(1)
    if (OUT/'neural-calibration-lock.json').exists():raise RuntimeError('Query support calibration sealed')
    assert not (OUT/'test-pair-features.npz').exists()
    plan=dict(createdAt=datetime.now(timezone.utc).isoformat(),sourceProtocolSha256=sha(OUT/'neural-protocol.json'),checkpoints=16,
        calibration='StandardScaler+LogisticRegressionC.1;139reference/titlefeatures+NNlogit; no-runtime-retrieval dropsfirst9,131features. Fittraining450allrealpairs,three-parent-fold OOF. Select checkpoint bymacrodevelopment225logloss, NOTbyopenedtest outcomes.',
        conditional='Aftercheckpoint selection:augment allcandidates with everyother singleton plus4hashpairs+4hashtriples. Allcandidate-positive contexts included, even stateswithnonewlink. g HGB160,leaf7/15,L2=1/10,min50,lr.04;last33(RAG)/11(noRuntimeRetrieval) aggregatesmonotone-decreasing. OOFsupportfeatures onlyforquerytraining.',
        jointSelection='Fiveold/new supportmixtures t=0,.25,.5,.75,1;alphas0,.1,.3,.5,2/3,1. MaxdevelopmentparentNG subjecttorich/ownedrich nonnegativevsidenticalC andalpha<1,4heads,120policiesperpath. StrongCcontrolsincludeall16checkpointCresults ineachpathplus35oldmixtures and5selectedNNmixtures. No finalpredictions untilseparatefreeze.',codeSha256=sha(__file__))
    write(OUT/'neural-query-plan.json',plan)
    tr=read(OUT/'training-frames.json');dv=read(OUT/'development-frames.json');TX=np.load(OUT/'training-pair-features.npz')['X'];DX=np.load(OUT/'development-pair-features.npz')['X'];TZ=logits('training');DZ=logits('development');fold=np.asarray([int(h(f'{SEED}:calibration:{c["id"]}')[:12],16)%3 for c in tr]);tp=np.zeros((*TZ.shape,2),dtype=np.float32);dp=np.zeros((*DZ.shape,2),dtype=np.float32);selected={};search=[]
    filenames=list(read(OUT/'neural-models.json')['models'])
    for j,neural in enumerate(filenames):
        for path,start,channel in [('RAG',0,0),('no-runtime-retrieval',9,1)]:
            tx=np.concatenate([TX[...,start:],TZ[...,j,None]],axis=-1);dx=np.concatenate([DX[...,start:],DZ[...,j,None]],axis=-1)
            for f in range(3):
                x,y,w=valid_pairs(tx,tr,fold!=f);m=pair_model('linear');m.fit(x,y,logisticregression__sample_weight=w);v=tx[fold==f];tp[fold==f,...,j,channel]=m.predict_proba(v.reshape(-1,v.shape[-1]))[:,1].reshape(v.shape[:-1])
            x,y,w=valid_pairs(tx,tr,np.ones(len(tr),bool));m=pair_model('linear');m.fit(x,y,logisticregression__sample_weight=w);prob=m.predict_proba(dx.reshape(-1,dx.shape[-1]))[:,1].reshape(dx.shape[:-1]);dp[...,j,channel]=prob
            loss=[]
            for i,c in enumerate(dv):
                p=np.clip(prob[i,:len(c['bottoms']),:len(c['contexts'])],1e-7,1-1e-7);yy=np.asarray(c['observed']);loss.append(float(-(yy*np.log(p)+(1-yy)*np.log(1-p)).mean()))
            name=f'calibration-{channel}-{j}.joblib';joblib.dump(m,OUT/name);row=dict(path=path,checkpoint=j,neural=neural,neuralSha256=sha(OUT/neural),calibrator=name,calibratorSha256=sha(OUT/name),developmentLogLoss=float(np.mean(loss)));search.append(row)
            if path not in selected or row['developmentLogLoss']<selected[path]['developmentLogLoss']:selected[path]=row
        print('CALIBRATED',j+1,'/',len(filenames),'best',json.dumps({k:v['developmentLogLoss'] for k,v in selected.items()}),flush=True)
    np.savez_compressed(OUT/'neural-training-calibrated.npz',P=tp);np.savez_compressed(OUT/'neural-development-calibrated.npz',P=dp)
    write(OUT/'neural-calibration-lock.json',dict(selected=selected,search=search,trainingOOF=True,folds=fold.tolist(),dataHashes={n:sha(OUT/n) for n in ['neural-training-calibrated.npz','neural-development-calibrated.npz']},planSha256=sha(OUT/'neural-query-plan.json')))
    print('NEURAL CALIBRATION',json.dumps(selected),flush=True)

def inputs(X,Z,P,cindices,windices,qindices,lex,dense,no_retrieval=False):
    ci=list(cindices);wi=list(windices);qs=list(qindices);n=len(ci);q=len(qs);cx=np.concatenate([X[ci][:,qs,9:] if no_retrieval else X[ci][:,qs],Z[ci][:,qs,None]],axis=-1);cp=P[ci][:,qs];op=P[wi][:,qs] if wi else np.zeros((0,q,P.shape[-1]));parts=[cx,cp];owned=[op.max(0) if wi else np.zeros((q,P.shape[-1])),op.sum(0)]
    if not no_retrieval:
        ok=X[wi][:,qs,:9] if wi else np.zeros((0,q,9));owned.extend([ok.max(0) if wi else np.zeros((q,9)),ok.sum(0)])
    o=np.concatenate(owned,axis=-1);parts.append(np.broadcast_to(o,(n,*o.shape)));sl=lex[np.ix_(ci,wi)];sd=dense[np.ix_(ci,wi)];sim=np.stack([sl.max(1) if wi else np.zeros(n),sd.max(1) if wi else np.zeros(n),sl.sum(1),sd.sum(1),np.full(n,len(wi))],axis=-1);parts.append(np.broadcast_to(sim[:,None,:],(n,q,5)))
    return np.concatenate(parts,axis=-1).astype(np.float32)

def stage_arrays(stage,path):
    lock=read(OUT/'neural-calibration-lock.json')['selected'][path];channel=0 if path=='RAG' else 1;X=np.load(OUT/f'{stage}-pair-features.npz')['X'];Z=np.load(OUT/f'neural-{stage}-logits.npz')['Z'][...,lock['checkpoint']];old=np.load(OUT/f'{stage}-probabilities.npz')['P'];new=np.load(OUT/f'neural-{stage}-calibrated.npz')['P'][...,lock['checkpoint'],channel]
    P=np.concatenate([old if channel==0 else old[...,2:],new[...,None]],axis=-1);base=read(OUT/'support-lock.json')['weights' if channel==0 else 'noRAGWeights'];base=np.asarray(base if channel==0 else base[2:]);return X,Z,P,base

def neural_views(stage,path):
    from backend.pog_refined_retrieval import Retriever
    X,Z,P,base=stage_arrays(stage,path);rows=read(OUT/f'{stage}-frames.json');e=Retriever(OUT);es=[]
    for i,c in enumerate(rows):
        ids=[e.ix[p] for p in c['bottoms']];lex=np.clip((e.lex[ids]@e.lex[ids].T).toarray(),0,1);dense=np.clip(e.vectors[ids]@e.vectors[ids].T,0,1);obs=np.asarray(c['observed'])
        for name,v in c['views'].items():
            ci,wi,qs=v['c'],v['w'],v['q'];prior=obs[np.ix_(wi,qs)].max(0) if wi else np.zeros(len(qs));gain=(obs[np.ix_(ci,qs)]*(1-prior)).sum(1)
            if not gain.max():continue
            es.append(dict(parent=i,id=c['id'],view=name,c=ci,w=wi,q=qs,gain=gain,X=inputs(X[i],Z[i],P[i],ci,wi,qs,lex,dense,path!='RAG'),components=P[i,ci][:,qs],ownedComponents=P[i,wi][:,qs] if wi else np.zeros((0,len(qs),P.shape[-1]))))
    return es,base

def train_overlap():
    from backend.pog_refined_retrieval import Retriever
    from sklearn.ensemble import HistGradientBoostingClassifier
    from threadpoolctl import threadpool_limits
    import joblib
    threadpool_limits(1)
    if (OUT/'neural-method-selection.json').exists():raise RuntimeError('Neural conditional selection sealed')
    assert not (OUT/'test-pair-features.npz').exists();engine=Retriever(OUT);rows=read(OUT/'training-frames.json');profiles={};audit=[]
    for path in ['RAG','no-runtime-retrieval']:
        X,Z,P,oldweights=stage_arrays('training',path);xx=[];yy=[];ww=[]
        for i,c in enumerate(rows):
            ids=[engine.ix[p] for p in c['bottoms']];lex=np.clip((engine.lex[ids]@engine.lex[ids].T).toarray(),0,1);dense=np.clip(engine.vectors[ids]@engine.vectors[ids].T,0,1);obs=np.asarray(c['observed']);px=[];py=[]
            for ci in range(len(c['bottoms'])):
                qs=np.flatnonzero(obs[ci]).tolist()
                if not qs:continue
                others=[j for j in range(len(c['bottoms'])) if j!=ci];states=[(j,) for j in others]
                for k in [2,3]:states+=sorted(combinations(others,k),key=lambda w:h(f'{SEED}:owned:{c["id"]}:{ci}:{w}'))[:4]
                for wi in states:
                    wi=list(wi);px.append(inputs(X[i],Z[i],P[i],[ci],wi,qs,lex,dense,path!='RAG')[0]);py.extend((1-obs[np.ix_(wi,qs)].max(0)).tolist())
            if py:
                xx.append(np.concatenate(px));yy.extend(py);ww.extend([1/len(py)]*len(py))
        x=np.concatenate(xx);y=np.asarray(yy);w=np.asarray(ww);w*=len(w)/w.sum();extra=33 if path=='RAG' else 11;monotone=[0]*(x.shape[1]-extra)+[-1]*extra;es,base=neural_views('development',path);dx=np.concatenate([e['X'].reshape(-1,x.shape[-1]) for e in es]);ends=np.cumsum([e['X'].shape[0]*e['X'].shape[1] for e in es])[:-1];best=None
        print('AUGMENTED CONDITIONAL',path,x.shape,'positive',int(y.sum()),flush=True)
        for leaves in [7,15]:
            for reg in [1,10]:
                m=HistGradientBoostingClassifier(max_iter=160,max_leaf_nodes=leaves,min_samples_leaf=50,l2_regularization=reg,learning_rate=.04,early_stopping=False,monotonic_cst=monotone,random_state=SEED);m.fit(x,y,sample_weight=w);filename=f'neural-{path}-overlap-leaf{leaves}-reg{reg}.joblib';joblib.dump(m,OUT/filename);ps=np.split(m.predict_proba(dx)[:,1],ends);factors=[p.reshape(e['components'].shape[:2]) if e['w'] else np.ones(e['components'].shape[:2]) for p,e in zip(ps,es)]
                for t in [0,.25,.5,.75,1]:
                    weights=np.r_[(1-t)*base,t];bases=[e['components']@weights for e in es];C=[b.mean(1) for b in bases];M=[(b*f).mean(1) for b,f in zip(bases,factors)]
                    for alpha in [0,.1,.3,.5,2/3,1]:
                        ss=[alpha*c+(1-alpha)*mm for c,mm in zip(C,M)];q=float(metric.parent_quality(ss,es).mean());diff={v:float(np.mean([metric.quality(s,e['gain'])-metric.quality(c,e['gain']) for s,c,e in zip(ss,C,es) if e['view']==v])) for v in ['rich','owned-rich']};eligible=alpha<1 and min(diff.values())>=-1e-12;r=dict(path=path,model=filename,sha256=sha(OUT/filename),dimension=x.shape[1],trainingPairs=len(y),positiveNovelty=int(y.sum()),leaves=leaves,reg=reg,newSupportWeight=t,weights=weights.tolist(),alpha=alpha,developmentQuality=q,richDifferences=diff,eligible=eligible);audit.append(r)
                        if eligible and (best is None or q>best['developmentQuality']+1e-12):best=r
                print('NEURAL OVERLAP',path,leaves,reg,'best',json.dumps(best),flush=True)
        if best is None:raise RuntimeError('No development-safe neural policy: '+path)
        profiles[path]=best
    write(OUT/'neural-method-selection.json',dict(profiles=profiles,search=audit,calibrationLockSha256=sha(OUT/'neural-calibration-lock.json'),queryPlanSha256=sha(OUT/'neural-query-plan.json'),codeSha256=sha(__file__)))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['calibrate','train-overlap']);a=p.parse_args();globals()[a.stage.replace('-','_')]()
