"""Equal-budget model selection for the separately constructed multi-link task."""
from pathlib import Path
import argparse,time
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from backend import thesis_pog_disjoint_study_20261027 as s
from backend import thesis_pog_disjoint_features_20261027 as features
from backend import thesis_repair_models_20261026 as b
from backend import multilink_metrics as metric
from backend.rank_support_model import NumpyRankSupport
from backend.thesis_direct_gain_objective_20261027 import neural as gain_neural
from backend.thesis_repair_expanded_models_20261026 import feature_view,make_model,fit
OUT=s.OUT

def neural(X,a,width,lam,seed):
    import torch
    from torch import nn
    from torch.nn import functional as F
    torch.set_num_threads(1);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    mean=X.reshape(-1,X.shape[-1]).mean(0,dtype=np.float64);scale=X.reshape(-1,X.shape[-1]).std(0,dtype=np.float64);scale[scale<1e-8]=1
    x=torch.tensor((X-mean)/scale,dtype=torch.float32);y=torch.tensor(a['Y'],dtype=torch.float32);mask=torch.tensor(a['mask'],dtype=torch.float32);gain=torch.tensor(a['gain'],dtype=torch.float32);counts=torch.tensor(a['counts'],dtype=torch.float32)
    dG=gain[...,0]-gain[...,1];sign=dG.sign();weight=dG.abs()/counts[None,:]
    model=nn.Sequential(nn.Linear(X.shape[-1],width),nn.Tanh(),nn.Linear(width,1)) if width else nn.Sequential(nn.Linear(X.shape[-1],1))
    opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01);trace=[]
    for epoch in range(300):
        opt.zero_grad();logits=model(x).squeeze(-1);prob=torch.sigmoid(logits);bce=F.binary_cross_entropy_with_logits(logits,y)
        M=torch.einsum('ncq,nvq->nvc',prob,mask)/counts[None,:,None]
        rank=(weight*F.softplus(-sign*(M[...,0]-M[...,1])/.1)).sum()/weight.sum().clamp_min(1e-12)
        loss=bce+lam*rank;assert bool(torch.isfinite(loss));loss.backward();assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters());opt.step()
        if epoch in [0,49,99,199,299]:trace.append(dict(epoch=epoch+1,BCE=float(bce.detach()),pairRank=float(rank.detach()),loss=float(loss.detach())))
    layers=[(layer.weight.detach().numpy().copy(),layer.bias.detach().numpy().copy()) for layer in model if isinstance(layer,nn.Linear)];frozen=NumpyRankSupport(mean,scale,layers)
    with torch.no_grad():ref=torch.sigmoid(model(x[:min(8,len(x))]).squeeze(-1)).numpy().ravel()
    actual=frozen.predict_proba(X[:min(8,len(X))].reshape(-1,X.shape[-1]))[:,1];err=float(np.max(abs(ref-actual)));assert err<2e-6
    return frozen,trace,err

def reserve():
    assert not (OUT/'learning-lock.json').exists() and not (OUT/'test-features.npz').exists()
    p=b.read(OUT/'protocol.json');assert p['codeSha256']==b.sha(s.__file__)
    for n,digest in p['files'].items():assert b.sha(OUT/n)==digest
    b.write(OUT/'learning-lock.json',dict(createdAt=b.stamp(),codeSha256=b.sha(__file__),metricSha256=b.sha(metric.__file__),featureSha256=b.sha(features.__file__),headSha256=b.sha(Path(__file__).with_name('rank_support_model.py')),
      protocolSha256=b.sha(OUT/'protocol.json'),
      objective='BCE plus lambda times difference-weighted pairwise logistic loss on(M0-M1)/0.1. Weight=abs(G0-G1)/queryCount; normalize by total nonzero weight. Equal-gain conditions have zero rank loss but retain BCE and all evaluation metrics. Inspired by RankNet,not claimed as its original invention.',
      models='Per RAG/noRAG path: compact neural widths0/32 x lambda0/.5/1;MSE-of-marginal-gain widths0/32;compact/wide HGB_leaf7/15. All24models use same97parent labels,folds,alphas and selection rule. Neural300epochs,AdamWlr.003decay.01;HGB200iterations,lr.05,L2=10,minleaf20,fixedseed2026102641.',
      proposal='Select best RAG model over all losses,includinglambda0; do not force adoption of a ranking term that loses development validation. No claim that the earlier listwise objective succeeded.',
      folds='Five fixed parent-ID hash folds; whole16conditions stay together. No retained profile shares query/owned/candidate IDs or normalized titles with another profile. All test-source candidate outfits excluded before any fitting.',
      selection='Highest parent-average16condition OOFNG,then lower BCE,then smaller alpha,then stable simpler-model order. C and M both use the actual3/4/6/8query subset.',
      alpha=[0,.1,.25,.5,1.],
      primary=['bestNoRAG','directTitle','sameSupportC'],
      statistics='10000paired bootstrap,500000two-sided sign flips,Holm3,plus factor6 source-evaluation-stage sensitivity. Full original-role source neighbourhoods are disjoint across97development and300test parents. EarlierASOSandPOGforcedconflict outcomes remain reportable;notexternalpreregistration.',
      secondary='Best pointwiseRAG,matched architecture retrieval/loss ablations,oldforwardK32,uniform,H-only equivalence,all16size cells,small/large and factor effects,sourcekind,knowncoverage,multi-gain and unequal-gain strata,alpha curve. All retained and descriptive; no outcome-based subgroup claims.',
      opening='Exactly one new test opening after selection,rawsourceaudit,numericalpreflight,evaluation and dependence plans are frozen.'))
    print('MULTILINK LEARNING LOCKED',flush=True)

def develop():
    assert not (OUT/'selection-lock.json').exists() and not (OUT/'test-features.npz').exists()
    lock=b.read(OUT/'learning-lock.json');assert lock['codeSha256']==b.sha(__file__) and lock['metricSha256']==b.sha(metric.__file__) and lock['featureSha256']==b.sha(features.__file__)
    assert b.read(OUT/'numerical-preflight.json')['allPassed'] and b.read(OUT/'raw-source-audit.json')['allPassed']
    threadpool_limits(1);z=np.load(OUT/'development-features.npz');frames=b.read(OUT/'development-wardrobes.json');a=metric.arrays(frames);n=len(frames);folds=np.asarray([int(s.key('fold:'+c['id'])[:8],16)%5 for c in frames]);search=[];traces={};start=time.monotonic()
    def record(path,tag,view,family,width,lam,oof,model,history=None,objective='BCE'):
        joblib.dump(model,OUT/(tag+'.joblib'));np.save(OUT/(tag+'-oof.npy'),oof)
        loss=float(np.mean(-(a['Y']*np.log(np.clip(oof,1e-8,1-1e-8))+(1-a['Y'])*np.log(np.clip(1-oof,1e-8,1-1e-8)))))
        for alpha in lock['alpha']:search.append(dict(path=path,tag=tag,view=view,family=family,width=width,lambdaRank=lam,objective=objective if lam==0 else 'BCE+pair',alpha=alpha,OOF_NG=float(metric.ng_values(oof,a,alpha).mean()),BCE=loss))
        if history is not None:traces[tag]=history
        print('MULTILINK OOF',tag,'NG',max(r['OOF_NG'] for r in search if r['tag']==tag),'seconds',round(time.monotonic()-start,1),flush=True)
    for path in ['noRAG','RAG']:
        for width in [0,32]:
            X=feature_view(z['X'],z['F'],z['R'],path,'compact')
            for lam in [0,.5,1.]:
                tag=f'{path}__compact__NN{width}__L{lam:g}';oof=np.empty((n,2,8));hist=[]
                for fold in range(5):
                    tr=folds!=fold;dv=~tr;aa={k:(v if k=='counts' else v[tr]) for k,v in a.items()};model,history,err=neural(X[tr],aa,width,lam,s.SEED+fold);oof[dv]=model.predict_proba(X[dv].reshape(-1,X.shape[-1]))[:,1].reshape(-1,2,8);hist.append(dict(fold=fold,trace=history,maxExportError=err))
                model,history,err=neural(X,a,width,lam,s.SEED+5);record(path,tag,'compact',f'NN{width}',width,lam,oof,model,dict(folds=hist,full=history,maxExportError=err))
        for width in [0,32]:
            X=feature_view(z['X'],z['F'],z['R'],path,'compact');tag=f'{path}__compact__NN{width}__MSE';oof=np.empty((n,2,8));hist=[]
            for fold in range(5):
                tr=folds!=fold;dv=~tr;aa={k:(v if k=='counts' else v[tr]) for k,v in a.items()};model,history,err=gain_neural(X[tr],aa,width,'MSE',300,s.SEED+fold);oof[dv]=model.predict_proba(X[dv].reshape(-1,X.shape[-1]))[:,1].reshape(-1,2,8);hist.append(dict(fold=fold,trace=history,maxExportError=err))
            model,history,err=gain_neural(X,a,width,'MSE',300,s.SEED+5);record(path,tag,'compact',f'NN{width}',width,0,oof,model,dict(folds=hist,full=history,maxExportError=err),objective='MSE')
        for view in ['compact','wide']:
            X=feature_view(z['X'],z['F'],z['R'],path,view)
            for family in ['HGB_leaf7','HGB_leaf15']:
                tag=f'{path}__{view}__{family}';oof=np.empty((n,2,8))
                for fold in range(5):
                    tr=folds!=fold;dv=~tr;model=fit(make_model(family),X[tr].reshape(-1,X.shape[-1]),a['Y'][tr].ravel(),np.ones(tr.sum()*16));oof[dv]=model.predict_proba(X[dv].reshape(-1,X.shape[-1]))[:,1].reshape(-1,2,8)
                model=fit(make_model(family),X.reshape(-1,X.shape[-1]),a['Y'].ravel(),np.ones(n*16));record(path,tag,view,family,7 if family.endswith('7') else 15,0,oof,model)
    for name in ['dense','char','hybrid']:
        x=z['X'][...,512:514];S=(1+(x[...,0] if name=='dense' else x[...,1] if name=='char' else x.mean(-1)))/2
        for alpha in lock['alpha']:search.append(dict(path='direct',tag=name,view='compact',family=name,width=0,lambdaRank=0,alpha=alpha,OOF_NG=float(metric.ng_values(S,a,alpha).mean()),BCE=0))
    choose=lambda rows:max(rows,key=lambda r:(r['OOF_NG'],-r['BCE'],-r['alpha'],-r['width'],-r['lambdaRank']))
    profiles=dict(proposed=choose([r for r in search if r['path']=='RAG']),bestNoRAG=choose([r for r in search if r['path']=='noRAG']),bestPointwiseRAG=choose([r for r in search if r['path']=='RAG' and r['lambdaRank']==0 and r['objective']=='BCE']),directTitle=choose([r for r in search if r['path']=='direct']))
    assert len(search)==135
    for p in profiles.values():
        if p['path']!='direct':p['modelSha256']=b.sha(OUT/(p['tag']+'.joblib'))
    b.write(OUT/'training-traces.json',traces);b.write(OUT/'selection-lock.json',dict(createdAt=b.stamp(),profiles=profiles,search=search,folds=folds.tolist(),parents=n,pairRows=n*16,learningSha256=b.sha(OUT/'learning-lock.json'),featuresSha256=b.sha(OUT/'development-features.npz'),modelFiles={p.name:b.sha(p) for p in OUT.glob('*.joblib')}))
    print('MULTILINK SELECTION LOCKED',profiles,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['reserve','develop']);globals()[ap.parse_args().stage]()
