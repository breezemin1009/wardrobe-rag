"""Directly learn marginal observed gain; preserve all earlier failed studies.

Support remains nonnegative and pointwise, so M retains the same monotonicity
and API contract. Only the training objective changes, not the label features.
"""
from pathlib import Path
import argparse,inspect,os,time
import joblib,numpy as np
from threadpoolctl import threadpool_limits
from backend import thesis_corpus_support_20261027 as prior
from backend import thesis_multilink_source_20261027 as construction
from backend import thesis_repair_models_20261026 as b
from backend import multilink_metrics as metric
from backend.rank_support_model import NumpyRankSupport
from backend.thesis_repair_expanded_models_20261026 import feature_view
OUT=prior.previous.REPAIR/'direct-gain-objective-20261027';SEED=prior.SEED
key=lambda value:b.stable(str(SEED)+':'+str(value))

def reserve():
    assert not OUT.exists();OUT.mkdir();old=prior.OUT
    for name in ['items.json','reference.json','candidate-pools.json','source-lock.json','vectors.npy','itemIds.json','encoder.json','development-wardrobes.json','development-features.npz']:os.link(old/name,OUT/name)
    oldframes=b.read(old/'development-wardrobes.json');pilot=construction.OLD
    allseen=oldframes+b.read(pilot/'development-wardrobes.json')+b.read(pilot/'test-wardrobes.json')
    # Reuse the already audited construction exactly, with explicit prior-role
    # exclusions and all unused quarantined candidate pools. No score filters.
    code=inspect.getsource(construction.prepare)
    replacements={
      "usedQ=set();usedW=set();usedC=set();usedTitle=set();usedOut=set();allframes={};reports={};start=time.monotonic()":"usedQ={p for c in allseen for p in c['contexts']};usedW={p for c in allseen for p in c['bottoms'][2:]};usedC={p for c in allseen for p in c['bottoms'][:2]};usedTitle={norm[p] for c in allseen for p in c['contexts']+c['bottoms']};usedOut=set().union(*(io[p] for p in usedC));allframes={'development':oldframes};reports={'development':{'parents':len(oldframes)}};start=time.monotonic()",
      "for stage in ['development','test']:":"for stage in ['test']:",
      "pools=b.read(OUT/'candidate-pools.json')":"pools0=b.read(OUT/'candidate-pools.json');pools={'test':sorted(set(pools0['development']+pools0['test']))}"
    }
    for oldtext,newtext in replacements.items():assert code.count(oldtext)==1,oldtext;code=code.replace(oldtext,newtext)
    (OUT/'frozen-construction.py.txt').write_text(code,encoding='utf8')
    namespace={**vars(construction),'OUT':OUT,'SEED':SEED,'key':lambda value:b.stable('2026102839:'+str(value)),'__file__':__file__,'allseen':allseen,'oldframes':oldframes}
    exec(code,namespace);namespace['prepare']()
    p=b.read(OUT/'protocol.json');p.update(createdAt=b.stamp(),codeSha256=b.sha(__file__),constructionSha256=b.sha(OUT/'frozen-construction.py.txt'),
      developmentExposure='All721earliermodel-evaluated profiles are development. Pilot385rawunscoredprofiles also excluded from new test roles and candidate source outfits. Auxiliary corpus was rejected and is not added to new objectives.',
      objectivePlan='Pointwise sigmoid support trained by marginal-gain MSE,opportunity-scaled MSE,or pure difference-weighted pairwise logistic. Width0/32,epochs300/1000. All24models plus previous28strong controls,alpha0/.1/.25/.5/1. Same objectives and budgets on RAG/noRAG. No candidateY enters inference features.',
      continuationGate='Best RAG OOFNG must exceed the previous best nonauxiliaryRAG OOFNG by>=.01 and exceed the new best noRAG OOFNG by>=.015 before any new test prediction. Otherwise preserve development outcomes and do not open a test for this repair.',
      previousDevelopmentSelectionSha256=b.sha(old/'selection-lock.json'),previousTestResultSha256=b.sha(prior.previous.OUT/'test-results.json'),
      selection='Same five parent folds and feature tensors as previous721development;choose all losses includingBCE by OOFNG,thenBCE,then smalleralpha. No forced direct-gain winner. Historical tests remain reportable;no universal-scope or external-preregistration claim.')
    b.write(OUT/'protocol.json',p);print('DIRECT GAIN STUDY RESERVED',p['counts'],flush=True)

def neural(X,a,width,objective,epochs,seed):
    import torch
    from torch import nn
    from torch.nn import functional as F
    torch.set_num_threads(1);torch.manual_seed(seed);torch.use_deterministic_algorithms(True)
    mean=X.reshape(-1,X.shape[-1]).mean(0,dtype=np.float64);scale=X.reshape(-1,X.shape[-1]).std(0,dtype=np.float64);scale[scale<1e-8]=1
    x=torch.tensor((X-mean)/scale,dtype=torch.float32);mask=torch.tensor(a['mask'],dtype=torch.float32);gain=torch.tensor(a['gain'],dtype=torch.float32);counts=torch.tensor(a['counts'],dtype=torch.float32)
    dg=gain[...,0]-gain[...,1];sign=dg.sign();weight=dg.abs()/counts[None,:];denom=gain.max(-1).values.clamp_min(1)
    model=nn.Sequential(nn.Linear(X.shape[-1],width),nn.Tanh(),nn.Linear(width,1)) if width else nn.Sequential(nn.Linear(X.shape[-1],1));opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.01);trace=[]
    for epoch in range(epochs):
        opt.zero_grad();prob=torch.sigmoid(model(x).squeeze(-1));M=torch.einsum('ncq,nvq->nvc',prob,mask)/counts[None,:,None]
        if objective=='MSE':loss=((M-gain/counts[None,:,None])**2).mean()
        elif objective=='scaledMSE':loss=(((M*counts[None,:,None]-gain)/denom[...,None])**2).mean()
        else:loss=(weight*F.softplus(-sign*(M[...,0]-M[...,1])/.1)).sum()/weight.sum().clamp_min(1e-12)
        assert bool(torch.isfinite(loss));loss.backward();assert all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters());opt.step()
        if epoch in [0,99,299,999]:trace.append(dict(epoch=epoch+1,loss=float(loss.detach())))
    layers=[(layer.weight.detach().numpy().copy(),layer.bias.detach().numpy().copy()) for layer in model if isinstance(layer,nn.Linear)];export=NumpyRankSupport(mean,scale,layers)
    with torch.no_grad():expected=torch.sigmoid(model(x[:8]).squeeze(-1)).numpy().ravel()
    actual=export.predict_proba(X[:8].reshape(-1,X.shape[-1]))[:,1];error=float(abs(actual-expected).max());assert error<2e-6
    return export,trace,error

def development_reserve():
    assert OUT.exists() and not (OUT/'protocol.json').exists() and not (OUT/'selection-lock.json').exists()
    b.write(OUT/'new-source-availability.json',dict(createdAt=b.stamp(),candidateProfiles=14,requiredMinimum=60,predictionsOpened=False,
      interpretation='The strict previous-role/source/title exclusion construction found only14freshprofiles,below the fixed minimum60. It failed before saving final test frames/protocol. Do not lower the minimum or relabel old profiles as fresh. Development-only objective validation can proceed;new evaluation requires additional adequate source data.',
      preservedConstructionSha256=b.sha(OUT/'frozen-construction.py.txt')))
    p=dict(createdAt=b.stamp(),codeSha256=b.sha(__file__),seed=SEED,developmentOnly=True,parents=721,
      files={name:b.sha(OUT/name) for name in ['items.json','reference.json','development-wardrobes.json']},
      objectivePlan='MSE,opportunity-scaledMSE,pure weighted pairlogistic;NN0/32;epochs300/1000;all24newmodels on the same721development,fiveparentfolds andalpha0/.1/.25/.5/1as previous28controls. Nonnegative pointwiseS andoriginalMformula unchanged.',
      continuationGate='RAG OOFNG improvement>=.01 over previousbestnonauxRAG,and>=.015overnewbestnoRAG;also an adequate newsource cohort>=60mustbefrozen beforeanynewtestpredictions. No forced positive result or reusing old evaluation as newtest.',
      previousDevelopmentSelectionSha256=b.sha(prior.OUT/'selection-lock.json'),priorTestResultSha256=b.sha(prior.previous.OUT/'test-results.json'),
      availabilitySha256=b.sha(OUT/'new-source-availability.json'))
    b.write(OUT/'protocol.json',p);print('DIRECT GAIN DEVELOPMENT RESERVED;FRESH SOURCE AVAILABILITY BELOW MINIMUM',flush=True)

def develop():
    assert not (OUT/'selection-lock.json').exists() and not (OUT/'test-features.npz').exists();p=b.read(OUT/'protocol.json');assert p['codeSha256']==b.sha(__file__)
    threadpool_limits(1);frames=b.read(OUT/'development-wardrobes.json');a=metric.arrays(frames);z=np.load(OUT/'development-features.npz');folds=np.asarray([int(key('fold:'+c['id'])[:8],16)%5 for c in frames]);old=b.read(prior.OUT/'selection-lock.json');policies=[{**row,'modelDirectory':str(prior.OUT)} for row in old['search']];traces={};start=time.monotonic();n=len(frames)
    for path in ['noRAG','RAG']:
        X=feature_view(z['X'],z['F'],z['R'],path,'compact')
        for width in [0,32]:
            for objective in ['MSE','scaledMSE','pairOnly']:
                for epochs in [300,1000]:
                    tag=f'{path}__NN{width}__{objective}__E{epochs}';oof=np.empty((n,2,8));hist=[]
                    for fold in range(5):
                        tr=folds!=fold;dv=~tr;aa={k:(v if k=='counts' else v[tr]) for k,v in a.items()};m,trace,error=neural(X[tr],aa,width,objective,epochs,SEED+fold);oof[dv]=m.predict_proba(X[dv].reshape(-1,X.shape[-1]))[:,1].reshape(-1,2,8);hist.append(dict(fold=fold,trace=trace,exportError=error))
                    m,trace,error=neural(X,a,width,objective,epochs,SEED+5);joblib.dump(m,OUT/(tag+'.joblib'));np.save(OUT/(tag+'-oof.npy'),oof);traces[tag]=dict(folds=hist,full=trace,exportError=error)
                    loss=float(np.mean(-(a['Y']*np.log(np.clip(oof,1e-8,1-1e-8))+(1-a['Y'])*np.log(np.clip(1-oof,1e-8,1-1e-8)))))
                    for alpha in [0,.1,.25,.5,1.]:policies.append(dict(path=path,tag=tag,view='compact',family='NN'+str(width),objective=objective,epochs=epochs,auxiliary=False,alpha=alpha,OOF_NG=float(metric.ng_values(oof,a,alpha).mean()),BCE=loss,modelDirectory=str(OUT)))
                    print('DIRECT GAIN OOF',tag,max(row['OOF_NG'] for row in policies if row['tag']==tag),'seconds',round(time.monotonic()-start,1),flush=True)
    choose=lambda rows:max(rows,key=lambda x:(x['OOF_NG'],-x['BCE'],-x['alpha']))
    profiles=dict(proposed=choose([p for p in policies if p['path']=='RAG']),bestNoRAG=choose([p for p in policies if p['path']=='noRAG']),directTitle=choose([p for p in policies if p['path']=='direct']))
    improvement=profiles['proposed']['OOF_NG']-old['profiles']['bestNonAuxiliaryRAG']['OOF_NG'];advantage=profiles['proposed']['OOF_NG']-profiles['bestNoRAG']['OOF_NG'];passed=bool(improvement>=.01 and advantage>=.015)
    b.write(OUT/'training-traces.json',traces);b.write(OUT/'selection-lock.json',dict(createdAt=b.stamp(),profiles=profiles,search=policies,folds=folds.tolist(),parents=n,previousBestRAG=old['profiles']['bestNonAuxiliaryRAG'],OOFImprovement=improvement,OOFNoRAGAdvantage=advantage,developmentContinuationGatePassed=passed,protocolSha256=b.sha(OUT/'protocol.json'),codeSha256=b.sha(__file__)))
    print('DIRECT GAIN SELECTION',profiles,'improvement',improvement,'noRAG advantage',advantage,'CONTINUATION',passed,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['reserve','development_reserve','develop']);globals()[ap.parse_args().stage]()
