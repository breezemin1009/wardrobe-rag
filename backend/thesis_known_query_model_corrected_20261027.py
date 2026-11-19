"""Fixed development and one opening for the explicit follow-up setting."""
from pathlib import Path
import argparse,time
import joblib
import numpy as np
from threadpoolctl import threadpool_limits
from backend import thesis_known_query_study_corrected_20261027 as study
from backend import thesis_repair_models_20261026 as b
from backend import thesis_repair_stable_kernel_20261026 as kernel
from backend.thesis_repair_expanded_models_20261026 import feature_view,make_model,fit
from backend.thesis_repair_final_evaluate_20261026 import values,paired,holm

OUT=study.OUT
VIEW_NAMES=[f'T{q}B{w}' for q in [3,4,6,8] for w in [2,3,4,5]]
SMALL=['T3B2','T3B3','T4B2','T4B3'];LARGE=['T6B4','T6B5','T8B4','T8B5']

def wardrobe_frames(stage):return b.read(OUT/(stage+'-wardrobes.json'))

def condition(S,c,name,alpha):
    view=c['views'][name];qs=view['q'];cc=dict(c)
    cc['contexts']=[c['contexts'][i] for i in qs]
    cc['observed']=np.asarray(c['observed'])[:,qs].tolist()
    cc['knownHistory']=np.asarray(c['knownHistory'])[:,qs].tolist()
    cc['views']={'primary':dict(q=list(range(len(qs))),c=list(range(4)),w=view['w'])}
    return values(np.asarray(S)[:,qs],cc,alpha)

def views(S,c,alpha):return {v:condition(S,c,v,alpha) for v in VIEW_NAMES}

def aggregate(vm):
    return {k:float(np.mean([m[k] for m in vm.values()])) for k in ['NG','selectedGain','selectedNewLinkFraction','maxGain']}

def selection_ng(S,c,alpha):return aggregate(views(S,c,alpha))['NG']

def verify():
    p=b.read(OUT/'protocol.json');assert p['codeSha256']==b.sha(study.__file__)
    for n,s in p['files'].items():assert b.sha(OUT/n)==s
    amendment=b.read(OUT/'size-amendment.json')
    assert amendment['codeSha256']==b.sha(Path(__file__).with_name('thesis_wardrobe_size_amendment_corrected_20261027.py'))
    for n,s in amendment['files'].items():assert b.sha(OUT/n)==s
    if (OUT/'method-lock.json').exists():
        lock=b.read(OUT/'method-lock.json');assert lock['codeSha256']==b.sha(__file__)
        assert lock['protocolSha256']==b.sha(OUT/'protocol.json')
        assert lock['kernelSha256']==b.sha(kernel.__file__)
        assert lock['modelFactorySha256']==b.sha(Path(__file__).with_name('thesis_repair_expanded_models_20261026.py'))
        for n,s in lock['encoderFiles'].items():assert b.sha(OUT/n)==s
        assert lock['sizeAmendmentSha256']==b.sha(OUT/'size-amendment.json')
    return p

def reserve():
    verify();assert not (OUT/'method-lock.json').exists()
    b.write(OUT/'method-lock.json',dict(createdAt=b.stamp(),codeSha256=b.sha(__file__),protocolSha256=b.sha(OUT/'protocol.json'),kernelSha256=b.sha(kernel.__file__),
      change='The original pointwise kernels with exclude_self=False. Near-candidate filters, fixed-block cosine operations, title representation, training architecture parameters and statistics unchanged.',
      modelFactorySha256=b.sha(Path(__file__).with_name('thesis_repair_expanded_models_20261026.py')),
      encoderFiles={n:b.sha(OUT/n) for n in ['encoder.json','vectors.npy','itemIds.json']},
      sizeAmendmentSha256=b.sha(OUT/'size-amendment.json'),
      training='Development parent total weight20, each of four candidates and Q queries equal weight5/Q. No new auxiliary source-pair learning. Full original candidate-outfit neighborhoods do not overlap among development parents.',
      evaluationOpening='Exactly once after selection lock. Zero-gain inputs retained and separately counted. Sixteen paired top3/4/6/8 by bottom2/3/4/5 conditions; each parent receives total weight1. No zero-owned inputs.',
      selectionMetric='Mean of16condition NG values per parent, then mean of parents; one parent contributes once.',
      matchedArchitectures='Report both development-selected architectures with identical alpha fixed to selected RAG alpha on both paths; exploratory ablations only, no test-based winner selection.',
      publication='Explicit exploratory follow-up after two failed studies; do not overwrite their results or claim a fully unseen-query task.',
      noAfterResultChanges=True))
    print('FOLLOW-UP METHOD LOCKED',b.sha(OUT/'method-lock.json'),flush=True)

def extract(stage):
    verify();assert (OUT/'method-lock.json').exists()
    if stage=='test':assert (OUT/'selection-lock.json').exists() and not (OUT/'test-results.json').exists()
    path=OUT/(stage+'-features.npz');assert not path.exists()
    from backend.pog_refined_retrieval import Retriever
    threadpool_limits(1);e=Retriever(OUT);frames=wardrobe_frames(stage);arrays={};start=time.monotonic()
    parts=OUT/(stage+'-parts');parts.mkdir(exist_ok=True)
    binding=dict(methodSha256=b.sha(OUT/'method-lock.json'),framesSha256=b.sha(OUT/(stage+'-wardrobes.json')))
    if (parts/'binding.json').exists():assert b.read(parts/'binding.json')==binding
    else:b.write(parts/'binding.json',binding)
    for i,c in enumerate(frames):
        cp=parts/f'{i:04d}.npz'
        if cp.exists():
            z=np.load(cp);arrays.update({f'{k}{i}':z[k] for k in ['X','F','R']});continue
        qs=sorted(c['contexts']);cs=c['bottoms'][:4];order=[qs.index(q) for q in c['contexts']]
        # One query at a time fixes the title-projection BLAS row count too.
        X=np.concatenate([kernel.base_features(e,[q],cs) for q in qs],axis=1)[:,order]
        ff=[];rr=[]
        for candidate in cs:
            f,r,_=kernel.support_features(e,qs,[candidate],exclude_self=False,exclude_near=True);ff.append(f);rr.append(r)
        F=np.concatenate(ff)[:,order];R=np.concatenate(rr)[:,order]
        assert X.shape==(4,len(qs),514) and F.shape==R.shape==(4,len(qs),12)
        np.savez_compressed(cp,X=X,F=F,R=R)
        arrays.update({f'X{i}':X,f'F{i}':F,f'R{i}':R})
        if i%20==0:print('FOLLOW-UP FEATURES',stage,i,'/',len(frames),'seconds',round(time.monotonic()-start,1),flush=True)
    np.savez_compressed(path,**arrays)
    b.write(OUT/(stage+'-feature-lock.json'),dict(createdAt=b.stamp(),featuresSha256=b.sha(path),framesSha256=b.sha(OUT/(stage+'-wardrobes.json')),methodSha256=b.sha(OUT/'method-lock.json')))
    print('FOLLOW-UP FEATURES SEALED',stage,len(frames),flush=True)

def develop():
    p=verify();assert not (OUT/'selection-lock.json').exists() and not (OUT/'test-features.npz').exists()
    threadpool_limits(1);frames=wardrobe_frames('development');z=np.load(OUT/'development-features.npz');n=len(frames)
    folds=np.asarray([int(study.key('fold:'+c['id'])[:8],16)%5 for c in frames]);q=[len(c['contexts']) for c in frames]
    offsets=np.cumsum([0]+[4*k for k in q]);rows=np.repeat(np.arange(n),[4*k for k in q]);rf=folds[rows]
    y=np.concatenate([np.asarray(c['observed'][:4]).ravel() for c in frames]);w=np.concatenate([np.full(4*k,5/k) for k in q])
    search=[];profiles={};start=time.monotonic()
    for path in ['noRAG','RAG']:
        for spec in p['models']:
            view=spec['view'];family=spec['family'];tag=path+'__'+view+'__'+family
            X=np.concatenate([feature_view(z[f'X{i}'],z[f'F{i}'],z[f'R{i}'],path,view).reshape(4*q[i],-1) for i in range(n)])
            prediction=np.empty(len(y))
            for fold in range(5):
                tr=rf!=fold;dv=~tr;m=fit(make_model(family),X[tr],y[tr],w[tr]);prediction[dv]=m.predict_proba(X[dv])[:,1]
            model=fit(make_model(family),X,y,w);joblib.dump(model,OUT/(tag+'.joblib'));np.save(OUT/(tag+'-oof.npy'),prediction)
            supports=[prediction[offsets[i]:offsets[i+1]].reshape(4,q[i]) for i in range(n)]
            loss=float(np.average(-(y*np.log(np.clip(prediction,1e-7,1-1e-7))+(1-y)*np.log(np.clip(1-prediction,1e-7,1-1e-7))),weights=w))
            for alpha in p['alphas']:
                ng=[selection_ng(s,c,alpha) for s,c in zip(supports,frames)]
                search.append(dict(path=path,view=view,model=family,tag=tag,alpha=alpha,OOF_NG=float(np.mean(ng)),logloss=loss))
            print('FOLLOW-UP OOF',tag,max(r['OOF_NG'] for r in search if r.get('tag')==tag),'seconds',round(time.monotonic()-start,1),flush=True)
        profiles[path]=max([r for r in search if r['path']==path],key=lambda r:(r['OOF_NG'],-r['logloss'],-r['alpha']))
        profiles[path]['modelSha256']=b.sha(OUT/(profiles[path]['tag']+'.joblib'))
    for name in ['dense','char','hybrid']:
        ss=[(1+(z[f'X{i}'][...,512] if name=='dense' else z[f'X{i}'][...,513] if name=='char' else z[f'X{i}'][...,512:514].mean(-1)))/2 for i in range(n)]
        for alpha in p['alphas']:
            search.append(dict(path='direct',model=name,alpha=alpha,OOF_NG=float(np.mean([selection_ng(s,c,alpha) for s,c in zip(ss,frames)]))))
    profiles['direct']=max([r for r in search if r['path']=='direct'],key=lambda r:(r['OOF_NG'],-r['alpha']))
    b.write(OUT/'selection-lock.json',dict(createdAt=b.stamp(),methodSha256=b.sha(OUT/'method-lock.json'),featureSha256=b.sha(OUT/'development-features.npz'),profiles=profiles,search=search,folds=folds.tolist(),parents=n,pairRows=len(y),testOpened=False))
    print('FOLLOW-UP SELECTION',profiles,flush=True)

def evaluate():
    verify();assert not (OUT/'test-results.json').exists() and not (OUT/'test-predictions.json').exists()
    threadpool_limits(1);selection=b.read(OUT/'selection-lock.json');z=np.load(OUT/'test-features.npz');frames=wardrobe_frames('test')
    models={name:joblib.load(OUT/(selection['profiles'][name]['tag']+'.joblib')) for name in ['RAG','noRAG']}
    predictions=[]
    for i,c in enumerate(frames):
        X,F,R=[z[f'{k}{i}'] for k in ['X','F','R']];S={}
        for name in models:
            a=feature_view(X,F,R,name,selection['profiles'][name]['view']);S[name]=models[name].predict_proba(a.reshape(-1,a.shape[-1]))[:,1].reshape(4,len(c['contexts']))
        direct=selection['profiles']['direct']['model'];S['direct']=(1+(X[...,512] if direct=='dense' else X[...,513] if direct=='char' else X[...,512:514].mean(-1)))/2
        S['oldRAG']=F[...,11];S['sameSupportC']=S['RAG'];S['uniform']=np.ones_like(S['RAG'])
        vm={name:views(s,c,1. if name=='sameSupportC' else selection['profiles'][name]['alpha'] if name in selection['profiles'] else 0.) for name,s in S.items()}
        predictions.append(dict(id=c['id'],metrics={name:aggregate(v) for name,v in vm.items()},viewMetrics=vm,support={k:v.tolist() for k,v in S.items()}))
    b.write(OUT/'test-predictions.json',predictions)
    names=['RAG','noRAG','direct','sameSupportC','oldRAG','uniform']
    summary={name:dict(NG=float(np.mean([r['metrics'][name]['NG'] for r in predictions])),selectedGain=float(np.mean([r['metrics'][name]['selectedGain'] for r in predictions]))) for name in names}
    comparisons={}
    for name in ['noRAG','direct','sameSupportC','oldRAG']:
        d=[r['metrics']['RAG']['NG']-r['metrics'][name]['NG'] for r in predictions];comparisons[name]=paired(d,study.SEED+1,study.SEED+2)
    interaction=[]
    for r in predictions:
        advantage=lambda names:np.mean([r['viewMetrics']['RAG'][v]['NG']-r['viewMetrics']['sameSupportC'][v]['NG'] for v in names])
        interaction.append(advantage(LARGE)-advantage(SMALL))
    comparisons['wardrobeSizeInteraction']=paired(interaction,study.SEED+3,study.SEED+4)
    adj=holm([r['rawP'] for r in comparisons.values()])
    for (name,v),a in zip(comparisons.items(),adj):
        v['holmP']=a;v['threeRoundSensitivityP']=min(1.,3*a)
        v['passedExploratoryCriterion']=bool(v['mean']>0 and v['CI95'][0]>0 and v['threeRoundSensitivityP']<.05)
        if name=='wardrobeSizeInteraction':v['twoSidedDifferenceDetected']=bool(v['threeRoundSensitivityP']<.05)
    conditionResults={}
    for v in VIEW_NAMES:
        conditionResults[v]={name:{k:float(np.mean([r['viewMetrics'][name][v][k] for r in predictions])) for k in ['NG','selectedGain','selectedNewLinkFraction','maxGain']} for name in names}
        conditionResults[v]['positiveOpportunityParents']=sum(r['viewMetrics']['RAG'][v]['maxGain']>0 for r in predictions)
    sizeGroups={}
    for label,vs in [('small',SMALL),('large',LARGE)]:
        sizeGroups[label]={name:{k:float(np.mean([np.mean([r['viewMetrics'][name][v][k] for v in vs]) for r in predictions])) for k in ['NG','selectedGain','selectedNewLinkFraction']} for name in names}
    sizeChanges={}
    for name in names:
        sizeChanges[name]={}
        for k in ['NG','selectedGain','selectedNewLinkFraction']:
            d=[np.mean([r['viewMetrics'][name][v][k] for v in LARGE])-np.mean([r['viewMetrics'][name][v][k] for v in SMALL]) for r in predictions]
            sizeChanges[name][k]=paired(d,study.SEED+5,study.SEED+6)
    result=dict(createdAt=b.stamp(),n=len(frames),conditionsPerParent=16,setting='Known wardrobe queries and cold candidate items; paired top3/4/6/8 by owned-bottom2/3/4/5 conditions, including different kinds of existing bottoms.',
      status='Exploratory follow-up after two previously failed repair rounds; does not prove their earlier hypotheses under query-self-excluded retrieval.',
      selectionSha256=b.sha(OUT/'selection-lock.json'),protocolSha256=b.sha(OUT/'protocol.json'),featuresSha256=b.sha(OUT/'test-features.npz'),predictionsSha256=b.sha(OUT/'test-predictions.json'),
      summary=summary,comparisons=comparisons,conditions=conditionResults,sizeGroups=sizeGroups,sizeChanges=sizeChanges,
      zeroGainConditions=sum(r['viewMetrics']['RAG'][v]['maxGain']==0 for r in predictions for v in VIEW_NAMES),
      allFourExploratoryComparisonsPassed=all(comparisons[k]['passedExploratoryCriterion'] for k in ['noRAG','direct','sameSupportC','oldRAG']))
    b.write(OUT/'test-results.json',result);print(result,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['reserve','development_features','test_features','develop','evaluate']);s=ap.parse_args().stage
    if s.endswith('_features'):extract(s.removesuffix('_features'))
    else:globals()[s]()
