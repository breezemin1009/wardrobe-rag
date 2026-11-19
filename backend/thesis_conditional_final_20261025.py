"""Immutable final evaluation and independent audits of expected novelty."""
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
import argparse,json
import numpy as np
from backend.thesis_conditional_models_20261025 import OUT,SEED,views,method_profiles
from backend.thesis_conditional_controls_20261025 import rank_features,support_values
from backend.thesis_study_20261023 import ROOT,read,write,sha,source
from backend import thesis_adaptive_policy_20261024 as common

def source_audit(stage='test'):
    items,edges,g,rev,_,_=source();io={p:set(v) for p,v in read(ROOT/'raw-outfit-incidence-audit-20261023/item-outfits.json').items()};eo={tuple(k.split(':')):set(v) for k,v in read(ROOT/'adversarial-audit-20261023/source-edge-outfits.json').items()}
    previous=sum([read(common.OLD/f'{s}-inputs.json') for s in ['training','development','test']],[]);pids={p for c in previous for p in c['contexts']+c['candidates']+c['owned']};pids.update(p for c in read(common.OUT/'test-frames.json') for p in c['contexts']+c['bottoms'])
    trainbottoms={p for stage in ['training','development'] for c in read(OUT/f'{stage}-frames.json') for p in c['bottoms']};trainout=set().union(*(io[p] for p in trainbottoms));refs=read(OUT/'reference.json');refids={r[s] for r in refs for s in ['top','bottom']};refouts=set().union(*(eo[(r['top'],r['bottom'])] for r in refs));oldtestids={p for c in read(common.OLD/'test-inputs.json') for p in c['contexts']+c['candidates']+c['owned']};oldtestids.update(p for c in read(common.OUT/'test-frames.json') for p in c['contexts']+c['bottoms']);oldtestouts=set().union(*(io[p] for p in oldtestids))
    seen=set();seenids=set();checks=0;degreechecks=0;exposed=0;known=0;degrees=[]
    previousouts=set().union(*(io[p] for p in pids));strict_overlap=0
    for c in read(OUT/f'{stage}-frames.json'):
        ids=set(c['contexts']+c['bottoms']);outs=set().union(*(io[p] for p in ids));assert not seen&outs and not seenids&ids;seen.update(outs);seenids.update(ids)
        bottoms=set(c['bottoms']);bo=set().union(*(io[p] for p in bottoms));assert not bottoms&pids and not bo&trainout and not bottoms&refids and not bo&refouts
        strict_overlap+=bool(bo&previousouts)
        if stage=='strict-test':assert not bo&previousouts
        exposed+=bool(bo&oldtestouts);known+=sum(q in refids for q in c['contexts']);obs=np.asarray(c['observed'])
        for i,b in enumerate(c['bottoms']):
            for j,q in enumerate(c['contexts']):assert obs[i,j]==int((q,b) in edges);checks+=1
        counts=Counter(len(rev[b]) for b in c['bottoms'][:-1]);assert all(n%2==0 for n in counts.values());degreechecks+=1;degrees.extend(len(rev[b]) for b in c['bottoms'])
    report=dict(stage=stage,testParents=len(read(OUT/f'{stage}-frames.json')),parentMasterGarmentOverlap=0,parentMasterSourceOutfitOverlap=0,
        finalBottomIdOverlapWithAllPriorPrimary=0,finalBottomOutfitOverlapWithTrainingDevelopmentBottoms=0,finalBottomIdOverlapWithReference=0,finalBottomOutfitOverlapWithReference=0,
        finalParentsSharingPriorDiagnosticTestOutfits=exposed,parentsSharingAnyOfThe1323PriorPrimaryInputOutfits=strict_overlap,referenceKnownContextTops=known,exactLabelChecks=checks,degreeMatchedControlChecks=degreechecks,
        selectedBottomDegreeQuantiles=np.quantile(degrees,[0,.25,.5,.75,1]).tolist(),
        scope='Fresh candidate IDs, source-outfit-disjoint from current supervised bottom training/development and reference; warm context inputs and prior diagnostic outfit exposure explicitly permitted. Final parent master inputs are mutually disjoint. Internal, previously inspected POG corpus.',
        sourceHashes={n:sha(ROOT/n) for n in ['raw-outfit-incidence-audit-20261023/item-outfits.json','adversarial-audit-20261023/source-edge-outfits.json']},codeSha256=sha(__file__))
    write(OUT/('source-audit.json' if stage=='test' else 'strict-source-audit.json'),report);print('SOURCE AUDIT',json.dumps(report),flush=True)

def scores(es):
    import joblib
    sel=method_profiles();ss={'same-support-C':[e['C'] for e in es],'positive-difference':[e['oldM'] for e in es]};estimated={}
    for path,name,key,basekey,ckey in [('RAG','proposed','X','base','C'),('no-RAG','no-RAG','noRAGX','noRAGBase','noRAGC')]:
        r=sel[path];m=joblib.load(OUT/r['model']);x=np.concatenate([e[key].reshape(-1,e[key].shape[-1]) for e in es]);ends=np.cumsum([e[key].shape[0]*e[key].shape[1] for e in es])[:-1];pred=np.split(m.predict_proba(x)[:,1],ends)
        M=[(e[basekey]*(p.reshape(e[basekey].shape) if e['w'] else 1)).mean(1) for e,p in zip(es,pred)];estimated[name]=M;ss[name]=[r['alpha']*e[ckey]+(1-r['alpha'])*v for e,v in zip(es,M)]
    ss['conditional-only']=estimated['proposed']
    controls=read(OUT/'control-selection.json');allrank=np.concatenate([rank_features(e) for e in es]);ends=np.cumsum([len(e['c']) for e in es])[:-1]
    for name,r in controls['profiles'].items():
        cfg=r['config']
        if name=='learned-ranker':ss[name]=list(np.split(joblib.load(OUT/cfg['path']).predict(allrank),ends));continue
        values=[]
        for e in es:
            if name=='old-CM':
                k=cfg['kernel'];c=e['rawKernels'][:,:,k];w=e['oldRawKernels'][:,:,k];prior=w.max(0) if e['w'] else np.zeros(c.shape[1]);a=cfg['alpha'];s=a*c.mean(1)+(1-a)*np.maximum(c-prior,0).mean(1)
            else:
                z=support_values(e,cfg['weights'])
                if name=='development-selected-C':s=z[0]
                elif name=='residual-max':s=z[2]
                elif name=='residual-product':s=z[3]
                else:s=z[0]*(1-e['maxLex' if name=='char-novelty' else 'maxDense'])
            values.append(s)
        ss[name]=values
    m=joblib.load(OUT/'unconstrained-novelty.joblib');x=np.concatenate([e['X'].reshape(-1,e['X'].shape[-1]) for e in es]);ends=np.cumsum([e['X'].shape[0]*e['X'].shape[1] for e in es])[:-1];pred=np.split(m.predict_proba(x)[:,1],ends);a=sel['RAG']['alpha']
    ss['unconstrained-novelty']=[a*e['C']+(1-a)*(e['base']*(p.reshape(e['base'].shape) if e['w'] else 1)).mean(1) for e,p in zip(es,pred)]
    return ss,estimated

def development_replay():
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    es=views('development');ss,_=scores(es);rates={n:float(common.parent_quality(s,es).mean()) for n,s in ss.items()};profiles=method_profiles();controls=read(OUT/'control-selection.json')['profiles']
    assert abs(rates['proposed']-profiles['RAG']['developmentQuality'])<1e-12
    assert abs(rates['no-RAG']-profiles['no-RAG']['developmentQuality'])<1e-12
    for n,p in controls.items():assert abs(rates[n]-p['developmentQuality'])<1e-12,(n,rates[n],p['developmentQuality'])
    write(OUT/'development-replay.json',dict(allPassed=True,rates=rates,parents=len({e['parent'] for e in es}),views=len(es)));print('DEVELOPMENT REPLAY',json.dumps(rates),flush=True)

def freeze():
    if (OUT/'final-plan.json').exists() or (OUT/'test-pair-features.npz').exists():raise RuntimeError('Final test already locked/opened')
    names=['protocol.json','support-lock.json','development-plan.json','novelty-selection.json','method-selection.json','control-plan.json','control-selection.json','source-audit.json','strict-source-audit.json','development-replay.json','development-invariants.json','training-pair-feature-lock.json','development-pair-feature-lock.json',
        'reference-pca.joblib','unconstrained-novelty.joblib','training-frames.json','development-frames.json','test-frames.json','strict-test-frames.json']
    names+=list(read(OUT/'support-lock.json')['models'])
    names+=[r['model'] for r in method_profiles().values()]
    names+=[r['config']['path'] for r in read(OUT/'control-selection.json')['profiles'].values() if 'path' in r['config']]
    code=['thesis_conditional_seal_20261025.py','thesis_conditional_final_20261025.py','thesis_conditional_novelty_20261025.py','thesis_conditional_models_20261025.py','thesis_conditional_controls_20261025.py','pog_retrieval.py','pog_refined_retrieval.py','learned_case_utility.py','thesis_adaptive_policy_20261024.py']
    lock=dict(createdAt=datetime.now(timezone.utc).isoformat(),files={n:sha(OUT/n) for n in names},referenceSha256=sha(OUT/'reference.json'),implementationHashes={n:sha(Path(__file__).parent/n) for n in code},
        primaryContrasts=read(OUT/'development-plan.json')['primaryContrasts'],
        primary='Mean of each independent parent mean normalized added observed-link gain across its applicable predefined views. Zero-gain views excluded and explicitly counted. Parent, not view, is resampling unit.',
        statistics='10000 paired parent bootstrap resamples;50000 paired sign flips;two-sided Holm correction across the3 primary contrasts, individual95% intervals. Other controls and per-view differences are secondary descriptive intervals, not multiplicity-corrected superiority claims.',
        adoption=read(OUT/'development-plan.json')['adoption'],
        history='V2 single-gain and rich315 diagnostics, V3 final179 failure, and earlier development studies retained. Broad fresh-candidate transfer can reuse prior diagnostic outfits through different IDs. Separate36-case confirmation excludes all1323 prior-primary-input source outfits from its candidate/owned bottoms. The two new cohorts may share input sources with each other and are never pooled or called independent external replications. Both are from inspected POG.',
        noRetuning='No method, hyperparameter, cohort, metric, acceptance condition or comparator changes after this opening.',codeSha256=sha(__file__))
    write(OUT/'final-plan.json',lock);(OUT/'frozen-final.py').write_bytes(Path(__file__).read_bytes());print('FINAL LOCK',sha(OUT/'final-plan.json'),flush=True)

def interval(a):
    a=np.asarray(a);rng=np.random.default_rng(SEED);boots=[]
    for _ in range(100):boots.extend(a[rng.integers(0,len(a),(100,len(a)))].mean(1))
    return dict(mean=float(a.mean()),ci95=np.quantile(boots,[.025,.975]).tolist())

def evaluate(stage='test'):
    from backend.thesis_adversarial_audit_20261023 import gain_metric,exact_tied_dcg
    from threadpoolctl import threadpool_limits
    threadpool_limits(1)
    prefix='' if stage=='test' else 'strict-'
    if (OUT/f'{prefix}final-report.json').exists():raise RuntimeError('Final outcomes already evaluated')
    lock=read(OUT/'final-plan.json')
    for n,value in lock['files'].items():assert sha(OUT/n)==value,n
    for n,value in lock['implementationHashes'].items():assert sha(Path(__file__).parent/n)==value,n
    assert sha(OUT/'reference.json')==lock['referenceSha256']
    es=views(stage);ss,est=scores(es);rates={n:common.parent_quality(s,es) for n,s in ss.items()};comparisons={};rng=np.random.default_rng(SEED)
    for n in ss:
        if n=='proposed':continue
        d=rates['proposed']-rates[n];r=interval(d)
        if n in lock['primaryContrasts']:
            count=0
            for _ in range(500):count+=np.count_nonzero(np.abs((d*rng.choice([-1.,1.],(100,len(d)))).mean(1))>=abs(d.mean())-1e-14)
            r['p']=(count+1)/50001
        comparisons[n]=r
    last=0
    for rank,n in enumerate(sorted(lock['primaryContrasts'],key=lambda k:comparisons[k]['p'])):
        last=max(last,min(1.,comparisons[n]['p']*(3-rank)));comparisons[n]['holmP']=last
    frames=read(OUT/f'{stage}-frames.json');byview={};raw=[];maxerr=0
    for view in frames[0]['views']:
        ix=[i for i,e in enumerate(es) if e['view']==view];rr={n:np.asarray([common.quality(s[i],es[i]['gain']) for i in ix]) for n,s in ss.items()}
        byview[view]=dict(n=len(ix),excludedNoGain=len(frames)-len(ix),rates={n:interval(v) for n,v in rr.items()},differences={n:interval(rr['proposed']-rr[n]) for n in ['same-support-C','development-selected-C','positive-difference','old-CM']})
    for i,e in enumerate(es):
        metrics={}
        for n,s in ss.items():
            val=gain_metric(s[i],e['gain']);maxerr=max(maxerr,abs(val-common.quality(s[i],e['gain'])));metrics[n]=dict(NG=val,nDCG=exact_tied_dcg(s[i],e['gain']))
        raw.append(dict(id=e['id'],parent=e['parent'],view=e['view'],C=e['C'].tolist(),M=est['proposed'][i].tolist(),gain=e['gain'].tolist(),scores={n:s[i].tolist() for n,s in ss.items()},metrics=metrics))
    assert maxerr<1e-12
    main=all(comparisons[n]['mean']>0 and comparisons[n]['holmP']<.05 for n in lock['primaryContrasts']);rich=all(byview[v]['differences']['same-support-C']['mean']>=-1e-12 for v in ['rich','owned-rich']);alpha=method_profiles()['RAG']['alpha']
    strata={}
    for cat in sorted({c['category'][1] for c in frames}):
        keep=np.asarray([c['category'][1]==cat for c in frames]);strata[cat]=dict(n=int(keep.sum()),rates={n:float(v[keep].mean()) for n,v in rates.items()})
    result=dict(createdAt=datetime.now(timezone.utc).isoformat(),stage=stage,n=len(frames),applicableViews=len(es),rates={n:interval(v) for n,v in rates.items()},comparisons=comparisons,byView=byview,strata=strata,
        trueGainDistribution={v:dict(Counter(int(e['gain'].max()) for e in es if e['view']==v)) for v in byview},
        independentMetricChecks=len(es)*len(ss),metricMaxError=maxerr,primaryCriterionMet=main,richPointSafeguardMet=rich,alpha=alpha,adoptionCriterionMet=main and rich and alpha<1,
        finalPlanSha256=sha(OUT/'final-plan.json'),scope=lock['history'])
    write(OUT/f'{prefix}final-predictions.json',raw);write(OUT/f'{prefix}final-report.json',result);print(json.dumps(result),flush=True)

def adoption():
    if (OUT/'adoption-report.json').exists():raise RuntimeError('Adoption decision already recorded')
    broad=read(OUT/'final-report.json');strict=read(OUT/'strict-final-report.json');primary=read(OUT/'final-plan.json')['primaryContrasts'];point=all(strict['comparisons'][n]['mean']>=-1e-12 for n in primary)
    result=dict(adoptionCriterionMet=broad['adoptionCriterionMet'] and point,primaryBroadCriterionMet=broad['adoptionCriterionMet'],strictPointSafeguardMet=point,
        broadReportSha256=sha(OUT/'final-report.json'),strictReportSha256=sha(OUT/'strict-final-report.json'),finalPlanSha256=sha(OUT/'final-plan.json'),
        scope='Predeclared broad paired superiority and rich-view point safeguards, plus separate strict-cohort point-direction confirmation. No claim of statistical noninferiority, external replication or superiority over every learned control.')
    write(OUT/'adoption-report.json',result);print('ADOPTION',json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['source-audit','development-replay','freeze','evaluate','adoption']);p.add_argument('--split',choices=['test','strict-test'],default='test');a=p.parse_args()
    if a.stage in ['source-audit','evaluate']:globals()[a.stage.replace('-','_')](a.split)
    else:globals()[a.stage.replace('-','_')]()
