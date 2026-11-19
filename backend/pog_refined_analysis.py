"""Development-only weight choice, then one frozen prospective evaluation."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import argparse,json
import numpy as np
from sklearn.model_selection import KFold
from backend.pog_cold_study import read,sha,stable,SEED
from backend.pog_refined_study import OUT
from backend.data_io import write_json_atomic as write

ALPHAS=[0,.1,.2,.3,.4,.5,.6,2/3,.7,.8,.9,1.]
POLICIES=[dict(alpha=a,marginal=m) for a in ALPHAS for m in ['positive-difference','product-residual','char-novelty','dense-novelty']]

def targets(cases):
    obs=np.asarray([r['observed'] for r in cases]);existing=np.asarray([r['existing'] for r in cases]);return (obs*(1-existing[:,None,:])).sum(-1),obs.mean(-1)
def tied(scores,target):
    mask=np.isclose(scores,scores.max(-1,keepdims=True),atol=1e-10,rtol=0);return (mask*target).sum(-1)/mask.sum(-1)
def scores(fs,profile):
    rows=fs[profile['configuration']];c=np.array([r['C'] for r in rows]);m=np.array([r[profile['policy']['marginal']] for r in rows]);a=profile['policy']['alpha'];return a*c+(1-a)*m
def select(cases,fs,only_c=False):
    target,_=targets(cases);best=None
    for config in fs:
        policies=[dict(alpha=1,marginal='positive-difference')] if only_c else POLICIES
        for policy in policies:
            profile=dict(configuration=config,policy=policy);quality=float(tied(scores(fs,profile),target).mean());cfg=json.loads(config)
            priority=(round(quality,12),policy['alpha'],policy['marginal']=='positive-difference',-cfg['k'],cfg['mode'])
            if best is None or priority>best[0]:best=(priority,{**profile,'developmentRate':quality})
    return best[1]
def summary(values,seed=SEED):
    a=np.asarray(values,float);rng=np.random.default_rng(seed)
    boots=np.concatenate([a[rng.integers(0,len(a),(500,len(a)))].mean(1) for _ in range(12)])
    null=np.concatenate([(rng.choice([-1,1],(1000,len(a)))*a).mean(1) for _ in range(20)])
    return dict(n=len(a),mean=float(a.mean()),ci95=list(map(float,np.quantile(boots,[.025,.975]))),pTwoSided=float((np.sum(np.abs(null)>=abs(a.mean())-1e-12)+1)/(len(null)+1)))

def develop(out):
    if (out/'selection.json').exists():raise ValueError('Selection already frozen')
    fs=read(out/'development-features.json');cases=read(out/'cohorts.json')['development'];best=select(cases,fs);control=select(cases,fs,True);folds=[]
    for tr,te in KFold(5,shuffle=True,random_state=SEED).split(cases):
        train=[cases[i] for i in tr];test=[cases[i] for i in te];tf={k:[r[i] for i in tr] for k,r in fs.items()};vf={k:[r[i] for i in te] for k,r in fs.items()}
        a=select(train,tf);b=select(train,tf,True);target,_=targets(test);delta=tied(scores(vf,a),target)-tied(scores(vf,b),target)
        folds.append(dict(cases=len(test),selected=a,strongestC=b,difference=float(delta.mean())))
    method_completion={};target,_=targets(cases)
    for name in ['dense-conditional','char-conditional','hybrid-conditional','density']:
        cfg=max([k for k in fs if json.loads(k)['mode']==name],key=lambda k:float(tied(np.array([r['uncovered'] for r in fs[k]]),target).mean()))
        method_completion[name]=cfg
    choice=dict(createdAt=datetime.now(timezone.utc).isoformat(),proposed=best,strongestC=control,folds=folds,completion=method_completion,protocolSha256=sha(out/'protocol.json'),developmentFeaturesSha256=sha(out/'development-features.json'),retrieverLockSha256=sha(out/'development-features-lock.json'),analysisCodeSha256=sha(__file__),developmentCases=len(cases),finalFeaturesOpened=False)
    write(out/'selection.json',choice);(out/'frozen-analysis.py').write_bytes(Path(__file__).read_bytes());print(json.dumps(choice,ensure_ascii=False),flush=True)

def evaluate(out):
    if (out/'evaluation-report.json').exists():raise ValueError('Final evaluation already reported')
    selected=read(out/'selection.json')
    if selected['analysisCodeSha256']!=sha(__file__):raise ValueError('Analysis code changed after selection')
    if selected['protocolSha256']!=sha(out/'protocol.json'):raise ValueError('Protocol changed after selection')
    fs=read(out/'evaluation-features.json');cases=read(out/'cohorts.json')['evaluation'];target,total=targets(cases);p=selected['proposed'];controls=['same_engine_C','strongest_C','old_2_to_1','random']
    profiles=dict(proposed=p,same_engine_C={**p,'policy':dict(alpha=1,marginal='positive-difference')},strongest_C=selected['strongestC'],old_2_to_1={**p,'policy':dict(alpha=2/3,marginal='positive-difference')},random=None)
    rows={};rates={};comparisons={}
    for name,profile in profiles.items():
        s=np.zeros_like(target) if profile is None else scores(fs,profile);gain=tied(s,target);support=tied(s,total);rows[name]=dict(gain=gain.tolist(),support=support.tolist(),scores=s.tolist());rates[name]=summary(gain)
    for name in controls:comparisons[name]=summary(np.array(rows['proposed']['gain'])-rows[name]['gain'])
    previous=0
    for i,name in enumerate(sorted(controls,key=lambda n:comparisons[n]['pTwoSided'])):
        previous=max(previous,min(1,(len(controls)-i)*comparisons[name]['pTwoSided']));comparisons[name]['pHolm']=previous
    completion={}
    for name,cfg in selected['completion'].items():
        s=np.array([r['uncovered'] for r in fs[cfg]]);v=tied(s,target);completion[name]=dict(configuration=cfg,rate=summary(v),aboveChance=summary(v-.5))
    all_used=set();disjoint=True
    for c in cases:
        ids=set(c['contexts']+c['candidates']+c['owned']);disjoint &=not bool(ids&all_used);all_used.update(ids)
    guard=summary(np.array(rows['proposed']['support'])-rows['same_engine_C']['support'])
    passed=all(comparisons[n]['ci95'][0]>0 and comparisons[n]['pHolm']<.05 for n in controls) and sum(f['difference']>0 for f in selected['folds'])>=4 and len(cases)>=30 and any(r['aboveChance']['ci95'][0]>0 for r in completion.values()) and guard['ci95'][0]>-.02
    report=dict(createdAt=datetime.now(timezone.utc).isoformat(),protocolSha256=sha(out/'protocol.json'),selectionSha256=sha(out/'selection.json'),evaluationFeaturesSha256=sha(out/'evaluation-features.json'),cases=len(cases),garmentDisjointScenarios=bool(disjoint),selected=p,rates=rates,comparisons=comparisons,completion=completion,totalObservedSupportDifference=guard,developmentPositiveFolds=sum(f['difference']>0 for f in selected['folds']),criterionMet=bool(passed),productionApproved=False)
    write(out/'evaluation-rows.json',rows);write(out/'evaluation-report.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['develop','evaluate']);parser.add_argument('--output',type=Path,default=OUT);args=parser.parse_args();globals()[args.stage](args.output)
