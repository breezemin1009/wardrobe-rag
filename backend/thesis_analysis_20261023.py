"""Consolidation, leakage audit, stronger old baselines and mathematical checks."""
from pathlib import Path
from collections import Counter
from datetime import datetime,timezone
import json,re
import numpy as np
from backend.thesis_study_20261023 import ROOT,PARENT,OUT as DIAGNOSTIC,SEED,read,write,sha,source
from backend.thesis_crossfit_20261023 import OUT
from backend.thesis_evaluate_20261023 import interval

def holm(diff):
    prev=0
    for j,m in enumerate(sorted(diff,key=lambda k:diff[k]['p'])):
        prev=max(prev,min(1,(len(diff)-j)*diff[m]['p']));diff[m]['pHolm']=prev

def proofs():
    rng=np.random.default_rng(SEED);maximum_error=0.;checks=Counter()
    for _ in range(10000):
        q=int(rng.integers(1,33));w=int(rng.integers(0,9));s=rng.random((w+3,q))
        def f(indices):return float(s[indices].max(0).mean()) if len(indices) else 0.
        a=list(range(w));b=a+[w];c=w+2
        prior=s[a].max(0) if a else np.zeros(q)
        m=float(np.maximum(s[c]-prior,0).mean());delta=f(a+[c])-f(a)
        maximum_error=max(maximum_error,abs(m-delta));assert abs(m-delta)<1e-12
        assert f(b+[c])-f(b)<=m+1e-12
        assert 0<=m<=s[c].mean()+1e-12
        assert abs((f(a+a) if a else 0)-f(a))<1e-12
        checks['exactMarginal']+=1;checks['diminishingReturns']+=1;checks['bounded']+=1;checks['duplicateInvariance']+=1
    return dict(randomMatrices=10000,checks=dict(checks),maximumAbsoluteError=maximum_error,
        scope='Implementation checks complement the algebraic proof; not recommendation accuracy or a new submodular objective.')

def old_baselines():
    from backend.pog_refined_analysis import targets,scores,tied
    cases=read(PARENT/'cohorts.json')['evaluation'];fs=read(PARENT/'evaluation-features.json')
    lock=read(DIAGNOSTIC/'protocol.json');target,_=targets(cases)
    profile={'proposed':lock['proposed'],'strongest-C':lock['strongestC'],**lock['alternatives']}
    rows={m:tied(scores(fs,p),target) for m,p in profile.items()}
    rates={m:interval(a) for m,a in rows.items()};d={m:interval(rows['proposed']-a) for m,a in rows.items() if m!='proposed'};holm(d)
    return dict(n=len(cases),scope='Retrospective stronger-baseline analysis of the old 511 cases; all alternative configurations chosen using old development only.',
        rates=rates,comparisons=d)

def audit():
    it,edges,g,rev,reference,exposed=source();alltest=set();folds=[]
    normalize=lambda x:re.sub(r'\s+','',x).casefold()
    from backend.pog_retrieval import text
    for fold in range(3):
        dest=OUT/f'fold-{fold}';blocks=read(dest/'blocks.json');refs=read(dest/'reference.json')
        test={p for c in blocks for p in c['contexts']+c['owned']+c['candidates']}
        refids={r[s] for r in refs for s in ['top','bottom']}
        aliases=lambda ids:{a for p in ids for a in it[p]['aliases']}
        families=lambda ids:{it[p]['family'] for p in ids}
        titles=lambda ids:{normalize(it[p]['title']) for p in ids}
        urls=lambda ids:{it[p]['imageUrl'].split('?')[0].split('://')[-1] for p in ids}
        checks=dict(garmentOverlap=len(test&refids),aliasOverlap=len(aliases(test)&aliases(refids)),
            familyOverlap=len(families(test)&families(refids)),titleOverlap=len(titles(test)&titles(refids)),
            photoUrlOverlap=len(urls(test)&urls(refids)),testAcrossFoldsOverlap=len(test&alltest))
        assert not any(checks.values()),checks
        featuretexts={text(it[p]) for p in refids}
        collisions=sum(text(it[p]) in featuretexts for p in test)
        p=read(dest/'protocol.json');assert sha(dest/'blocks.json')==p['blocksSha256']
        for n,s in p['files'].items():assert sha(dest/n)==s
        e=read(dest/'evaluation-lock.json');assert e['protocolSha256']==sha(dest/'protocol.json')
        assert e['codeSha256']==sha(dest/'frozen-evaluator.py')
        alltest.update(test);folds.append(dict(fold=fold,testGarments=len(test),referencePairs=len(refs),checks=checks,
            strippedEncoderTextCollisions=collisions,
            collisionInterpretation='Same cleaned wording can occur for different original products; not exact product identity. No outcome labels enter text features.'))
    return dict(folds=folds,distinctTestGarments=len(alltest),allIdentityChecksPassed=True)

def main():
    if (OUT/'report.json').exists():raise RuntimeError('Immutable report exists')
    reports={};foldsummary=[]
    for variant in read(OUT/'fold-0/protocol.json')['variants']:
        groups=[read(OUT/f'fold-{f}/predictions.json')[variant] for f in range(3)]
        rows=sum(groups,[]);methods=list(rows[0]['metrics']);rates={};diff={}
        # Conditional, fold-stratified bootstrap preserves each reference split's
        # sample count. The three trained references are not independent trials.
        def stratified(arrays):
            a=np.concatenate(arrays);result=interval(a);rng=np.random.default_rng(SEED)
            boots=np.zeros(6000)
            for v in arrays:
                v=np.asarray(v);boots+=v[rng.integers(0,len(v),(6000,len(v)))].sum(1)
            result['ci95']=np.quantile(boots/len(a),[.025,.975]).tolist();return result
        for m in methods:
            rates[m]={k:stratified([[r['metrics'][m][k] for r in group] for group in groups]) for k in ['normalizedGain','hit','gain','support','regret','ndcg']}
            rates[m]['ties']=sum(r['metrics'][m]['tie']>0 for r in rows)
            if m!='proposed':diff[m]=stratified([[r['metrics']['proposed']['normalizedGain']-r['metrics'][m]['normalizedGain'] for r in group] for group in groups])
        holm(diff);strata={}
        for kind in sorted({r['category'] for r in rows}):
            group=[r for r in rows if r['category']==kind]
            strata[kind]=dict(n=len(group),proposed=interval([r['metrics']['proposed']['normalizedGain'] for r in group]),
                deltaC=interval([r['metrics']['proposed']['normalizedGain']-r['metrics']['same-C']['normalizedGain'] for r in group]))
        reports[variant]=dict(n=len(rows),rates=rates,comparisons=diff,strata=strata)
        if variant=='primary':
            for i,group in enumerate(groups):
                foldsummary.append(dict(fold=i,n=len(group),proposed=float(np.mean([r['metrics']['proposed']['normalizedGain'] for r in group])),
                    sameC=float(np.mean([r['metrics']['same-C']['normalizedGain'] for r in group])),
                    delta=float(np.mean([r['metrics']['proposed']['normalizedGain']-r['metrics']['same-C']['normalizedGain'] for r in group]))))
    result=dict(createdAt=datetime.now(timezone.utc).isoformat(),results=reports,folds=foldsummary,
        audit=audit(),mathematicalChecks=proofs(),original511StrongerBaselines=old_baselines(),
        provenance={str(f):{n:sha(OUT/f'fold-{f}'/n) for n in ['protocol.json','blocks.json','predictions.json','report.json']} for f in range(3)})
    write(OUT/'report.json',result)
    print(json.dumps(dict(primary={m:r['normalizedGain'] for m,r in result['results']['primary']['rates'].items()},
        comparisons=result['results']['primary']['comparisons'],folds=foldsummary,old=result['original511StrongerBaselines']),ensure_ascii=False),flush=True)

if __name__=='__main__':main()
