"""Locked extension evaluation, all methods and sizes, no post-test tuning."""
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone
import argparse, json, time, math
import numpy as np
from backend.thesis_study_20261023 import OUT, PARENT, SEED, read, write, sha, source

def values(matrix, case, spec, vectors=None, lex=None, ix=None):
    nq,nw,nc=spec['q'],spec['w'],spec['n']
    cm=matrix[:nc,:nq]; wm=matrix[16:16+nw,:nq]
    prior=wm.max(0) if nw else np.zeros(nq)
    c=cm.mean(1);m=np.maximum(cm-prior,0).mean(1)
    result={'C':c,'positive-difference':m,'product-residual':(cm*np.prod(1-wm,axis=0)).mean(1)}
    if vectors is not None:
        ci=[ix[p] for p in case['candidates'][:nc]];oi=[ix[p] for p in case['owned'][:nw]]
        d=np.clip((vectors[ci]@vectors[oi].T).max(1),0,1) if nw else np.zeros(nc)
        l=np.clip((lex[ci]@lex[oi].T).toarray().max(1),0,1) if nw else np.zeros(nc)
        result['char-novelty']=c*(1-l);result['dense-novelty']=c*(1-d)
    return result

def metric(scores, gains, total):
    s,g,t=map(lambda a:np.asarray(a,float),(scores,gains,total))
    top=np.isclose(s,s.max(),atol=1e-10,rtol=0);best=g.max()
    if best<=0: return None
    selected=float(g[top].mean())
    # Average DCG discounts within score ties, rather than imposing item-ID order.
    order=np.argsort(-s,kind='stable');dcg=0.;start=0
    while start<len(s):
        end=start+1
        while end<len(s) and abs(s[order[end]]-s[order[start]])<=1e-10:end+=1
        dcg+=float(g[order[start:end]].mean()*np.sum(1/np.log2(np.arange(start,end)+2)));start=end
    ideal=float(np.sum(np.sort(g)[::-1]/np.log2(np.arange(len(g))+2)))
    return dict(normalizedGain=selected/best,hit=float((g[top]==best).mean()),gain=selected,
                support=float(t[top].mean()),regret=float(best-selected),ndcg=dcg/ideal,tie=int(top.sum()>1))

def interval(a, seed=SEED):
    a=np.asarray(a,float);rng=np.random.default_rng(seed)
    boots=np.concatenate([a[rng.integers(0,len(a),(500,len(a)))].mean(1) for _ in range(12)])
    null=np.concatenate([(a*rng.choice([-1,1],(1000,len(a)))).mean(1) for _ in range(20)])
    return dict(n=len(a),mean=float(a.mean()),ci95=np.quantile(boots,[.025,.975]).tolist(),
                p=float((np.sum(np.abs(null)>=abs(a.mean())-1e-12)+1)/(len(null)+1)))

def check_metrics():
    assert metric([1,1],[2,0],[3,1])['normalizedGain']==.5
    assert metric([1,0],[2,0],[3,1])['ndcg']==1
    assert metric([1,1],[0,0],[1,1]) is None
    m=np.array([[.8,.2],[.3,.9]]+[[0,0]]*14+[[.7,.4],[.1,.5],[.5,.5],[.2,.3]])
    c={'candidates':list(range(16)),'owned':list(range(16,20))}
    empty=values(m,c,dict(q=2,w=0,n=2));assert np.allclose(empty['C'],empty['positive-difference'])
    one=values(m,c,dict(q=2,w=1,n=2));two=values(m,c,dict(q=2,w=2,n=2));assert np.all(two['positive-difference']<=one['positive-difference'])

def predict():
    from backend.pog_refined_retrieval import Retriever
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=1);check_metrics()
    if (OUT/'predictions.json').exists():raise RuntimeError('Predictions already frozen')
    lock=read(OUT/'protocol.json');blocks=read(OUT/'blocks.json')
    assert sha(OUT/'blocks.json')==lock['blocksSha256']
    for n,s in lock['parentFiles'].items():assert sha(PARENT/n)==s,n
    write(OUT/'evaluation-lock.json',dict(createdAt=datetime.now(timezone.utc).isoformat(),
        protocolSha256=sha(OUT/'protocol.json'),codeSha256=sha(__file__),predictionsOpened=False))
    (OUT/'frozen-evaluator.py').write_bytes(Path(__file__).read_bytes())
    engine=Retriever(OUT if (OUT/'reference.json').exists() else PARENT)
    profiles={'proposed':lock['proposed'],'strongest-C':lock['strongestC'],**lock['alternatives']}
    configs={p['configuration'] for p in profiles.values()}
    configs.add(json.dumps(dict(k=96,mode='hybrid-conditional'),sort_keys=True,separators=(',',':')))
    matrices={};timings=[];started=time.monotonic()
    for cfg in sorted(configs):
        rows=[]
        for i,c in enumerate(blocks):
            tick=time.monotonic();rows.append(engine.matrix(c['contexts'],c['candidates']+c['owned'],json.loads(cfg)).tolist());timings.append(time.monotonic()-tick)
        matrices[cfg]=rows
        print('matrix',cfg,'blocks',len(rows),'elapsed',round(time.monotonic()-started,1),flush=True)
    null=[];original={k:v['b'].copy() for k,v in engine.pools.items()}
    for seed in range(3):
        rng=np.random.default_rng(SEED+seed)
        for k,p in engine.pools.items():
            p['b']=rng.permutation(original[k]);assert np.array_equal(np.sort(p['b']),np.sort(original[k]))
        engine.cache.clear();rows=[]
        for c in blocks:rows.append(engine.matrix(c['contexts'],c['candidates']+c['owned'],dict(k=96,mode='hybrid-density')).tolist())
        null.append(rows);print('shuffled reference',seed,'elapsed',round(time.monotonic()-started,1),flush=True)
    result={};distributions={}
    for name,spec in lock['variants'].items():
        rows=[];no_gain=0;counts=Counter();shape=Counter()
        for i,c in enumerate(blocks):
            obs=np.asarray(c['observed'])[:spec['n'],:spec['q']]
            hist=np.asarray(c['history'])[:spec['w'],:spec['q']]
            known=hist.max(0) if spec['w'] else np.zeros(spec['q'])
            gains=(obs*(1-known)).sum(1);total=obs.sum(1)
            if gains.max()==0:no_gain+=1;continue
            fs={cfg:values(np.asarray(matrices[cfg][i]),c,spec,engine.vectors,engine.lex,engine.ix) for cfg in matrices}
            f=fs[lock['proposed']['configuration']]
            if spec['w']==0:assert np.allclose(f['C'],f['positive-difference'])
            scores={'proposed':f['positive-difference'],'same-C':f['C'],
                    'old-2:1':(2*f['C']+f['positive-difference'])/3,'random':np.zeros(spec['n']),
                    'no-density':fs[json.dumps(dict(k=96,mode='hybrid-conditional'),sort_keys=True,separators=(',',':'))]['positive-difference']}
            for m,p in profiles.items():
                a=p['policy']['alpha'];fp=fs[p['configuration']]
                scores[m]=a*fp['C']+(1-a)*fp[p['policy']['marginal']]
            rowscores={m:metric(v,gains,total) for m,v in scores.items()}
            nullmetrics=[metric(values(np.asarray(null[j][i]),c,spec)['positive-difference'],gains,total) for j in range(3)]
            rowscores['shuffled-reference']={k:float(np.mean([m[k] for m in nullmetrics])) for k in nullmetrics[0]}
            rows.append(dict(id=c['id'],category=c['category'][1],gain=gains.tolist(),total=total.tolist(),metrics=rowscores,
                scores={k:v.tolist() for k,v in scores.items()}))
            counts.update(map(int,gains));shape[(int(total.min()),int(total.max()))]+=1
        result[name]=rows;distributions[name]=dict(evaluated=len(rows),zeroGainExcluded=no_gain,
            gainCounts={str(k):v for k,v in counts.items()},totalSupportRanges={str(k):v for k,v in shape.items()})
    write(OUT/'predictions.json',result)
    write(OUT/'prediction-audit.json',dict(distributions=distributions,elapsedSeconds=time.monotonic()-started,
        matrixSeconds=dict(median=float(np.median(timings)),p95=float(np.quantile(timings,.95))),
        protocolSha256=sha(OUT/'protocol.json'),evaluatorSha256=sha(__file__),predictionsSha256=sha(OUT/'predictions.json')))
    print(json.dumps(distributions,ensure_ascii=False),flush=True)

def analyze():
    if (OUT/'report.json').exists():raise RuntimeError('Report already frozen')
    check_metrics();pred=read(OUT/'predictions.json');report={}
    for name,rows in pred.items():
        methods=list(rows[0]['metrics']);rates={};diff={}
        for m in methods:
            rates[m]={k:interval([r['metrics'][m][k] for r in rows]) for k in ['normalizedGain','hit','gain','support','regret','ndcg']}
            rates[m]['ties']=sum(r['metrics'][m]['tie']>0 for r in rows)
            if m!='proposed':diff[m]=interval([r['metrics']['proposed']['normalizedGain']-r['metrics'][m]['normalizedGain'] for r in rows])
        prev=0
        for j,m in enumerate(sorted(diff,key=lambda k:diff[k]['p'])):
            prev=max(prev,min(1,(len(diff)-j)*diff[m]['p']));diff[m]['pHolm']=prev
        strata={}
        for cat in sorted({r['category'] for r in rows}):
            group=[r for r in rows if r['category']==cat]
            strata[cat]=dict(n=len(group),proposed=interval([r['metrics']['proposed']['normalizedGain'] for r in group]),
                deltaC=interval([r['metrics']['proposed']['normalizedGain']-r['metrics']['same-C']['normalizedGain'] for r in group]))
        report[name]=dict(blocks=len(rows),rates=rates,comparisons=diff,strata=strata)
    write(OUT/'report.json',dict(createdAt=datetime.now(timezone.utc).isoformat(),
        protocolSha256=sha(OUT/'protocol.json'),predictionsSha256=sha(OUT/'predictions.json'),results=report))
    print(json.dumps({v:{k:round(a['normalizedGain']['mean'],4) for k,a in x['rates'].items()} for v,x in report.items()},ensure_ascii=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['predict','analyze']);p.add_argument('--output',type=Path,default=OUT)
    a=p.parse_args();OUT=a.output;globals()[a.stage]()
