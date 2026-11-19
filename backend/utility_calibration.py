"""Development-only policy selection, followed by a locked held-out evaluation.

Run prepare -> select -> evaluate. Original v1 experiment files are immutable.
The new partition uses the original archive's train outfits only; its evaluation
records never enter the new retrieval corpus, vocabulary or parameter selection.
This is a new internal resplit, not an external replication dataset.
"""
from pathlib import Path
from collections import defaultdict
from datetime import datetime,timezone
import argparse,csv,hashlib,json
import numpy as np
from sklearn.model_selection import GroupKFold
from backend.case_benchmark import load_data,TOP,BOTTOM,product_id,text
from backend.case_recommendation import CaseRecommender
from backend.wardrobe_utility_benchmark import build_scenarios
from backend.utility_policy import FEATURE_SCHEMA,UtilityPolicy,utility_features,policy_scores,tie_aware_metrics

SEED=20261020
ALPHAS=sorted(set([i/10 for i in range(11)]+[2/3]))
FLOORS=[0,.75,.9,1]
MODES=['joint','aligned']
KS=[1,5,10]
COMPATIBILITY_MARGIN=.02

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
def read(path):return json.loads(Path(path).read_text(encoding='utf8'))
def inventory(outfits):
    return {(product_id(i),text(i)) for o in outfits for i in o['items'] if i['categoryid'] in TOP|BOTTOM}
def purge(outfits,blocked):
    pids={p for p,_ in blocked};names={n for _,n in blocked}
    return [{**o,'items':[i for i in o['items'] if product_id(i) not in pids and text(i) not in names]} for o in outfits]
def pairs(outfits):
    records={}
    for o in outfits:
        for t in [i for i in o['items'] if i['categoryid'] in TOP and text(i)]:
            for b in [i for i in o['items'] if i['categoryid'] in BOTTOM and text(i)]:
                key=(product_id(t),product_id(b))
                records.setdefault(key,{'sourceId':o['set_id'],'topProduct':key[0],'bottomProduct':key[1],
                    'top':text(t),'bottom':text(b),'topCategory':t['categoryid'],'bottomCategory':b['categoryid']})
    return [records[k] for k in sorted(records)]

def prepare(archive,out):
    if (out/'protocol.json').exists():raise ValueError('Use a new output directory; a frozen run cannot be overwritten')
    out.mkdir(parents=True,exist_ok=True)
    partition=defaultdict(list)
    for o in load_data(archive)['train']:
        bucket=int(hashlib.sha256(('utility-v2:'+o['set_id']).encode()).hexdigest()[:8],16)%100
        partition['retrieval' if bucket<50 else 'selection' if bucket<75 else 'evaluation'].append(o)
    partition['evaluation']=purge(partition['evaluation'],inventory(partition['selection']))
    partition['retrieval']=purge(partition['retrieval'],inventory(partition['selection'])|inventory(partition['evaluation']))
    inventories={k:inventory(v) for k,v in partition.items()}
    leakage={}
    for a,b in [('retrieval','selection'),('retrieval','evaluation'),('selection','evaluation')]:
        leakage[a+'-'+b]={'products':len({p for p,_ in inventories[a]}&{p for p,_ in inventories[b]}),
                         'descriptions':len({n for _,n in inventories[a]}&{n for _,n in inventories[b]})}
    assert all(v['products']==v['descriptions']==0 for v in leakage.values())
    datasets={};audit={}
    for name,split in [('selection','valid'),('evaluation','test')]:
        datasets[name],audit[name]=build_scenarios(partition[name],split)
        write(out/(name+'-scenarios.json'),datasets[name])
    records=pairs(partition['retrieval'])
    (out/'retrieval-cases.jsonl').write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in records)+'\n',encoding='utf8')
    protocol={'version':'utility-calibration-v2','featureSchema':FEATURE_SCHEMA,'seed':SEED,
        'createdAt':datetime.now(timezone.utc).isoformat(),'archiveSha256':digest(archive),
        'partition':'original train outfits only; hash outfit 50/25/25; purge evaluation against selection, retrieval against both by product ID AND canonical description',
        'externalReplication':False,'selectionAlreadyInspected':False,'evaluationAlreadyInspected':False,
        'scope':'controlled recovery of recorded additional co-outfit links; not aesthetic judgement or purchase satisfaction',
        'unobserved':'absence of a recorded relation, not a negative compatibility label',
        'retrievalCases':len(records),'scenarios':audit,'leakage':leakage,
        'retrievalGrid':{'mode':MODES,'k':KS},'alphaGrid':ALPHAS,'relativeCombinationFloorGrid':FLOORS,
        'selection':'choose retrieval by selection-only combination newLinks; then maximize newLinks among policies respecting total-recorded-links guard; ties prefer totalLinks, simpler ungated policy and larger alpha',
        'compatibilityMargin':COMPATIBILITY_MARGIN,'primary':'locked proposed policy minus combination-only with exactly the SAME selected retrieval, anchor-macro mean recorded newLinks at top1',
        'secondary':'fixed2:1, tuned weight only, pure marginal and old joint retrieval shown descriptively',
        'claimRule':'alpha<1 AND primary difference bootstrap95 lower>0 AND normalized totalLinks difference95 lower> -compatibilityMargin',
        'uncertainty':'anchor cluster bootstrap5000; paired sign permutation20000; 5-fold grouped internal selection stability',
        'applicationBoundary':'paired-description policy; no transfer to style-annotation mobile score without feature-matched labels',
        'files':{n:digest(out/n) for n in ['retrieval-cases.jsonl','selection-scenarios.json','evaluation-scenarios.json']}}
    write(out/'protocol.json',protocol)
    print(json.dumps({'retrievalCases':len(records),'scenarios':audit,'leakage':leakage}),flush=True)

def verify(out):
    protocol=read(out/'protocol.json')
    for name,sha in protocol['files'].items():
        if digest(out/name)!=sha:raise ValueError('Frozen dataset changed: '+name)
    return protocol

class FeatureIndex:
    def __init__(self,corpus):
        self.model=CaseRecommender.load(corpus);self.cache={}
        self.indices={}
        for side in ['상의','하의']:
            key='topCategory' if side=='상의' else 'bottomCategory'
            for category in {r[key] for r in self.model.records}:
                self.indices[(side,category)]=[i for i,r in enumerate(self.model.records) if r[key]==category]
    def pair(self,c,g):
        key=(c['category'],c['categoryId'],c['description'],g['description'])
        if key in self.cache:return self.cache[key]
        model=self.model
        top,bottom=(c,g) if c['category']=='상의' else (g,c)
        ix=self.indices.get((c['category'],c['categoryId']),list(range(len(model.records))))
        a=(model.vectorizer.transform([top['description']])@model.tops[ix].T).toarray()[0]
        b=(model.vectorizer.transform([bottom['description']])@model.bottoms[ix].T).toarray()[0]
        joint=(model.vectorizer.transform([top['description']+' '+bottom['description']])@model.matrix[ix].T).toarray()[0]
        values={'joint':joint,'aligned':np.sqrt(np.maximum(a*b,0))};result={}
        for mode,arr in values.items():
            nearest=np.sort(arr)[::-1][:10]
            for k in KS:result[mode+'-'+str(k)]=float(nearest[:k].mean())
        self.cache[key]=result
        return result
    def encode(self,scenarios):
        rows=[]
        for s in scenarios:
            matrices={mode+'-'+str(k):[] for mode in MODES for k in KS}
            for c in [*s['candidates'],s['existing']]:
                values=[self.pair(c,g) for g in s['contexts']]
                for name in matrices:matrices[name].append([v[name] for v in values])
            rows.append({'scenarioId':s['scenarioId'],'anchor':s['anchor'],
                'features':{name:[a.tolist() for a in utility_features(np.asarray(m)[:4],np.asarray(m)[4:])]
                            for name,m in matrices.items()},
                'observed':s['observedCandidateEdges'],'existing':s['observedExistingEdges']})
        return rows

def evaluate_rows(rows,model,policy):
    result=[]
    for r in rows:
        c,m=r['features'][model]
        scores=policy_scores(c,m,policy)
        metrics=tie_aware_metrics(scores,r['observed'],r['existing'])
        result.append({'scenarioId':r['scenarioId'],'anchor':r['anchor'],**metrics})
    return result
def anchors(rows,key):
    groups=defaultdict(list)
    for r in rows:groups[r['anchor']].append(r[key])
    return {g:float(np.mean(v)) for g,v in sorted(groups.items())}
def average(rows,key):return float(np.mean(list(anchors(rows,key).values())))
def summary(rows):return {k:average(rows,k) for k in ['newLinks','totalLinks','regret','topTies']}
def pick_retrieval(rows):
    scores=[]
    for mode in MODES:
        for k in KS:
            model=mode+'-'+str(k);metrics=summary(evaluate_rows(rows,model,UtilityPolicy(1)))
            scores.append({'model':model,**metrics})
    # Use same chosen pair engine for the with/without marginal ablation.
    best=max(scores,key=lambda r:(r['newLinks'],r['totalLinks'],-int(r['model'].split('-')[1]),r['model']=='joint-10'))
    return best['model'],scores
def pick_policy(rows,model,allow_gate=True):
    baseline=average(evaluate_rows(rows,model,UtilityPolicy(1)),'totalLinks')
    candidates=[]
    for alpha in ALPHAS:
        for floor in FLOORS if allow_gate else [0]:
            policy=UtilityPolicy(alpha,floor);metrics=summary(evaluate_rows(rows,model,policy))
            candidates.append({**policy.to_dict(),**metrics,'eligible':metrics['totalLinks']/3>=baseline/3-COMPATIBILITY_MARGIN-1e-12})
    eligible=[r for r in candidates if r['eligible']]
    best=max(eligible,key=lambda r:(round(r['newLinks'],12),round(r['totalLinks'],12),r['relative_floor']==0,r['alpha'],-r['relative_floor']))
    return UtilityPolicy(best['alpha'],best['relative_floor']),candidates

def select(out):
    protocol=verify(out)
    if (out/'selection.json').exists():raise ValueError('Selection already frozen; use a new run for a new method')
    index=FeatureIndex(out/'retrieval-cases.jsonl')
    rows=index.encode(read(out/'selection-scenarios.json'));write(out/'selection-features.json',rows)
    model,retrieval_grid=pick_retrieval(rows)
    policy,grid=pick_policy(rows,model)
    weight,_=pick_policy(rows,model,False)
    old_weight,old_grid=pick_policy(rows,'joint-10',False)
    cv=[];groups=[r['anchor'] for r in rows]
    for fold,(train,test) in enumerate(GroupKFold(n_splits=5).split(rows,groups=groups)):
        a=[rows[i] for i in train];b=[rows[i] for i in test]
        engine,_=pick_retrieval(a);chosen,_=pick_policy(a,engine)
        proposed=evaluate_rows(b,engine,chosen);control=evaluate_rows(b,engine,UtilityPolicy(1))
        cv.append({'fold':fold,'model':engine,'policy':chosen.to_dict(),'heldout':summary(proposed),
                   'control':summary(control),'heldoutAnchors':len(set(r['anchor'] for r in b))})
    record={'protocolSha256':digest(out/'protocol.json'),'codeSha256':digest(__file__),
            'featureSchema':FEATURE_SCHEMA,'selectedRetrieval':model,'proposedPolicy':policy.to_dict(),
            'tunedWeightOnly':weight.to_dict(),'tunedOldWeight':old_weight.to_dict(),
            'retrievalGrid':retrieval_grid,'policyGrid':grid,'oldWeightGrid':old_grid,'selectionStability':cv,
            'selectionSummary':summary(evaluate_rows(rows,model,policy)),
            'selectionFeaturesSha256':digest(out/'selection-features.json')}
    write(out/'selection.json',record)
    print(json.dumps({k:record[k] for k in ['selectedRetrieval','proposedPolicy','tunedWeightOnly','tunedOldWeight','selectionSummary']}),flush=True)

def paired_difference(first,second,key,normalizer=1):
    a,b=anchors(first,key),anchors(second,key)
    if set(a)!=set(b):raise ValueError('Unpaired anchors')
    d=np.array([(a[k]-b[k])/normalizer for k in a]);rng=np.random.default_rng(SEED)
    boots=d[rng.integers(len(d),size=(5000,len(d)))].mean(axis=1)
    permutations=(rng.choice([-1,1],size=(20000,len(d)))*d).mean(axis=1)
    return {'anchors':len(d),'difference':float(d.mean()),'ci95':np.quantile(boots,[.025,.975]).tolist(),
            'pTwoSided':float((1+(abs(permutations)>=abs(d.mean())-1e-12).sum())/20001),
            'better':int((d>1e-9).sum()),'worse':int((d< -1e-9).sum()),'same':int((abs(d)<=1e-9).sum())}

def evaluate(out):
    protocol=verify(out);selection=read(out/'selection.json')
    if selection['protocolSha256']!=digest(out/'protocol.json') or selection['codeSha256']!=digest(__file__):
        raise ValueError('Protocol/code changed after selection')
    if (out/'evaluation-report.json').exists():raise ValueError('Held-out evaluation already recorded; do not retune on its results')
    def policy(v):return UtilityPolicy(v['alpha'],v['relative_floor'])
    index=FeatureIndex(out/'retrieval-cases.jsonl');rows=index.encode(read(out/'evaluation-scenarios.json'))
    model=selection['selectedRetrieval']
    methods={'old_combination':('joint-10',UtilityPolicy(1)),
             'old_fixed_2_to_1':('joint-10',UtilityPolicy(2/3)),
             'old_tuned_weight':('joint-10',policy(selection['tunedOldWeight'])),
             'combination':(model,UtilityPolicy(1)),
             'marginal':(model,UtilityPolicy(0)),
             'fixed_2_to_1':(model,UtilityPolicy(2/3)),
             'tuned_weight_only':(model,policy(selection['tunedWeightOnly'])),
             'proposed':(model,policy(selection['proposedPolicy']))}
    results={name:evaluate_rows(rows,engine,p) for name,(engine,p) in methods.items()}
    primary=paired_difference(results['proposed'],results['combination'],'newLinks')
    compatibility=paired_difference(results['proposed'],results['combination'],'totalLinks',3)
    controls=[]
    for row in rows:
        c=np.asarray(row['features'][model][0])
        scores=policy_scores(c,c,policy(selection['proposedPolicy']))
        controls.append(set(np.flatnonzero(np.isclose(scores,scores.max(),atol=1e-7,rtol=0)))==set(np.flatnonzero(np.isclose(c,c.max(),atol=1e-7,rtol=0))))
    assert all(controls),'Empty wardrobe control failed'
    change=sum(a['topIndices']!=b['topIndices'] for a,b in zip(results['proposed'],results['combination']))
    report={'protocol':protocol,'selectionSha256':digest(out/'selection.json'),'selectedRetrieval':model,
        'proposedPolicy':selection['proposedPolicy'],'scenarios':len(rows),'anchors':len(set(r['anchor'] for r in rows)),
        'summary':{k:summary(v) for k,v in results.items()},'primary':primary,'compatibilityGuard':compatibility,
        'superioritySupported':selection['proposedPolicy']['alpha']<1 and primary['ci95'][0]>0 and compatibility['ci95'][0]> -COMPATIBILITY_MARGIN,
        'changedTopSets':change,'emptyWardrobeControlPassed':len(controls),
        'secondaryDescriptive':{k:paired_difference(results['proposed'],results[k],'newLinks') for k in ['old_fixed_2_to_1','old_combination','marginal','tuned_weight_only']},
        'applicationBoundary':protocol['applicationBoundary']}
    write(out/'evaluation-features.json',rows);write(out/'evaluation-results.json',results);write(out/'evaluation-report.json',report)
    print(json.dumps({k:report[k] for k in ['selectedRetrieval','proposedPolicy','scenarios','anchors','summary','primary','compatibilityGuard','superioritySupported','changedTopSets']}),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','select','evaluate']);p.add_argument('--archive',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.stage=='prepare':prepare(args.archive,args.output)
    elif args.stage=='select':select(args.output)
    else:evaluate(args.output)
