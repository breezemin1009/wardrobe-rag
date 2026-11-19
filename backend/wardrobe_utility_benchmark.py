"""Observed-outfit marginal-coverage experiment, paired by wardrobe scenario.

Same candidate/context descriptions and case similarities are reused by C and M.
C is mean combination similarity; M is the facility-location marginal gain.
Reference utility counts additional recorded co-outfit links over the owned set.
Repeated variants of one anchor are aggregated before uncertainty estimation.
"""
from pathlib import Path
from collections import defaultdict,Counter
import argparse,csv,hashlib,itertools,json,random
from datetime import datetime,timezone
import numpy as np
from backend.case_benchmark import TOP,BOTTOM,product_id,text,load_data
from backend.case_recommendation import CaseRecommender

SEED=20261019


def build_scenarios(outfits,split):
    items={};graphs=defaultdict(lambda:defaultdict(set));sourceids=defaultdict(set)
    for outfit in outfits:
        tops=[i for i in outfit['items'] if i['categoryid'] in TOP and text(i)]
        bottoms=[i for i in outfit['items'] if i['categoryid'] in BOTTOM and text(i)]
        for item in tops+bottoms:items[product_id(item)]=item
        for t in tops:
            for b in bottoms:
                ti,bi=product_id(t),product_id(b)
                graphs[('bottom',b['categoryid'])][ti].add(bi)
                graphs[('top',t['categoryid'])][bi].add(ti)
                sourceids[(ti,bi)].add(outfit['set_id'])
    rng=random.Random(SEED+(11 if split=='valid' else 12))
    scenarios=[];exclusions=Counter();anchor_count=0
    for (side,category),graph in sorted(graphs.items()):
        targets=sorted(set().union(*graph.values()))
        for context,linked in sorted(graph.items()):
            if len(linked)<2:continue
            anchor_count+=1
            # Every ordered existing/covered-candidate pair is included.
            for existing,covered in itertools.permutations(sorted(linked),2):
                if text(items[existing])==text(items[covered]):
                    exclusions['identical_existing_candidate_description']+=1;continue
                forbidden={existing,covered};contexts=[context];new_candidates=[]
                other_contexts=sorted(graph);rng.shuffle(other_contexts)
                for g in other_contexts:
                    if g==context or text(items[g]) in {text(items[x]) for x in contexts}:continue
                    # These two other contexts supply recorded new links over the owned alternative.
                    if existing in graph[g]:continue
                    options=[c for c in sorted(graph[g]) if c not in forbidden and text(items[c]) not in
                             {text(items[x]) for x in forbidden}]
                    if not options:continue
                    candidate=rng.choice(options);contexts.append(g);new_candidates.append(candidate)
                    forbidden.add(candidate)
                    if len(contexts)==3:break
                if len(contexts)<3:
                    exclusions['insufficient_distinct_contexts']+=1;continue
                observed=set().union(*(graph[g] for g in contexts))
                distractors=[c for c in targets if c not in observed and c not in forbidden and
                             text(items[c]) not in {text(items[x]) for x in forbidden}]
                if not distractors:
                    exclusions['no_unobserved_alternative']+=1;continue
                distractor=rng.choice(distractors)
                choices=[covered,*new_candidates,distractor];rng.shuffle(choices)
                # One statistical anchor groups all category/ordered-pair variants of this context product.
                anchor=split+'-'+side+'-'+context
                reference=[[int(c in graph[g]) for g in contexts] for c in choices]
                owned_reference=[int(existing in graph[g]) for g in contexts]
                def garment(pid,kind):
                    return {'id':pid,'description':items[pid]['name'],'category':kind,'categoryId':items[pid]['categoryid']}
                tc='상의' if side=='top' else '하의';oc='하의' if side=='top' else '상의'
                refs=[]
                for g in contexts:
                    for c in choices+[existing]:
                        if c in graph[g]:
                            pair=(c,g) if side=='top' else (g,c)
                            refs.extend(sorted(sourceids[pair]))
                scenarios.append({'scenarioId':f'{split}-{side}-{category}-{context}-{existing}-{covered}',
                    'anchor':anchor,'targetCategory':category,'contexts':[garment(g,oc) for g in contexts],
                    'existing':garment(existing,tc),'candidates':[garment(c,tc) for c in choices],
                    'observedCandidateEdges':reference,'observedExistingEdges':owned_reference,
                    'sourceOutfits':sorted(set(refs))})
    return scenarios,{'eligibleCategoryAnchors':anchor_count,'scenarios':len(scenarios),
                      'statisticalAnchors':len({s['anchor'] for s in scenarios}),'exclusions':dict(exclusions)}


def ranking_values(scores,gains):
    scores=np.asarray(scores,float);gains=np.asarray(gains,float)
    top=np.isclose(scores,scores.max(),atol=1e-7,rtol=0)
    selected=float(gains[top].mean());ideal=float(gains.max())
    ordered=sorted(range(len(scores)),key=lambda i:-scores[i]);dcg=0;i=0
    while i<len(ordered):
        group=[ordered[i]];j=i+1
        while j<len(ordered) and abs(scores[ordered[j]]-scores[ordered[i]])<=1e-7:
            group.append(ordered[j]);j+=1
        dcg+=float(gains[group].mean())*sum(1/np.log2(r+2) for r in range(i,j));i=j
    ideal_dcg=sum(v/np.log2(i+2) for i,v in enumerate(sorted(gains,reverse=True)))
    return {'newLinks':selected,'regret':ideal-selected,'optimal':float((gains[top]==ideal).mean()),
            'ndcgAt4':float(dcg/ideal_dcg) if ideal_dcg else None,'topTies':int(top.sum()>1)}


def clustered_summary(rows,condition,method):
    groups=defaultdict(list)
    for r in rows:
        if r['ownedState']==condition and r['method']==method:groups[r['anchor']].append(r)
    values={key:np.asarray([np.mean([r[key] for r in cluster]) for cluster in groups.values()])
            for key in ['newLinks','regret','optimal','ndcgAt4']}
    rng=np.random.default_rng(SEED);summary={}
    for key,a in values.items():
        boot=np.asarray([a[rng.integers(len(a),size=len(a))].mean() for _ in range(5000)])
        summary[key]={'mean':float(a.mean()),'sdAcrossAnchors':float(a.std(ddof=1)),
                      'ci95':np.quantile(boot,[.025,.975]).tolist()}
    return {'scenarios':sum(len(v) for v in groups.values()),'anchors':len(groups),'metrics':summary}


def run(archive,corpus,output):
    output.mkdir(exist_ok=True,parents=True)
    source=load_data(archive)
    datasets={};audit={}
    for split in ['valid','test']:datasets[split],audit[split]=build_scenarios(source[split],split)
    protocol={'version':'wardrobe-observed-utility-1','createdAt':datetime.now(timezone.utc).isoformat(),
        'seed':SEED,'archiveSha256':hashlib.sha256(archive.read_bytes()).hexdigest(),
        'retrievalCorpusSha256':hashlib.sha256(corpus.read_bytes()).hexdigest(),
        'scenarios':audit,'contexts':3,'candidates':4,'targetType':'same fine category',
        'factorMethods':['combination','marginal'],'factorOwnedState':['empty','present'],
        'reference':'number of recorded context-candidate links not already linked to the owned alternative',
        'unobserved':'0 counts recorded links; it is not an incompatibility label',
        'construction':'all anchors with >=2 recorded alternatives; all ordered existing/covered choices; '
        'two additional distinct contexts with new recorded candidates and one alternative unobserved in these contexts',
        'fixed':'same descriptions, candidate/context pool, k=10 lexical case retrieval for all four conditions',
        'primary':'present-owned marginal minus combination in recorded new links of top recommendation',
        'analysis':'average variants per anchor; macro mean over anchor product; cluster bootstrap 5000; sign permutation 20000',
        'control':'empty owned set makes combination and marginal numerically equal',
        'fitbRelation':'uses the existing held-out outfit splits; this is a different task derived from the same public material'}
    (output/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8')
    (output/'scenarios.json').write_text(json.dumps(datasets,ensure_ascii=False),encoding='utf8')
    print(json.dumps({'scenarioAudit':audit}),flush=True)
    model=CaseRecommender.load(corpus);cache={};rows=[];details=[]
    def sim(c,g):
        key=(c['categoryId'],c['description'],g['description'])
        if key not in cache:cache[key]=model.pair(c,g)['similarity']
        return cache[key]
    for scenario in datasets['test']:
        matrix=np.asarray([[sim(c,g) for g in scenario['contexts']] for c in scenario['candidates']])
        existing=np.asarray([sim(scenario['existing'],g) for g in scenario['contexts']])
        observed=np.asarray(scenario['observedCandidateEdges'])
        actual_existing=np.asarray(scenario['observedExistingEdges'])
        for condition in ['empty','present']:
            best=np.zeros(3) if condition=='empty' else existing
            observed_best=np.zeros(3) if condition=='empty' else actual_existing
            reference=np.maximum(observed-observed_best,0).sum(axis=1)
            scores={'combination':matrix.mean(axis=1),'marginal':np.maximum(matrix-best,0).mean(axis=1)}
            if condition=='empty':np.testing.assert_allclose(scores['combination'],scores['marginal'],rtol=0,atol=1e-12)
            for method,rank in scores.items():
                result=ranking_values(rank,reference)
                rows.append({'scenarioId':scenario['scenarioId'],'anchor':scenario['anchor'],'ownedState':condition,
                    'method':method,**result,**{f'score{i}':float(v) for i,v in enumerate(rank)},
                    **{f'reference{i}':int(v) for i,v in enumerate(reference)}})
        details.append({'scenarioId':scenario['scenarioId'],'anchor':scenario['anchor'],'similarityMatrix':matrix.tolist(),
                        'existingSimilarities':existing.tolist()})
    summary={condition:{method:clustered_summary(rows,condition,method) for method in ['combination','marginal']}
             for condition in ['empty','present']}
    by_anchor=defaultdict(lambda:defaultdict(list))
    for r in rows:
        if r['ownedState']=='present':by_anchor[r['anchor']][r['method']].append(r['newLinks'])
    diffs=np.asarray([np.mean(v['marginal'])-np.mean(v['combination']) for v in by_anchor.values()])
    rng=np.random.default_rng(SEED)
    boot=np.asarray([diffs[rng.integers(len(diffs),size=len(diffs))].mean() for _ in range(5000)])
    p=(1+sum(abs((diffs*rng.choice([-1,1],len(diffs))).mean())>=abs(diffs.mean())-1e-12 for _ in range(20000)))/20001
    paired={'anchors':len(diffs),'meanDifference':float(diffs.mean()),'ci95':np.quantile(boot,[.025,.975]).tolist(),
            'p':p,'betterAnchors':int((diffs>1e-9).sum()),'worseAnchors':int((diffs<-1e-9).sum()),
            'sameAnchors':int((abs(diffs)<=1e-9).sum())}
    report={'protocol':protocol,'summary':summary,'primaryPaired':paired,
            'codeSha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    (output/'case-details.json').write_text(json.dumps(details,ensure_ascii=False),encoding='utf8')
    with (output/'case-results.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps({'summary':summary,'primaryPaired':paired}),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--archive',type=Path,required=True);p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();run(a.archive,a.corpus,a.output)
