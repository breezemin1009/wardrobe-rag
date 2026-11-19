"""Independent adversarial audit; never imports the experiment metric helpers.

Reconstruct labels from the author ZIP, inspect construction shortcuts, trace
prior selection exposure, and recompute metrics/statistics separately.
"""
from pathlib import Path
from collections import defaultdict,Counter
from datetime import datetime,timezone
import io,json,zipfile,hashlib,itertools,math
import numpy as np
from scipy.stats import binomtest

ROOT=Path(__file__).parent/'data/menswear-expansion-v3/pog'
PARENT=ROOT/'refined-study-v3-scope-audited'
STUDY=ROOT/'thesis-crossfit-20261023'
OUT=ROOT/'adversarial-audit-20261023'
SEED=2026102302
def read(p):return json.loads(Path(p).read_text(encoding='utf8'))
def write(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf8')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def gain_metric(scores,g):
    scores=np.asarray(scores);g=np.asarray(g,float)
    best=max(scores);indexes=[j for j,s in enumerate(scores) if abs(s-best)<=1e-10]
    return sum(float(g[j]) for j in indexes)/len(indexes)/max(g)

def exact_tied_dcg(scores,g):
    # Enumerate every permutation within each tie group independently.
    order=sorted(range(len(scores)),key=lambda j:-scores[j]);groups=[]
    for j in order:
        if not groups or abs(scores[j]-scores[groups[-1][0]])>1e-10:groups.append([j])
        else:groups[-1].append(j)
    dcg=0.;offset=0
    for group in groups:
        # An independent expectation: each label occurs equally often in every
        # position of its tie group; no sorting by target within the tie.
        dcg+=sum(g[j] for j in group)/len(group)*sum(1/math.log2(k+2) for k in range(offset,offset+len(group)))
        offset+=len(group)
    return dcg/sum(v/math.log2(k+2) for k,v in enumerate(sorted(g,reverse=True)))

def paired(v):
    a=np.array(v,float);rng=np.random.default_rng(SEED)
    boots=[]
    for _ in range(6000):boots.append(float(np.mean(rng.choice(a,len(a),replace=True))))
    wins=int((a>1e-12).sum());losses=int((a< -1e-12).sum())
    return dict(n=len(a),mean=float(a.mean()),ci95=np.quantile(boots,[.025,.975]).tolist(),
        wins=wins,losses=losses,ties=len(a)-wins-losses,
        exactSignP=float(binomtest(wins,wins+losses,.5).pvalue) if wins+losses else 1.)

def main():
    if OUT.exists():raise RuntimeError('Fresh audit required')
    OUT.mkdir()
    items=read(PARENT/'items.json');excluded={r['id'] for r in read(PARENT/'protocol.json')['scopeCorrection']['excludedItems']}
    items={k:v for k,v in items.items() if k not in excluded}
    aliases={a:k for k,v in items.items() for a in v['aliases']}
    graph=defaultdict(set);edge_outfits=defaultdict(set);outfit_edges=defaultdict(set);source_count=0
    with zipfile.ZipFile(ROOT/'outfit_data.txt.zip') as z:
        with z.open('outfit_data.txt') as stream:
            for line in io.TextIOWrapper(stream,encoding='utf8'):
                if not line.strip():continue
                source_count+=1;oid,ids=line.strip().split(',',1)
                present={aliases[a] for a in ids.split(';') if a in aliases}
                tops={a for a in present if items[a]['side']=='top'};bottoms=present-tops
                for t in tops:
                    for b in bottoms:
                        graph[t].add(b);edge_outfits[(t,b)].add(oid);outfit_edges[oid].add((t,b))
    raw_edges={(t,b) for t,bs in graph.items() for b in bs}
    prior_reference={r[s] for r in read(PARENT/'reference.json') for s in ['top','bottom']}
    previous_dev={p for c in read(PARENT/'cohorts.json')['development'] for p in c['contexts']+c['owned']+c['candidates']}
    old_original=read(PARENT/'cohorts.json')['evaluation'];original_bad=0
    for c in old_original:
        o=[[int(b in graph[t]) for t in c['contexts']] for b in c['candidates']]
        e=[int(any(b in graph[t] for b in c['owned'])) for t in c['contexts']]
        original_bad+=int(o!=c['observed'] or e!=c['existing'])
    folds=[];all_rows=[];shortcut=[];label_mismatch=[];metric_errors=[];all_case_outfits={}
    for fold in range(3):
        p=STUDY/f'fold-{fold}';blocks=read(p/'blocks.json');pred={r['id']:r for r in read(p/'predictions.json')['primary']}
        ref=read(p/'reference.json');ref_outfits=set().union(*(edge_outfits[(r['top'],r['bottom'])] for r in ref))
        reference_missing=sum((r['top'],r['bottom']) not in raw_edges for r in ref)
        case_ref_outfit_overlap=0;prior_exposure=0;dev_exposure=0;all_case_ids=set();metrics=[]
        for c in blocks:
            ids=set(c['contexts']+c['owned']+c['candidates']);all_case_ids.update(ids)
            prior_exposure+=int(bool(ids&prior_reference));dev_exposure+=int(bool(ids&previous_dev))
            obs=np.array([[int(b in graph[t]) for t in c['contexts']] for b in c['candidates']])
            hist=np.array([[int(b in graph[t]) for t in c['contexts']] for b in c['owned']])
            if obs.tolist()!=c['observed'] or hist.tolist()!=c['history']:label_mismatch.append(c['id'])
            g=(obs[:8]*(1-hist[:2].max(0))).sum(1);row=pred[c['id']]
            if g.tolist()!=row['gain']:label_mismatch.append(c['id']+'-gain')
            expected={m:gain_metric(scores,g) for m,scores in row['scores'].items()}
            for m,value in expected.items():
                if abs(value-row['metrics'][m]['normalizedGain'])>1e-12:metric_errors.append((c['id'],m,'gain'))
                ndcg=exact_tied_dcg(row['scores'][m],g)
                if abs(ndcg-row['metrics'][m]['ndcg'])>1e-12:metric_errors.append((c['id'],m,'ndcg'))
            metrics.append(dict(id=c['id'],**expected));all_rows.append(metrics[-1])
            shortcut.append(dict(id=c['id'],first4=float(np.mean(g[:4])/g.max()),last4=float(np.mean(g[4:])/g.max())))
            observed_edges={(t,b) for t in c['contexts'] for b in c['candidates']+c['owned'] if b in graph[t]}
            out=set().union(*(edge_outfits[e] for e in observed_edges))
            all_case_outfits[c['id']]=out;case_ref_outfit_overlap+=int(bool(out&ref_outfits))
        folds.append(dict(fold=fold,cases=len(blocks),rawMissingReferenceEdges=reference_missing,
            caseReferenceOutfitOverlap=case_ref_outfit_overlap,priorSelectionReferenceExposureBlocks=prior_exposure,
            priorDevelopmentExposureBlocks=dev_exposure))
    # Test-source outfit clustering must be checked separately from item IDs.
    parent={c:c for c in all_case_outfits};seen={}
    def find(x):
        while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
        return x
    for c,oids in all_case_outfits.items():
        for oid in oids:
            if oid in seen:parent[find(c)]=find(seen[oid])
            else:seen[oid]=c
    clusters=Counter(find(c) for c in parent)
    comparisons={m:paired([r['proposed']-r[m] for r in all_rows]) for m in all_rows[0] if m not in ['id','proposed']}
    result=dict(createdAt=datetime.now(timezone.utc).isoformat(),rawSourceOutfits=source_count,rawReconstructedEdges=len(raw_edges),
        original511LabelMismatch=original_bad,crossfitLabelMismatch=label_mismatch,metricMismatches=metric_errors,
        folds=folds,testOutfitClusters=len(clusters),largestTestOutfitCluster=max(clusters.values()),
        independentRates={m:float(np.mean([r[m] for r in all_rows])) for m in all_rows[0] if m!='id'},
        independentComparisons=comparisons,
        constructionPositionShortcut=dict(first4Mean=float(np.mean([s['first4'] for s in shortcut])),last4Mean=float(np.mean([s['last4'] for s in shortcut])),
            scope='Diagnostic only: candidate positions 0..3 were generated from observed links, controls occupy later positions. Published methods never consume this position, but full blinding must be verified.'),
        implementationSha256=sha(__file__),sourceZipSha256=sha(ROOT/'outfit_data.txt.zip'))
    write(OUT/'raw-and-statistical-audit.json',result)
    write(OUT/'source-edge-outfits.json',{t+':'+b:sorted(oids) for (t,b),oids in edge_outfits.items()})
    print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
