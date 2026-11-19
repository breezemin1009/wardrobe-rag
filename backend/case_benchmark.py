"""Text-only paired-case retrieval experiment on the public Maryland Polyvore data.

Each held-out outfit contributes one top/bottom query with three same-category
distractors. Item IDs, outfit descriptions and the answer position are excluded
from features. Selection uses validation only. Outputs include every query and
the unchanged protocol, enabling failures and ties to be inspected.
"""
from pathlib import Path
from collections import defaultdict, Counter
import argparse
import csv
import hashlib
import json
import platform
import random
import re
import tarfile
from datetime import datetime, timezone

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, precision_score, recall_score
from scipy.stats import binomtest

TOP = {11, 15, 17, 18, 19, 21, 104, 309}
BOTTOM = {7, 8, 9, 10, 27, 28, 29, 237, 238, 239, 240, 241, 310}
SEED = 20261019


def product_id(item):
    match = re.search(r'(?:[?&])tid=(\d+)', item.get('image', ''))
    return match.group(1) if match else 'name:' + item['name'].lower().strip()


def load_data(archive):
    with tarfile.open(archive) as source:
        return {split: json.load(source.extractfile(split + '_no_dup.json'))
                for split in ['train', 'valid', 'test']}


def text(item):
    # Only item description is observable. IDs, prices and set descriptions leak no features.
    return re.sub(r'\s+', ' ', item['name'].lower()).strip()


def question_set(outfits, split):
    rng = random.Random(SEED + (1 if split == 'valid' else 2))
    pools = defaultdict(dict)
    positives = defaultdict(set)
    eligible = []
    for outfit in outfits:
        tops = [i for i in outfit['items'] if i['categoryid'] in TOP and text(i)]
        bottoms = [i for i in outfit['items'] if i['categoryid'] in BOTTOM and text(i)]
        for item in tops + bottoms:
            pools[item['categoryid']][product_id(item)] = item
        for top in tops:
            for bottom in bottoms:
                positives[product_id(top)].add(product_id(bottom))
                positives[product_id(bottom)].add(product_id(top))
        if tops and bottoms:
            eligible.append((outfit, tops, bottoms))
    rows = []
    exclusions = Counter()
    for outfit, tops, bottoms in eligible:
        target_top = rng.randrange(2) == 0
        target = rng.choice(tops if target_top else bottoms)
        context = rng.choice(bottoms if target_top else tops)
        forbidden = {product_id(i) for i in outfit['items']} | positives[product_id(context)]
        pool = [i for pid, i in sorted(pools[target['categoryid']].items()) if pid not in forbidden]
        if len(pool) < 3:
            exclusions['insufficient_same_category_distractors'] += 1
            continue
        choices = [target] + rng.sample(pool, 3)
        rng.shuffle(choices)
        rows.append({'queryId': split + '-' + outfit['set_id'], 'split': split,
                     'targetIsTop': target_top, 'context': context, 'choices': choices,
                     'correct': next(i for i, c in enumerate(choices) if product_id(c) == product_id(target))})
    return rows, {'outfits': len(outfits), 'topBottomOutfits': len(eligible), 'questions': len(rows),
                  'excluded': dict(exclusions), 'positives': dict(Counter(r['correct'] for r in rows))}


class PairedCaseRetriever:
    """Retrieve training outfits by aligned top and bottom description similarity."""
    def __init__(self, pairs):
        self.pairs = pairs
        # Vocabulary and document frequencies are fitted on retrieval corpus only.
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=1, sublinear_tf=True,
                                          strip_accents='unicode', dtype=np.float32)
        self.vectorizer.fit([text(t) for t, _, _ in pairs] + [text(b) for _, b, _ in pairs])
        self.tops = self.vectorizer.transform([text(t) for t, _, _ in pairs])
        self.bottoms = self.vectorizer.transform([text(b) for _, b, _ in pairs])
        self.joint = self.vectorizer.transform([text(t) + ' ' + text(b) for t, b, _ in pairs])

    def scores(self, rows):
        specs = []
        for q in rows:
            for c in q['choices']:
                t, b = (c, q['context']) if q['targetIsTop'] else (q['context'], c)
                specs.append((text(t), text(b)))
        result = {mode: {k: [] for k in [1, 5, 10]} for mode in ['joint', 'aligned']}
        for start in range(0, len(specs), 128):
            batch = specs[start:start+128]
            a = (self.vectorizer.transform([t for t, _ in batch]) @ self.tops.T).toarray()
            b = (self.vectorizer.transform([b for _, b in batch]) @ self.bottoms.T).toarray()
            aligned = np.sqrt(np.maximum(a*b, 0))
            joint = (self.vectorizer.transform([t+' '+b for t, b in batch]) @ self.joint.T).toarray()
            for mode, values in [('joint', joint), ('aligned', aligned)]:
                nearest = np.sort(np.partition(values, -10, axis=1)[:, -10:], axis=1)[:, ::-1]
                for k in [1, 5, 10]:
                    result[mode][k].extend(nearest[:, :k].mean(axis=1).astype(float).tolist())
        return {m: {k: np.asarray(v).reshape(len(rows),4) for k,v in options.items()} for m,options in result.items()}


def query_metrics(scores, rows):
    correct=np.asarray([q['correct'] for q in rows])
    accuracy=[]; reciprocal=[]; ties=[]
    for row, answer in zip(scores, correct):
        best=np.isclose(row, row.max(), rtol=0, atol=1e-7)
        accuracy.append(float(best[answer])/best.sum())
        tied=np.isclose(row, row[answer], rtol=0, atol=1e-7)
        greater=int(np.sum((row > row[answer]) & ~tied))
        count=int(tied.sum())
        reciprocal.append(sum(1/(greater+j) for j in range(1,count+1))/count)
        ties.append(int(best.sum()>1))
    return np.asarray(accuracy),np.asarray(reciprocal),np.asarray(ties)


def interval(values, seed=SEED, samples=5000):
    rng=np.random.default_rng(seed)
    values=np.asarray(values,dtype=float)
    estimates=np.asarray([values[rng.integers(0,len(values),len(values))].mean() for _ in range(samples)])
    return [float(x) for x in np.quantile(estimates,[.025,.975])]


def run(archive, output):
    output.mkdir(exist_ok=True, parents=True)
    splits=load_data(archive)
    valid,va=question_set(splits['valid'],'valid')
    test,ta=question_set(splits['test'],'test')
    heldout={product_id(i) for s in ['valid','test'] for o in splits[s] for i in o['items']}
    heldout_names={text(i) for s in ['valid','test'] for o in splits[s] for i in o['items']}
    pairs=[]; seen=set(); removed=0
    for outfit in splits['train']:
        for top in [i for i in outfit['items'] if i['categoryid'] in TOP and text(i)]:
            for bottom in [i for i in outfit['items'] if i['categoryid'] in BOTTOM and text(i)]:
                if any(product_id(i) in heldout or text(i) in heldout_names for i in [top,bottom]):
                    removed+=1;continue
                key=(product_id(top),product_id(bottom))
                if key not in seen:
                    seen.add(key);pairs.append((top,bottom,outfit['set_id']))
    protocol={'version':'paired-case-text-1','dataset':'Maryland Polyvore ACM MM 2017',
              'archiveUrl':'https://raw.githubusercontent.com/xthan/polyvore-dataset/master/polyvore.tar.gz',
              'archiveSha256':hashlib.sha256(archive.read_bytes()).hexdigest(), 'seed':SEED,
              'features':'item name text only; train-only word 1-2gram TF-IDF, sublinear TF',
              'corpus':'unique train top/bottom product pairs, excluding any validation/test product ID or exact item name',
              'task':'one top/bottom completion per eligible outfit; 3 same fine category distractors; answer shuffled',
              'distractors':'same split; exclude all known co-outfit positives for context and original outfit products',
              'selection':'validation mean tie-aware top-1 accuracy, k in [1,5,10], smallest k on ties',
              'test':'one frozen comparison; expected uniform tie-breaking, question bootstrap 5000',
              'interpretation':'recovery of observed co-outfit item among sampled same-category alternatives',
              'trainOutfits':len(splits['train']),'trainPairs':len(pairs),'excludedTrainPairs':removed,
              'valid':va,'test':ta,'topCategories':sorted(TOP),'bottomCategories':sorted(BOTTOM)}
    (output/'protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding='utf8')
    (output/'questions.json').write_text(json.dumps({'valid':valid,'test':test},ensure_ascii=False),encoding='utf8')
    print(json.dumps({'trainPairs':len(pairs),'valid':va,'test':ta}),flush=True)
    retriever=PairedCaseRetriever(pairs)
    validation=retriever.scores(valid)
    choices={}
    tuning=[]
    for mode,options in validation.items():
        for k,scores in options.items():
            accuracy,mrr,tie=query_metrics(scores,valid)
            tuning.append({'mode':mode,'k':k,'accuracy':float(accuracy.mean()),'mrr':float(mrr.mean())})
        choices[mode]=max([1,5,10],key=lambda k:(query_metrics(options[k],valid)[0].mean(),-k))
    print(json.dumps({'validation':tuning,'selected':choices}),flush=True)
    # Threshold is selected solely from validation balanced accuracy, then frozen.
    thresholds={}
    vlabels=np.zeros((len(valid),4),int)
    vlabels[np.arange(len(valid)),[q['correct'] for q in valid]]=1
    for mode,k in choices.items():
        s=validation[mode][k].ravel();y=vlabels.ravel()
        candidates=np.unique(np.quantile(s,np.linspace(0,1,201)))
        thresholds[mode]=float(max(candidates,key=lambda t:(balanced_accuracy_score(y,s>=t),t)))
    (output/'selection.json').write_text(json.dumps({'tuning':tuning,'k':choices,'threshold':thresholds},indent=2),encoding='utf8')
    # No model or threshold changes occur after this point.
    testing=retriever.scores(test)
    rows=[];summary={};accuracies={}
    labels=np.zeros((len(test),4),int);labels[np.arange(len(test)),[q['correct'] for q in test]]=1
    for mode,k in choices.items():
        s=testing[mode][k];a,m,t=query_metrics(s,test);accuracies[mode]=a
        summary[mode]={'k':k,'threshold':thresholds[mode],'questions':len(test),'accuracy':float(a.mean()),
                       'accuracy95ci':interval(a),'mrr':float(m.mean()),'topTies':int(t.sum()),
                       'auc':float(roc_auc_score(labels.ravel(),s.ravel())),
                       'balancedAccuracy':float(balanced_accuracy_score(labels.ravel(),s.ravel()>=thresholds[mode])),
                       'precision':float(precision_score(labels.ravel(),s.ravel()>=thresholds[mode],zero_division=0)),
                       'recall':float(recall_score(labels.ravel(),s.ravel()>=thresholds[mode],zero_division=0))}
        for q,score,acc,mrr,tie in zip(test,s,a,m,t):
            rows.append({'queryId':q['queryId'],'mode':mode,'k':k,'correct':q['correct'],
                         'accuracy':float(acc),'mrr':float(mrr),'topTie':int(tie),
                         **{'score'+str(i):float(x) for i,x in enumerate(score)}})
    diff=accuracies['aligned']-accuracies['joint']
    rng=np.random.default_rng(SEED)
    # Paired randomization; ties retain their expected credit rather than answer order.
    observed=abs(diff.mean());count=0
    for _ in range(20000):
        count+=abs((diff*rng.choice([-1,1],len(diff))).mean())>=observed-1e-12
    paired={'meanAccuracyDifference':float(diff.mean()),'difference95ci':interval(diff),
            'randomizationP':(count+1)/20001,'randomizationDraws':20000,
            'betterQueries':int((diff>1e-9).sum()),'worseQueries':int((diff<-1e-9).sum()),
            'unchangedQueries':int((abs(diff)<=1e-9).sum())}
    report={'generatedAt':datetime.now(timezone.utc).isoformat(),'protocol':protocol,'validation':tuning,
            'summary':summary,'paired':paired,'environment':{'python':platform.python_version(),
            'numpy':np.__version__},'codeSha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
    with (output/'query-results.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    # Save the retrieval corpus as evidence records, without images or hidden answer fields.
    (output/'training-cases.jsonl').write_text('\n'.join(json.dumps({'sourceId':sid,'top':text(t),'bottom':text(b),
        'topProduct':product_id(t),'bottomProduct':product_id(b),
        'topCategory':t['categoryid'],'bottomCategory':b['categoryid']},ensure_ascii=False) for t,b,sid in pairs),encoding='utf8')
    print(json.dumps({'summary':summary,'paired':paired}),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    run(args.archive,args.output)
