"""Case-retrieval comparison module returning observable ranks and group counts.

The lexical score is a retrieval similarity. We return scores/ranks per confirmed
description and top cases, rather than mapping similarity to aesthetic percent.
This research module is separate from the existing mobile assessment endpoint.
"""
from pathlib import Path
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


class CaseRecommender:
    def __init__(self, records, k=10, mode='joint'):
        if mode not in {'joint','aligned'} or not isinstance(k,int) or k < 1:
            raise ValueError('Unsupported paired retrieval configuration')
        self.k=k
        self.mode=mode
        # Repeated rows carry no extra weight.
        self.records=list({(r['topProduct'],r['bottomProduct']):r for r in records}.values())
        if not self.records:raise ValueError('A nonempty paired-case corpus is required')
        self.vectorizer=TfidfVectorizer(ngram_range=(1,2),min_df=1,sublinear_tf=True,
            strip_accents='unicode',dtype=np.float32)
        self.vectorizer.fit([r['top'] for r in self.records]+[r['bottom'] for r in self.records])
        self.matrix=self.vectorizer.transform([r['top']+' '+r['bottom'] for r in self.records])
        self.tops=self.vectorizer.transform([r['top'] for r in self.records])
        self.bottoms=self.vectorizer.transform([r['bottom'] for r in self.records])

    @classmethod
    def load(cls,path,**options):
        return cls([json.loads(line) for line in Path(path).read_text(encoding='utf8').splitlines() if line],**options)

    def pair(self,target,context):
        for item in [target,context]:
            if not item.get('description','').strip():raise ValueError('Confirmed item description is required')
        if target['category']==context['category']:raise ValueError('A top and a bottom are required')
        top,bottom=(target,context) if target['category']=='상의' else (context,target)
        field='topCategory' if target['category']=='상의' else 'bottomCategory'
        eligible=[i for i,r in enumerate(self.records) if r[field]==target['categoryId']]
        fallback=not eligible
        if fallback:eligible=list(range(len(self.records)))
        if self.mode == 'aligned':
            a=(self.vectorizer.transform([top['description']]) @ self.tops[eligible].T).toarray()[0]
            b=(self.vectorizer.transform([bottom['description']]) @ self.bottoms[eligible].T).toarray()[0]
            values=np.sqrt(np.maximum(a*b,0))
        else:
            vector=self.vectorizer.transform([top['description'].lower()+' '+bottom['description'].lower()])
            values=(vector @ self.matrix[eligible].T).toarray()[0]
        selected=np.argsort(-values,kind='stable')[:min(self.k,len(eligible))]
        return {'similarity':float(values[selected].mean()),'categoryFallback':fallback,
                'cases':[{'sourceId':self.records[eligible[j]]['sourceId'],
                          'top':self.records[eligible[j]]['top'],'bottom':self.records[eligible[j]]['bottom'],
                          'similarity':float(values[j])} for j in selected]}

    @staticmethod
    def key(item):
        return (item['category'],item['categoryId'],' '.join(item['description'].lower().split()))

    def compare(self,candidate,wardrobe):
        groups={self.key(i):i for i in wardrobe if i['category']!=candidate['category']}
        alternatives={self.key(i):i for i in wardrobe if i['category']==candidate['category']
                      and i['categoryId']==candidate['categoryId']}
        rows=[]
        for key,context in sorted(groups.items()):
            assessed=self.pair(candidate,context)
            baselines=[self.pair(item,context)['similarity'] for _,item in sorted(alternatives.items())]
            score=assessed['similarity'];best=max(baselines) if baselines else None
            if best is None:relation='candidate-only'
            elif abs(score-best)<=1e-7:relation='tie'
            elif score>best:relation='candidate-ahead'
            else:relation='existing-ahead'
            rank=1+sum(s>score+1e-7 for s in baselines)
            rows.append({'context':context['description'],'candidateSimilarity':score,'bestExistingSimilarity':best,
                'rank':rank,'relation':relation,'evidence':assessed['cases']})
        return {'method':'paired-case-lexical-k10','totalGroups':len(rows),
            'existingGroups':len(alternatives),'counts':{name:sum(r['relation']==name for r in rows)
                for name in ['candidate-only','candidate-ahead','tie','existing-ahead']},
            'combinationSimilarity':float(np.mean([r['candidateSimilarity'] for r in rows])) if rows else None,
            'marginalSimilarity':float(np.mean([max(r['candidateSimilarity']-(r['bestExistingSimilarity']
                if r['bestExistingSimilarity'] is not None else 0),0) for r in rows])) if rows else None,
            'meanRank':float(np.mean([r['rank'] for r in rows])) if rows else None,'groups':rows}
