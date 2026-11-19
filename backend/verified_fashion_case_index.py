"""Photograph-based case RAG with one shared evidence axis per wardrobe context.

Retrieval depends on the context photo and fine categories, never on the answer.
All candidates and existing alternatives are compared to the same real source
garments. Added support is measured before aggregating source-case evidence.
"""
from pathlib import Path
from collections import defaultdict
import csv,json,sqlite3
import numpy as np
from backend.catalog_coverage import TOP_CATEGORIES,BOTTOM_CATEGORIES

MODES=['raw','centered'];NEIGHBOURS=[8,32,96];TEMPERATURES=[.05,.1]
AGGREGATIONS=['case_increment','context_increment','confirmed_residual']
CONFIGURATIONS=[{'mode':mode,'k':k,'temperature':temperature,'aggregation':aggregation}
    for mode in MODES for k in NEIGHBOURS for temperature in TEMPERATURES for aggregation in AGGREGATIONS]


def key(configuration):
    return json.dumps(configuration,sort_keys=True,separators=(',',':'))


def normalized(values):
    values=np.asarray(values,np.float32)
    return values/np.maximum(np.linalg.norm(values,axis=-1,keepdims=True),1e-12)


class FashionCaseIndex:
    def __init__(self,source,cache,device='cpu'):
        self.source,self.cache=Path(source),Path(cache)
        profile=json.loads((self.cache/'encoder.json').read_text(encoding='utf8'))
        if profile['model']!='patrickjohncyh/fashion-clip' or profile['dimension']!=512:
            raise ValueError('Frozen FashionCLIP photograph descriptors required')
        ids=json.loads((self.cache/'itemIds.json').read_text(encoding='utf8'))
        taxonomy=Path(__file__).parent/'data/fashion-taxonomy-v28'
        verified=json.loads((taxonomy/'categories.json').read_text(encoding='utf8'))
        from backend.utility_calibration import digest
        manifest=json.loads((taxonomy/'manifest.json').read_text(encoding='utf8'))
        if digest(taxonomy/'categories.json')!=manifest['categoriesSha256']:raise ValueError('Verified taxonomy changed')
        self.categories={int(pid):value for pid,value in verified.items()}
        self.ids=[pid for pid in ids if int(pid) in self.categories]
        self.ix={int(pid):i for i,pid in enumerate(ids) if int(pid) in self.categories}
        self.raw=np.load(self.cache/'vectors.npy',mmap_mode='r')
        if self.raw.shape!=(len(ids),512) or not np.isfinite(self.raw).all():raise ValueError('Complete finite image cache required')
        self.pixels={int(r['itemId']):r['pixelSha256'] for r in map(json.loads,(self.source/'photo-hashes.jsonl').read_text(encoding='utf8').splitlines())}
        self.records=[];seen=set();self.known=set();self.pools=defaultdict(list);role_indices=defaultdict(set)
        # Compatibility facts from author validation/test CSVs are held out.
        for split in ['train']:
            with (self.source/f'data/{split}.csv').open(encoding='utf8') as stream:
                for line_number,row in enumerate(csv.reader(stream),1):
                    _,top,bottom,_=map(int,row)
                    if top not in self.ix or bottom not in self.ix:continue
                    category_top,category_bottom=self.categories[top],self.categories[bottom]
                    # Both IDs must occupy their verified semantic roles.
                    if category_top not in TOP_CATEGORIES or category_bottom not in BOTTOM_CATEGORIES:continue
                    identity=(self.pixels[top],self.pixels[bottom]);self.known.add(identity)
                    if identity in seen:continue
                    seen.add(identity)
                    index=len(self.records)
                    self.records.append({'top':top,'bottom':bottom,'sourceId':f'gpbpr:{split}:{line_number}'})
                    self.pools[(category_top,category_bottom)].append(index)
                    role_indices[('top',category_top)].add(self.ix[top]);role_indices[('bottom',category_bottom)].add(self.ix[bottom])
        # Stable physical-case ordering makes corpus row order irrelevant.
        self.records.sort(key=lambda r:(self.pixels[r['top']],self.pixels[r['bottom']]))
        self.pools=defaultdict(list)
        for index,r in enumerate(self.records):self.pools[(self.categories[r['top']],self.categories[r['bottom']])].append(index)
        self.mean={}
        for role,indices in role_indices.items():self.mean[role]=self.raw[sorted(indices)].mean(axis=0)
        self.centered=np.array(self.raw,copy=True)
        for pid,i in self.ix.items():
            category=self.categories[pid]
            role='top' if ('top',category) in self.mean else 'bottom'
            self.centered[i]=normalized(self.raw[i]-self.mean[(role,category)])
        self.device=device;self.gpu={};self.pool_rows={};self.context_groups={};self.retrieval_cache={}
        if device=='cuda':
            import torch
            self.gpu={mode:torch.from_numpy(np.array(matrix,copy=True)).to('cuda') for mode,matrix in [('raw',self.raw),('centered',self.centered)]}
        for category,indices in self.pools.items():
            groups=defaultdict(list)
            for i in indices:groups[self.pixels[self.records[i]['top']]].append(i)
            self.context_groups[category]=[groups[pixel] for pixel in sorted(groups)]
            self.pool_rows[category]=np.array([self.ix[self.records[group[0]]['top']] for group in self.context_groups[category]],dtype=np.int64)

    def direct(self,top,bottom):
        return (self.pixels[top],self.pixels[bottom]) in self.known

    def retrieve(self,top,bottom_category,mode,maximum=96):
        cache_key=(top,bottom_category,mode)
        if cache_key in self.retrieval_cache:return self.retrieval_cache[cache_key]
        group=(self.categories[top],bottom_category);indices=self.pools.get(group,[])
        if not indices:return None
        rows=self.pool_rows[group];matrix=self.raw if mode=='raw' else self.centered
        if self.device=='cuda':
            import torch
            with torch.inference_mode():
                products=self.gpu[mode].index_select(0,torch.from_numpy(rows).to('cuda'))
                similarities=(products@self.gpu[mode][self.ix[top]]).cpu().numpy()
        else:similarities=matrix[rows]@matrix[self.ix[top]]
        order=np.argsort(-similarities,kind='stable')[:maximum]
        order=order[similarities[order]>0]
        if not len(order):return None
        records=[];scores=[];ranks=[];divisors=[]
        for rank,j in enumerate(order):
            siblings=self.context_groups[group][j]
            for index in siblings:
                records.append(self.records[index]);scores.append(similarities[j]);ranks.append(rank);divisors.append(len(siblings))
        result={'records':records,'similarities':np.array(scores),'ranks':np.array(ranks),'divisors':np.array(divisors),
                'targets':np.array([matrix[self.ix[r['bottom']]] for r in records])}
        self.retrieval_cache[cache_key]=result
        return result

    def features(self,scenario):
        contexts=list(map(int,scenario['contexts']));candidates=list(map(int,scenario['candidates']));owned=list(map(int,scenario['owned']))
        def unique_photos(items):
            grouped={}
            for pid in sorted(items):
                if pid not in self.ix:raise ValueError('A verified photograph descriptor is required')
                grouped.setdefault(self.pixels[pid],pid)
            return [grouped[pixel] for pixel in sorted(grouped)]
        contexts=unique_photos(contexts);owned=unique_photos(owned)
        if not candidates or not contexts:raise ValueError('Candidate and context photographs required')
        category=self.categories[candidates[0]]
        if any(self.categories[pid]!=category for pid in candidates+owned):raise ValueError('Same fine category required')
        if category not in BOTTOM_CATEGORIES or any(self.categories[pid] not in TOP_CATEGORIES for pid in contexts):
            raise ValueError('This profile compares bottom candidates with top contexts')
        accum={key(c):[] for c in CONFIGURATIONS}
        candidates_and_owned=candidates+owned;n=len(candidates)
        same_owned=np.array([any(self.pixels[c]==self.pixels[e] for e in owned) for c in candidates])
        for context in contexts:
            direct_candidates=np.array([self.direct(context,c) for c in candidates])
            direct_owned=np.array([self.direct(context,e) for e in owned])
            for mode in MODES:
                chosen=self.retrieve(context,category,mode)
                if chosen is None:continue
                matrix=self.raw if mode=='raw' else self.centered
                target=np.array([matrix[self.ix[pid]] for pid in candidates_and_owned])
                support=np.maximum(target@chosen['targets'].T,0)
                for k in NEIGHBOURS:
                    mask=chosen['ranks']<k;count=int(mask.sum());local=support[:,mask]
                    for temperature in TEMPERATURES:
                        logits=(chosen['similarities'][mask]-chosen['similarities'][0])/temperature
                        weight=np.exp(logits)/chosen['divisors'][mask];weight/=weight.sum()
                        average=local@weight;c=np.where(direct_candidates,1.,average[:n])
                        existing=np.where(direct_owned,1.,average[n:])
                        maximum=existing.max() if len(existing) else 0.
                        prior=local[n:].max(axis=0) if owned else np.zeros(count)
                        values={'case_increment':np.maximum(local[:n]-prior,0)@weight,
                                'context_increment':np.maximum(c-maximum,0),
                                'confirmed_residual':c*(not direct_owned.any())}
                        for aggregation in AGGREGATIONS:
                            m=values[aggregation].copy();m[same_owned]=0
                            config={'mode':mode,'k':k,'temperature':temperature,'aggregation':aggregation}
                            accum[key(config)].append([c.tolist(),np.minimum(m,c).tolist()])
        return {name:{'coverage':len(rows)/len(contexts),'C':np.array(rows)[:,0].mean(axis=0).tolist() if rows else None,
                      'M':np.array(rows)[:,1].mean(axis=0).tolist() if rows else None} for name,rows in accum.items()}
