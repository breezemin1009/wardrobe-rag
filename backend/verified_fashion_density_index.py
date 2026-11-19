"""Case retrieval with a source-only correction for common target appearance.

FashionCLIP retrieves similar products, not complementary garments. This layer
compares the target density in retrieved real outfits with its marginal density
in the same source collection. Neither query outcomes nor query creators enter
the estimator. Known pairs use the same estimator as unobserved query pairs.
"""
import numpy as np
from scipy.special import expit, logsumexp
from backend.verified_fashion_case_index import FashionCaseIndex, key

CONFIGURATIONS = [dict(mode=mode,k=k,temperature=t,aggregation=aggregation)
    for mode in ['raw','centered'] for k in [8,32,96]
    for t in [.025,.05,.1]
    for aggregation in ['positive_difference','confirmed_residual']]


class FashionDensityIndex(FashionCaseIndex):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.marginal_cache={}

    def marginal(self,items,category,mode,temperature):
        cache_key=(tuple(items),category,mode,temperature)
        if cache_key in self.marginal_cache:return self.marginal_cache[cache_key]
        groups=self.context_groups[category]
        records=[(i,len(group)) for group in groups for i in group]
        indices=np.array([self.ix[self.records[i]['bottom']] for i,_ in records])
        logweights=-np.log(np.array([n for _,n in records]))-np.log(len(groups))
        rows=np.array([self.ix[pid] for pid in items])
        if self.device=='cuda':
            import torch
            with torch.inference_mode():
                target=self.gpu[mode].index_select(0,torch.from_numpy(rows).to('cuda'))
                references=self.gpu[mode].index_select(0,torch.from_numpy(indices).to('cuda'))
                similarities=(target@references.T).cpu().numpy()
        else:
            matrix=self.raw if mode=='raw' else self.centered
            similarities=matrix[rows]@matrix[indices].T
        result=logsumexp(similarities/temperature+logweights,axis=1)
        self.marginal_cache[cache_key]=result
        return result

    def features(self,scenario):
        def unique(items):
            grouped={}
            for pid in sorted(map(int,items)):grouped.setdefault(self.pixels[pid],pid)
            return [grouped[pixel] for pixel in sorted(grouped)]
        contexts=unique(scenario['contexts']);owned=unique(scenario['owned'])
        candidates=list(map(int,scenario['candidates']));items=candidates+owned;n=len(candidates)
        if not contexts or not candidates:raise ValueError('Photographs required')
        category=self.categories[candidates[0]]
        if any(self.categories[pid]!=category for pid in items):raise ValueError('Same fine target category required')
        same_owned=np.array([any(self.pixels[c]==self.pixels[e] for e in owned) for c in candidates])
        result={key(cfg):[] for cfg in CONFIGURATIONS}
        for top in contexts:
            covered=any(self.direct(top,e) for e in owned)
            for mode in ['raw','centered']:
                evidence=self.retrieve(top,category,mode)
                if evidence is None:continue
                matrix=self.raw if mode=='raw' else self.centered
                similarities=matrix[[self.ix[pid] for pid in items]]@evidence['targets'].T
                pool=(self.categories[top],category)
                for k in [8,32,96]:
                    mask=evidence['ranks']<k
                    for temperature in [.025,.05,.1]:
                        logweights=evidence['similarities'][mask]/temperature-np.log(evidence['divisors'][mask])
                        logweights-=logsumexp(logweights)
                        conditional=logsumexp(similarities[:,mask]/temperature+logweights,axis=1)
                        ratio=conditional-self.marginal(items,pool,mode,temperature)
                        # Equal-prior source association index; not aesthetic probability.
                        support=expit(ratio);c=support[:n]
                        prior=support[n:].max() if owned else 0.
                        for aggregation in ['positive_difference','confirmed_residual']:
                            m=np.maximum(c-prior,0) if aggregation=='positive_difference' else c*(not covered)
                            m[same_owned]=0
                            cfg=dict(mode=mode,k=k,temperature=temperature,aggregation=aggregation)
                            result[key(cfg)].append([c.tolist(),m.tolist()])
        return {name:dict(coverage=len(rows)/len(contexts),
            C=np.array(rows)[:,0].mean(0).tolist() if rows else None,
            M=np.array(rows)[:,1].mean(0).tolist() if rows else None) for name,rows in result.items()}
