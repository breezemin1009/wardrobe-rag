"""Known-source wardrobe comparison with mixed-kind existing bottoms.

This research endpoint has a different information boundary from the retained
earlier service. It does not silently replace the application's photo path.
"""
from functools import lru_cache
from threading import RLock
import hashlib,json,time
import joblib
import numpy as np
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field,ConfigDict
from threadpoolctl import threadpool_limits
from backend import thesis_known_query_study_corrected_20261027 as study
from backend import thesis_known_query_model_corrected_20261027 as method
from backend import thesis_repair_models_20261026 as b
from backend import thesis_repair_stable_kernel_20261026 as kernel
from backend.thesis_repair_expanded_models_20261026 import feature_view
from backend.pog_refined_retrieval import Retriever
from backend.pog_retrieval import text

SOURCE='https://osf.io/cg6eq/'

class WardrobeResearchUtility:
    def __init__(self,out=study.OUT):
        method.verify();self.out=out;self.selection=b.read(out/'selection-lock.json');self.report=b.read(out/'test-results.json')
        if self.report['selectionSha256']!=b.sha(out/'selection-lock.json'):raise ValueError('Selection/result binding changed')
        if not b.read(out/'size-source-audit.json')['allPassed']:raise ValueError('Source audit missing')
        self.profile=self.selection['profiles']['RAG'];path=out/(self.profile['tag']+'.joblib')
        if b.sha(path)!=self.profile['modelSha256']:raise ValueError('Model changed')
        self.model=joblib.load(path);self.engine=Retriever(out);self.items=self.engine.items;self.lock=RLock()
        self._cached_support=lru_cache(maxsize=65536)(self._point_support)
        self.canonical={a:p for p,x in self.items.items() if p in self.engine.ix for a in [p]+x['aliases']}
        self.record_lookup={(r['top'],r['bottom']):r for r in self.engine.records}
        self.confirmed=set(self.record_lookup);self.known_tops={q for q,w in self.confirmed}
        self.version='known-wardrobe-mixed-bottoms-v1-'+b.sha(out/'selection-lock.json')[:12]

    def identified(self,ids):
        try:return list(dict.fromkeys(self.canonical[p] for p in ids))
        except KeyError:raise ValueError('연구 자료에서 확인한 상품 ID를 입력해주세요.') from None

    def _point_support(self,q,candidate):
        x=kernel.base_features(self.engine,[q],[candidate])
        f,r,_=kernel.support_features(self.engine,[q],[candidate],exclude_self=False,exclude_near=True)
        features=feature_view(x,f,r,'RAG',self.profile['view'])
        return float(self.model.predict_proba(features.reshape(1,-1))[0,1])

    def assess(self,candidates,contexts,owned,include_evidence=True):
        start=time.perf_counter();candidates=sorted(self.identified(candidates));contexts=sorted(self.identified(contexts));owned=sorted(self.identified(owned))
        if not candidates or not contexts or not owned:raise ValueError('구매 후보와 보유 상의·하의를 함께 입력해주세요.')
        if len(candidates)>12 or len(contexts)>64 or len(owned)>64:raise ValueError('비교할 의류 수를 줄여주세요.')
        if any(self.items[p]['side']!='top' for p in contexts) or any(self.items[p]['side']!='bottom' for p in candidates+owned):raise ValueError('상의와 하의의 구분을 확인해주세요.')
        if not set(contexts)<=self.known_tops:raise ValueError('이 연구 비교에는 공개 코디 기록이 연결된 보유 상의가 필요합니다.')
        with self.lock,threadpool_limits(1):
            # Pairwise support depends on q,c and this immutable model/reference
            # version, not wardrobe size or other candidates. Reuse only that
            # support; recompute confirmed history and C/M/U on every request.
            S=np.asarray([[self._cached_support(q,c) for q in contexts] for c in candidates])
            facts=np.asarray([[int((q,c) in self.confirmed) for q in contexts] for c in candidates],int);S=np.maximum(S,facts)
            H=np.asarray([[int((q,w) in self.confirmed) for q in contexts] for w in owned],int)
            covered=H.max(0);C=S.mean(1);M=(S*(1-covered)).mean(1);a=self.profile['alpha'];U=a*C+(1-a)*M
            rows=[]
            for i,p in enumerate(candidates):
                have=p in owned;rows.append(dict(id=p,title=self.items[p]['title'],alreadyOwned=have,eligibleForPurchase=not have,
                    combinationSupport=float(100*C[i]),additionalContribution=0. if have else float(100*M[i]),score=0. if have else float(100*U[i]),
                    support=S[i].tolist(),confirmedCandidateLinks=facts[i].tolist(),evidence=self.evidence(p,contexts) if include_evidence else []))
            eligible=[r for r in rows if not r['alreadyOwned']]
            for r in rows:r['rank']=None if r['alreadyOwned'] else 1+sum(v['score']>r['score']+1e-8 for v in eligible)
            rows.sort(key=lambda r:(r['alreadyOwned'],-r['score'],r['id']))
        fingerprint=hashlib.sha256(json.dumps(dict(candidates=candidates,contexts=contexts,owned=owned,version=self.version),sort_keys=True).encode()).hexdigest()
        return dict(mode='research-known-wardrobe-comparison',version=self.version,evaluationFingerprint=fingerprint,results=rows,contexts=contexts,
            confirmedContextMask=covered.tolist(),inferenceSeconds=time.perf_counter()-start,
            method=dict(alpha=a,formula='전체 조합 지원 C와 기존 기록 이후의 추가 지원 M의 가중 평균',
                support='상품명과 허용된 양방향 코디 검색 특징으로 학습한 지원 지수',source=SOURCE,referencePairs=len(self.engine.records)),
            meaning=dict(scale='0~100 근거 지수',input='공개 코디가 연결된 보유 상의와 원상품 후보·기존 하의',ownedKinds='기존 청바지·바지·반바지를 함께 반영'),aestheticProbability=False)

    def evidence(self,candidate,contexts):
        e=self.engine;ci=e.ix[candidate];target=self.items[candidate];found=[]
        # Known candidate relationships set support to1; cite that actual
        # relationship before analogous retrieved examples.
        for q in contexts:
            r=self.record_lookup.get((q,candidate))
            if r is not None:found.append(dict(contextId=q,source=SOURCE,sourceOutfits=r['outfits'][:3],
                top=dict(id=q,title=self.items[q]['title']),bottom=dict(id=candidate,title=target['title']),
                relation='원자료에 확인된 후보 코디; 확인 지원1의 직접 근거'))
        for q in contexts[:4]:
            source=self.items[q];p=e.pools.get((source['sourceCategory'],source['kind'],target['sourceCategory'],target['kind']))
            if p is None:continue
            qi=e.ix[q];ad=e.vectors[qi]@e.vectors[p['ti']].T;al=(e.lex[qi]@e.lex[p['ti']].T).toarray().ravel()
            bd=kernel.fixed_rows(e.vectors[[ci]],e.vectors[p['bi']].T)[0];bl=(e.lex[ci]@e.lex[p['bi']].T).toarray().ravel()
            near=((bd>=.98)&(bl>=.90))|np.asarray([text(self.items[c]).casefold()==text(target).casefold() for c in p['bottoms']])
            keep=(~near)[p['b']];tt=p['t'][keep];bb=p['b'][keep]
            if not len(tt):continue
            av=(ad+al)/2;bv=(bd+bl)/2;t=np.unique(tt);v=np.unique(bb)
            t=t[np.argsort(-av[t],kind='stable')[:32]];v=v[np.argsort(-bv[v],kind='stable')[:32]]
            ix=np.flatnonzero(keep)[np.isin(tt,t)|np.isin(bb,v)];score=av[p['t'][ix]]+bv[p['b'][ix]]
            for j in ix[np.argsort(-score,kind='stable')[:2]]:
                r=p['rows'][int(j)];found.append(dict(contextId=q,source=SOURCE,sourceOutfits=r['outfits'][:3],
                    top=dict(id=r['top'],title=self.items[r['top']]['title']),bottom=dict(id=r['bottom'],title=self.items[r['bottom']]['title']),
                    relation='허용된 정·역방향 검색 범위의 실제 코디; 특징의 인과적 기여도는 아님'))
        return found

    def summary(self):return dict(version=self.version,study=self.report,selectedModel=self.profile)

class Query(BaseModel):
    model_config=ConfigDict(extra='forbid')
    candidates:list[str]=Field(min_length=1,max_length=12)
    contexts:list[str]=Field(min_length=1,max_length=64)
    owned:list[str]=Field(min_length=1,max_length=64)

@lru_cache(maxsize=1)
def engine():return WardrobeResearchUtility()

app=FastAPI(title='Known Wardrobe Comparison Research API')
@app.get('/study')
def summary():return engine().summary()
@app.post('/compare')
def compare(request:Query):
    try:return engine().assess(request.candidates,request.contexts,request.owned)
    except ValueError as e:raise HTTPException(422,str(e)) from None
@app.post('/explain')
def explain(request:Query):
    from backend.wardrobe_case_explanation import explain_case
    try:return explain_case(engine(),request)
    except ValueError as e:raise HTTPException(422,str(e)) from None
