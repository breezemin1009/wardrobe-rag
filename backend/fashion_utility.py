"""Serving adapter for the prospectively validated, identified-catalog RAG.

Keeps this profile separate from the application's text/design rule assessment.
All inputs are verified catalog photographs. The server does not silently map
an arbitrary user photograph to a catalog identity.
"""
from pathlib import Path
import json,hashlib,threading
import numpy as np
from scipy.special import logsumexp,expit
from backend.verified_fashion_density_index import FashionDensityIndex


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


class FashionUtility:
    def __init__(self,validation):
        validation=Path(validation);self.protocol=json.loads((validation/'protocol.json').read_text(encoding='utf8'))
        self.report=json.loads((validation/'evaluation-report.json').read_text(encoding='utf8'))
        if not self.report['statisticalCriterionMet'] or self.report['protocolSha256']!=digest(validation/'protocol.json'):
            raise ValueError('Verified, frozen evaluation required')
        for name,sha in self.protocol['codeHashes'].items():
            if digest(Path(__file__).parent/name)!=sha:raise ValueError('Validated implementation changed')
        self.profile=self.report['selected'];self.configuration=self.profile['configuration']
        self.profile_sha256=digest(validation/'protocol.json')
        if self.configuration['aggregation'] not in {'confirmed_residual','positive_difference'} or not 0<=self.profile['alpha']<=1:
            raise ValueError('Serving adapter requires a verified utility profile')
        for name,sha in self.protocol['encoderLock'].items():
            if digest(Path(self.protocol['cache'])/name)!=sha:raise ValueError('Validated photograph cache changed')
        for name,sha in self.protocol['sourceHashes'].items():
            if digest(Path(self.protocol['source'])/name)!=sha:raise ValueError('Validated reference changed')
        for name,sha in self.protocol['taxonomyHashes'].items():
            if digest(Path(__file__).parent/'data/fashion-taxonomy-v28'/name)!=sha:raise ValueError('Validated taxonomy changed')
        self.index=FashionDensityIndex(Path(self.protocol['source']),Path(self.protocol['cache']))
        self.lock=threading.RLock()

    def _context(self,top,candidates,owned):
        e=self.index;cfg=self.configuration;items=candidates+owned;n=len(candidates)
        category=e.categories[candidates[0]];evidence=e.retrieve(top,category,cfg['mode'])
        if evidence is None:raise ValueError('No comparable source cases')
        matrix=e.raw if cfg['mode']=='raw' else e.centered
        similarity=matrix[[e.ix[pid] for pid in items]]@evidence['targets'].T
        mask=evidence['ranks']<cfg['k'];temperature=cfg['temperature']
        logweights=evidence['similarities'][mask]/temperature-np.log(evidence['divisors'][mask])
        logweights-=logsumexp(logweights)
        logterms=similarity[:,mask]/temperature+logweights
        conditional=logsumexp(logterms,axis=1)
        marginal=e.marginal(items,(e.categories[top],category),cfg['mode'],temperature)
        full_support=expit(conditional-marginal)
        support=full_support[:n]
        covered=[pid for pid in owned if e.direct(top,pid)]
        if cfg['aggregation']=='confirmed_residual':gains=support*int(not covered)
        else:gains=np.maximum(support-(full_support[n:].max() if owned else 0.),0)
        records=[r for r,m in zip(evidence['records'],mask) if m]
        sources=[]
        for i in range(n):
            order=np.argsort(-logterms[i],kind='stable')[:3]
            sources.append([dict(sourceId=records[j]['sourceId'],topProduct=str(records[j]['top']),bottomProduct=str(records[j]['bottom']),
                relativeEvidenceWeight=float(np.exp(logterms[i,j]-conditional[i]))) for j in order])
        return support,gains,dict(contextProduct=str(top),existingConfirmedProducts=list(map(str,covered))),sources

    def assess(self,candidates,contexts,owned):
        with self.lock:
            try:return self._assess(candidates,contexts,owned)
            finally:
                # Request-scoped memoization must not grow with a user's history.
                self.index.retrieval_cache.clear();self.index.marginal_cache.clear()

    def _assess(self,candidates,contexts,owned):
        e=self.index
        def canonical(items):
            seen={}
            for pid in sorted(set(map(int,items))):
                if pid not in e.ix:raise ValueError('Verified catalog photograph required')
                seen.setdefault(e.pixels[pid],pid)
            return [seen[pixel] for pixel in sorted(seen)]
        candidates=canonical(candidates);contexts=canonical(contexts);owned=canonical(owned)
        if not candidates or not contexts:raise ValueError('Candidate and wardrobe contexts required')
        category=e.categories[candidates[0]]
        from backend.catalog_coverage import TOP_CATEGORIES,BOTTOM_CATEGORIES
        if category not in BOTTOM_CATEGORIES or any(e.categories[pid]!=category for pid in candidates+owned):
            raise ValueError('Bottom candidates and existing alternatives need the same fine category')
        if any(e.categories[pid] not in TOP_CATEGORIES for pid in contexts):raise ValueError('Top contexts required')
        same_owned=[any(e.pixels[c]==e.pixels[o] for o in owned) for c in candidates]
        groups=[[] for _ in candidates];cs=[];ms=[]
        for top in contexts:
            c,m,base,refs=self._context(top,candidates,owned)
            m[np.array(same_owned)]=0;cs.append(c);ms.append(m)
            for i,pid in enumerate(candidates):
                groups[i].append(dict(**base,candidateSupport=float(c[i]),additionalSupport=float(m[i]),
                    candidateRelationConfirmed=e.direct(top,pid),evidence=refs[i]))
        combination=np.array(cs).mean(0);additional=np.array(ms).mean(0)
        alpha=self.profile['alpha'];score=100*(alpha*combination+(1-alpha)*additional)
        results=[]
        for i,pid in enumerate(candidates):
            results.append(dict(candidateId=str(pid),score=float(score[i]),combination=float(100*combination[i]),
                contribution=float(100*additional[i]),contextCount=len(contexts),alreadyOwned=same_owned[i],groups=groups[i]))
        results.sort(key=lambda r:(-r['score'],r['candidateId']))
        scores=np.array([r['score'] for r in results])
        for row in results:row['rank']=1+int((scores>row['score']+1e-6).sum())
        identity=dict(profile=self.profile_sha256,candidates=sorted(e.pixels[pid] for pid in candidates),
            contexts=sorted(e.pixels[pid] for pid in contexts),owned=sorted(e.pixels[pid] for pid in owned))
        fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()
        return dict(mode='verified-fashion-utility-v30',results=results,evaluationFingerprint=fingerprint,
            meaning='공개 코디에서 확인된 기존 조합을 제외하고, 사진으로 검색한 코디 사례의 추가 지원을 비교합니다.',
            scoreMeaning='100은 검색 사례의 추가 지원 지수가 상한에 가까운 경우입니다. 어울림 확률이나 확인된 조합 수와 구분합니다.',
            sourceScope='공개 카탈로그에서 확인된 상품 사진과 종류를 사용한 하의 후보 비교',
            method=dict(alpha=alpha,formula=('구매 우선 지수 = 각 상의의 추가 지원 점수 평균' if alpha==0
                else f'구매 우선 지수 = 조합 지원 점수 × {alpha:g} + 추가 지원 점수 × {1-alpha:g}'),
                retrieval=f"패션 사진 특징 → 같은 종류의 상의 사례 {self.configuration['k']}개 → 실제 연결된 하의 사례 → 흔한 디자인의 검색 쏠림 보정",
                configuration=self.configuration),
            validation=dict(cases=self.report['cases'],selection='개발 자료에서 설정을 고정한 뒤 사진이 겹치지 않는 자료로 검증',
                task='새로운 공개 코디 조합을 추가하는 후보를 고르는 평가',top1Rate=self.report['rates']['proposed']),
            aestheticProbability=False)
