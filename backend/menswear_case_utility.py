"""Serving adapter for the frozen, validated menswear source-title RAG study.

This interface accepts identified source garments. The separate photo-attribute
assessment retains its own evidence rules; no text-study accuracy is assigned
to an untested photo-to-title transformation.
"""
from pathlib import Path
from threading import RLock
import json,hashlib
import numpy as np
from backend.audit_pog_age_scope import OUT
from backend.pog_refined_retrieval import Retriever
from backend.pog_cold_study import read,sha,stable

SOURCE='https://github.com/wenyuer/POG'

class MenswearCaseUtility:
    def __init__(self,out=OUT):
        self.out=Path(out);self.lock=RLock();self.report=read(self.out/'evaluation-report.json');self.selection=read(self.out/'selection.json');self.protocol=read(self.out/'protocol.json')
        if not self.report['criterionMet'] or self.report['selectionSha256']!=sha(self.out/'selection.json') or self.report['protocolSha256']!=sha(self.out/'protocol.json'):raise ValueError('Frozen menswear validation does not match this deployment')
        lock=read(self.out/'development-features-lock.json')
        if lock['codeSha256']!=sha(Path(__file__).with_name('pog_refined_retrieval.py')) or lock['baseCodeSha256']!=sha(Path(__file__).with_name('pog_retrieval.py')):raise ValueError('Frozen menswear retrieval implementation changed')
        for name,field in [('vectors.npy','vectorsSha256'),('itemIds.json','idsSha256')]:
            if sha(self.out/name)!=read(self.out/'encoder.json')[field]:raise ValueError('Frozen encoder data changed')
        self.engine=Retriever(self.out);self.items=self.engine.items;self.profile=self.selection['proposed'];self.config=json.loads(self.profile['configuration'])
        from backend.prepare_pog_menswear import classify
        self.canonical={p:k for k,v in self.items.items() if classify(v['title'])[0] is not None for p in v['aliases']}
        self.version='menswear-case-rag-v1-'+self.report['selectionSha256'][:12]

    def identified(self,ids):
        try:return list(dict.fromkeys(self.canonical[p] for p in ids))
        except KeyError:raise ValueError('확인된 남성복 원자료의 상품을 선택해주세요.') from None

    def assess(self,candidates,contexts,owned,include_evidence=True):
        candidates=self.identified(candidates);contexts=sorted(self.identified(contexts));owned=sorted(self.identified(owned))
        if not candidates or not contexts:raise ValueError('구매 후보와 보유 상의를 선택해주세요.')
        category={(self.items[p]['sourceCategory'],self.items[p]['kind']) for p in candidates+owned}
        if len(category)!=1 or any(self.items[p]['side']!='bottom' for p in candidates+owned) or any(self.items[p]['side']!='top' for p in contexts):raise ValueError('같은 종류의 하의 후보·보유 대안과 보유 상의를 선택해주세요.')
        if len(candidates)>12 or len(contexts)>64 or len(owned)>64:raise ValueError('한 번에 비교할 의류 수를 줄여주세요.')
        target=self.items[candidates[0]]
        if any((self.items[t]['sourceCategory'],self.items[t]['kind'],target['sourceCategory'],target['kind']) not in self.engine.pools for t in contexts):
            raise ValueError('해당 상의·하의 종류의 코디 근거를 먼저 확인해주세요.')
        case=dict(candidates=candidates,contexts=contexts,owned=owned)
        from threadpoolctl import threadpool_limits
        with self.lock,threadpool_limits(limits=1):
            try:
                features=self.engine.features(case,self.config);a=self.profile['policy']['alpha'];m=features[self.profile['policy']['marginal']];scores=a*features['C']+(1-a)*m
                rows=[]
                for i,pid in enumerate(candidates):
                    rows.append(dict(id=pid,title=self.items[pid]['title'],audience='mens',combinationSupport=float(features['C'][i]*100),additionalContribution=float(m[i]*100),score=float(scores[i]*100),alreadyOwned=pid in owned,evidence=self.evidence(pid,contexts) if include_evidence else []))
                rows.sort(key=lambda r:(-r['score'],r['id']))
                for row in rows:row['rank']=1+sum(r['score']>row['score']+1e-8 for r in rows)
                fingerprint=hashlib.sha256(stable(dict(candidates=sorted(candidates),contexts=contexts,owned=owned,version=self.version)).encode()).hexdigest()
                return dict(mode='identified-menswear-source-title-rag',version=self.version,results=rows,evaluationFingerprint=fingerprint,
                    method=dict(retrieval=self.config,alpha=a,marginal=self.profile['policy']['marginal'],formula='보유 상의별로 후보의 근거 지원에서 기존 하의의 최고 지원을 뺀 양수 부분을 평균',ranking='추가 조합 기여를 우선하는 개발 자료 선택값',sourcePairs=len(self.engine.records)),
                    meaning=dict(combinationSupport='비슷한 실제 코디와 비교한 조건부 근거 지원 지수',additionalContribution='기존 하의가 제공하던 지원을 넘어서는 추가 근거의 크기',scale='0~100 지수; 큰 값일수록 해당 근거가 강함',input='원자료의 상품 정보가 확인된 의류',source=SOURCE),aestheticProbability=False)
            finally:self.engine.cache.clear()

    def evidence(self,candidate,contexts):
        """Real source records supporting retrieval; never a generated example."""
        result=[];target=self.items[candidate];ci=self.engine.ix[candidate]
        for top in contexts[:4]:
            source=self.items[top];category=(source['sourceCategory'],source['kind'],target['sourceCategory'],target['kind']);pool=self.engine.pools.get(category)
            if pool is None:continue
            qi=self.engine.ix[top]
            a=(self.engine.vectors[qi]@self.engine.vectors[pool['ti']].T+(self.engine.lex[qi]@self.engine.lex[pool['ti']].T).toarray().ravel())/2
            b=(self.engine.vectors[ci]@self.engine.vectors[pool['bi']].T+(self.engine.lex[ci]@self.engine.lex[pool['bi']].T).toarray().ravel())/2
            order=np.argsort(-a,kind='stable')[:min(self.config['k'],len(a))];eligible=np.flatnonzero(np.isin(pool['t'],order));strength=a[pool['t'][eligible]]+b[pool['b'][eligible]]
            for j in eligible[np.argsort(-strength,kind='stable')[:2]]:
                row=pool['rows'][int(j)];result.append(dict(contextId=top,source=SOURCE,sourceOutfits=list(dict.fromkeys(row['outfits']))[:3],top=dict(id=row['top'],title=self.items[row['top']]['title']),bottom=dict(id=row['bottom'],title=self.items[row['bottom']]['title']),relation='원자료에서 함께 코디된 의류; 입력 의류와의 유사도를 기준으로 검색'))
        return result

    def summary(self):
        r=self.report;return dict(version=self.version,source=SOURCE,corpus=dict(verifiedMenswearItems=len(set(self.canonical.values())),canonicalPairsBeforeAgeAudit=106393,referencePairs=len(self.engine.records),ageScopeExclusions=len(self.items)-len(set(self.canonical.values()))),
            validation=dict(cases=r['cases'],garmentDisjoint=r['garmentDisjointScenarios'],selectedRate=r['rates']['proposed']['mean'],strongestComparisonRate=r['rates']['strongest_C']['mean'],difference=r['comparisons']['strongest_C'],criterionMet=r['criterionMet']),
            method=dict(retrieval=self.config,alpha=self.profile['policy']['alpha'],formula='새로 더해지는 조합 근거를 우선하고 기존 대안을 비교'),inputRequirement='원자료의 상품 제목과 종류가 확인된 의류',aestheticProbability=False)

    def examples(self):
        kinds=dict(tee='티셔츠',shirt='셔츠',hoodie='스웨트셔츠',knit='니트',jacket='재킷',blazer='블레이저',trousers='팬츠',jeans='청바지',shorts='반바지')
        def public(pid):
            x=self.items[pid];colours=[]
            for word,label in [('黑色','블랙'),('白色','화이트'),('灰色','그레이'),('蓝色','블루'),('藏青','네이비'),('卡其','카키'),('米色','베이지')]:
                if word in x['title']:colours.append(label)
            return dict(id=pid,label='·'.join(colours)+' '+kinds.get(x['kind'],'의류') if colours else kinds.get(x['kind'],'의류'),originalTitle=x['title'],audience='mens')
        examples=[]
        for n,c in enumerate(read(self.out/'cohorts.json')['development'][:3]):
            examples.append(dict(id=f'development-{n+1}',title=f'코디 기록 비교 {n+1}',request={k:c[k] for k in ['candidates','contexts','owned']},garments={k:[public(p) for p in c[k]] for k in ['candidates','contexts','owned']},basis='개발 자료에서 가져온 실제 원자료 비교 예시'))
        return dict(examples=examples,source=SOURCE)
