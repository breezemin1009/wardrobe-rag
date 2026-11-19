"""Deploy the frozen, independently tested 2026-10-24 support model.

Scores still use the same C and exact positive max-coverage increment M.
Training/development choose support models and weights; serving never refits.
"""
from pathlib import Path
from threading import RLock
import hashlib,json
import numpy as np
import joblib
from backend.menswear_case_utility import MenswearCaseUtility
from backend.pog_refined_retrieval import Retriever
from backend.pog_cold_study import read,sha

ROOT=Path(__file__).parent/'data/menswear-expansion-v3/pog'
DEFAULT=ROOT/'prospective-upgrade-20261024'
CONFIGS=[dict(mode=m,k=k) for m in ['char-density','hybrid-density','hybrid-conditional'] for k in [8,32,96]]

class LearnedSupportRetriever(Retriever):
    def __init__(self,out=DEFAULT):
        super().__init__(out);out=Path(out)
        if (out/'support-lock.json').exists():
            lock=read(out/'support-lock.json')
            paths=['retrieval-linear.joblib','retrieval-boosted.joblib','text-linear.joblib','text-boosted.joblib']
            expected=[lock['models'][name] for name in paths];self.weights=np.asarray(lock['weights'])
        else:
            calibration=read(out/'calibration-lock.json');no_rag=read(out/'no-rag-selection.json')
            paths=['pair-linear-model.joblib','pair-boosted-model.joblib','no-rag-pair-linear.joblib','no-rag-pair-boosted.joblib']
            expected=[calibration['models']['linear']['modelSha256'],calibration['models']['boosted']['modelSha256'],
                      no_rag['modelHashes']['no-rag-pair-linear.joblib'],no_rag['modelHashes']['no-rag-pair-boosted.joblib']]
            self.weights=np.asarray(read(out/'support-mixture-selection.json')['profiles']['upgraded']['weights'])
        assert len(self.weights)==4 and np.isclose(self.weights.sum(),1) and (self.weights>=0).all()
        self.support_models=[]
        for name,value,w in zip(paths,expected,self.weights):
            if sha(out/name)!=value:raise ValueError('Frozen support model changed')
            self.support_models.append(joblib.load(out/name) if w else None)
        expected_pca=read(out/'training-pair-feature-lock.json')['pcaSha256']
        if sha(out/'reference-pca.joblib')!=expected_pca:raise ValueError('Frozen text projection changed')
        self.projection=joblib.load(out/'reference-pca.joblib').transform(self.vectors)
        # Provenance is not a model feature. Only reference-edge origins are
        # retained for citations; no query target labels are consulted.
        origins=read(ROOT/'adversarial-audit-20261023/source-edge-outfits.json')
        for row in self.records:row['outfits']=origins[row['top']+':'+row['bottom']]

    def matrix(self,contexts,candidates,config=None):
        contexts=list(dict.fromkeys(contexts));candidates=list(candidates)
        qs=[self.ix[p] for p in contexts];cs=[self.ix[p] for p in candidates];n=len(cs);q=len(qs)
        matrices=[super(LearnedSupportRetriever,self).matrix(contexts,candidates,cfg) for cfg in CONFIGS]
        lex=(self.lex[cs]@self.lex[qs].T).toarray();dense=self.vectors[cs]@self.vectors[qs].T
        top=np.broadcast_to(self.projection[qs][None,:,:],(n,q,32));bottom=np.broadcast_to(self.projection[cs][:,None,:],(n,q,32))
        X=np.concatenate([np.stack(matrices+[lex,dense],axis=-1),top,bottom,np.abs(top-bottom),top*bottom],axis=-1).astype(np.float32)
        value=np.zeros((n,q),dtype=np.float64)
        for index,(weight,model) in enumerate(zip(self.weights,self.support_models)):
            if weight:
                features=X if index<2 else X[:,:,9:]
                value+=weight*model.predict_proba(features.reshape(-1,features.shape[-1]))[:,1].reshape(n,q)
        if not np.isfinite(value).all() or (value<0).any() or (value>1).any():raise ValueError('Invalid learned support')
        return value

class LearnedCaseUtility(MenswearCaseUtility):
    def __init__(self,out=DEFAULT):
        self.out=Path(out);self.lock=RLock();self.report=read(self.out/'final-report.json');self.protocol=read(self.out/'protocol.json')
        self.final_plan=read(self.out/'final-plan.json')
        if not self.report['adoptionCriterionMet'] or self.report['finalPlanSha256']!=sha(self.out/'final-plan.json'):
            raise ValueError('Final learned-support adoption is not verified')
        for name,value in self.final_plan['files'].items():
            if sha(self.out/name)!=value:raise ValueError('Frozen selection or protocol changed')
        self.engine=LearnedSupportRetriever(self.out);self.items=self.engine.items
        self.profile={'policy':{'alpha':read(self.out/'support-mixture-selection.json')['profiles']['upgraded']['alpha'],'marginal':'positive-difference'}}
        self.config={'mode':'learned-support-mixture','k':32}
        from backend.prepare_pog_menswear import classify
        self.canonical={p:k for k,v in self.items.items() if classify(v['title'])[0] is not None for p in v['aliases']}
        self.version='case-rag-learned-support-v2-'+self.report['finalPlanSha256'][:12]

    def assess(self,*args,**kwargs):
        result=super().assess(*args,**kwargs)
        result['method'].update(supportWeights=self.engine.weights.tolist(),
            supportModels=['검색 근거·상품 표현의 선형 모델','검색 근거·상품 표현의 트리 모델','상품 표현의 선형 모델','상품 표현의 트리 모델'],
            supportTraining='학습 자료에서 조합 지원을 추정하고 개발 자료에서 가중치를 고정')
        result['meaning']['combinationSupport']='검색한 코디 근거와 상품 정보로 추정한 보유 상의별 조합 지원의 평균'
        return result

    def summary(self):
        r=self.report;reference='https://github.com/wenyuer/POG'
        return dict(version=self.version,source=reference,
            corpus=dict(verifiedMenswearItems=len(set(self.canonical.values())),canonicalPairsBeforeAgeAudit=106362,
                        referencePairs=len(self.engine.records),ageScopeExclusions=len(self.items)-len(set(self.canonical.values()))),
            validation=dict(cases=r['n'],garmentDisjoint=r['n'],sourceOutfitDisjoint=True,
                selectedRate=r['rates']['upgraded']['mean'],strongestComparisonRate=r['rates']['development-selected-C']['mean'],
                difference=r['comparisons']['development-selected-C'],criterionMet=r['adoptionCriterionMet'],
                metric='정규화 추가 관측 연결 획득률',candidateCount=8,scope='학습·개발과 분리한 내부 최종 시험'),
            method=dict(retrieval=self.config,alpha=self.profile['policy']['alpha'],supportWeights=self.engine.weights.tolist(),
                        formula='같은 조합 지원에서 전체 활용도와 기존 옷 대비 추가 활용을 나누어 계산'),
            inputRequirement='원자료의 상품 제목과 종류가 확인된 의류',aestheticProbability=False)

    def examples(self):
        curated=self.out/'serving-examples.json'
        if curated.exists():return read(curated)['public']
        inputs=read(self.out/'development-inputs.json');targets=read(self.out/'development-targets.json')
        kinds=dict(tee='티셔츠',shirt='셔츠',hoodie='스웨트셔츠',knit='니트',jacket='재킷',blazer='블레이저',trousers='바지',jeans='청바지',shorts='반바지')
        def public(pid):
            x=self.items[pid];colours=[]
            for word,label in [('黑色','블랙'),('白色','화이트'),('灰色','그레이'),('蓝色','블루'),('藏青','네이비'),('卡其','카키'),('米色','베이지')]:
                if word in x['title']:colours.append(label)
            return dict(id=pid,label=('·'.join(colours)+' ' if colours else '')+kinds.get(x['kind'],'의류'),originalTitle=x['title'],audience='mens')
        examples=[]
        for i,(c,t) in enumerate(zip(inputs,targets)):
            o=np.asarray(t['observed']);hist=np.asarray(t['history']);gain=(o*(1-hist.max(0))).sum(1)
            # An explicitly labelled development illustration, not a new test.
            high=int(np.argmax(gain));low=int(np.argmin(gain))
            if high==low:continue
            request={k:c[k] for k in ['contexts','owned']};request['candidates']=[c['candidates'][low],c['candidates'][high]]
            examples.append(dict(id=f'learned-development-{i+1}',title=f'코디 기록 비교 {len(examples)+1}',request=request,
                garments={k:[public(p) for p in request[k]] for k in ['candidates','contexts','owned']},basis='가중치 선택에 사용한 개발 자료의 실제 비교 예시'))
            if len(examples)==3:break
        return dict(examples=examples,source='https://github.com/wenyuer/POG')
