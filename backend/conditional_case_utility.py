"""Serve the sealed conditional additional-link estimator without refitting.

The support and conditional heads use title/retrieval features only. Ground
truth links are used by offline evaluation, never by this serving adapter.
"""
from pathlib import Path
from threading import RLock
import numpy as np
import joblib
from backend.menswear_case_utility import MenswearCaseUtility
from backend.pog_refined_retrieval import Retriever
from backend.pog_cold_study import read, sha

DEFAULT = Path(__file__).parent/'data/menswear-expansion-v3/pog/conditional-final-20261025'
CONFIGS = [dict(mode=m, k=k) for m in ['char-density','hybrid-density','hybrid-conditional'] for k in [8,32,96]]


class ConditionalRetriever(Retriever):
    def __init__(self, out=DEFAULT):
        super().__init__(out)
        self.out = Path(out)
        selected = read(self.out/'method-selection.json')['profiles']['RAG']
        self.weights = np.asarray(selected['supportWeights'])
        support = read(self.out/'support-lock.json')
        self.models = []
        for name in ['retrieval-linear.joblib','retrieval-boosted.joblib','text-linear.joblib','text-boosted.joblib']:
            if sha(self.out/name) != support['models'][name]:
                raise ValueError('Frozen support model changed')
            self.models.append(joblib.load(self.out/name))
        if sha(self.out/selected['model']) != selected['sha256']:
            raise ValueError('Frozen conditional model changed')
        self.conditional = joblib.load(self.out/selected['model'])
        if sha(self.out/'reference-pca.joblib') != read(self.out/'training-pair-feature-lock.json')['pcaSha256']:
            raise ValueError('Frozen projection changed')
        self.projection = joblib.load(self.out/'reference-pca.joblib').transform(self.vectors)
        root = self.out.parent/'adversarial-audit-20261023/source-edge-outfits.json'
        origins = read(root)
        for record in self.records:
            record['outfits'] = origins[record['top']+':'+record['bottom']]

    def pair_features(self, contexts, bottoms):
        qs = [self.ix[p] for p in contexts]
        cs = [self.ix[p] for p in bottoms]
        n, q = len(cs), len(qs)
        kernels = [super(ConditionalRetriever,self).matrix(contexts,bottoms,cfg) for cfg in CONFIGS]
        lex = (self.lex[cs]@self.lex[qs].T).toarray()
        dense = self.vectors[cs]@self.vectors[qs].T
        top = np.broadcast_to(self.projection[qs][None,:,:],(n,q,32))
        bottom = np.broadcast_to(self.projection[cs][:,None,:],(n,q,32))
        X = np.concatenate([np.stack(kernels+[lex,dense],axis=-1),top,bottom,np.abs(top-bottom),top*bottom],axis=-1).astype(np.float32)
        # Match the float32 probability cache used in the frozen experiment.
        P = np.stack([m.predict_proba((X if i<2 else X[:,:,9:]).reshape(-1,139 if i<2 else 130))[:,1].reshape(n,q).astype(np.float32) for i,m in enumerate(self.models)],axis=-1)
        return X,P

    def features(self, case, config=None):
        from backend.thesis_conditional_models_20261025 import marginal_estimates
        contexts = list(dict.fromkeys(case['contexts']))
        candidates = list(dict.fromkeys(case['candidates']))
        owned = list(dict.fromkeys(case['owned']))
        bottoms = list(dict.fromkeys(candidates+owned))
        X,P = self.pair_features(contexts,bottoms)
        ci = [bottoms.index(p) for p in candidates]
        wi = [bottoms.index(p) for p in owned]
        indices = [self.ix[p] for p in bottoms]
        lex = np.clip((self.lex[indices]@self.lex[indices].T).toarray(),0,1)
        dense = np.clip(self.vectors[indices]@self.vectors[indices].T,0,1)
        C,M,_ = marginal_estimates(self.conditional,X,P,ci,wi,list(range(len(contexts))),lex[np.ix_(ci,wi)],dense[np.ix_(ci,wi)],self.weights)
        return {'C':C,'conditional-addition':M}


class ConditionalCaseUtility(MenswearCaseUtility):
    def __init__(self, out=DEFAULT):
        self.out = Path(out)
        adoption = read(self.out/'adoption-report.json')
        plan = read(self.out/'final-plan.json')
        if not adoption['adoptionCriterionMet'] or adoption['finalPlanSha256'] != sha(self.out/'final-plan.json'):
            raise ValueError('Conditional method has not passed its frozen adoption plan')
        for name,value in plan['files'].items():
            if sha(self.out/name) != value:
                raise ValueError('Frozen conditional artifact changed: '+name)
        for name,key in [('final-report.json','broadReportSha256'),('strict-final-report.json','strictReportSha256')]:
            if sha(self.out/name) != adoption[key]:
                raise ValueError('Frozen outcome changed')
        encoder = read(self.out/'encoder.json')
        for name,key in [('vectors.npy','vectorsSha256'),('itemIds.json','idsSha256')]:
            if sha(self.out/name) != encoder[key]:
                raise ValueError('Frozen encoder changed')
        self.lock = RLock()
        self.report = read(self.out/'final-report.json')
        self.protocol = read(self.out/'protocol.json')
        self.engine = ConditionalRetriever(self.out)
        self.items = self.engine.items
        selected = read(self.out/'method-selection.json')['profiles']['RAG']
        self.profile = {'policy':{'alpha':selected['alpha'],'marginal':'conditional-addition'}}
        self.config = {'mode':'conditional-additional-links','k':32}
        from backend.prepare_pog_menswear import classify
        self.canonical = {p:k for k,v in self.items.items() if classify(v['title'])[0] is not None for p in v['aliases']}
        self.version = 'conditional-additional-links-v4-'+adoption['finalPlanSha256'][:12]

    def assess(self, *args, **kwargs):
        result = super().assess(*args, **kwargs)
        for row in result['results']:
            row['eligibleForPurchase'] = not row['alreadyOwned']
            if row['alreadyOwned']:
                row['additionalContribution'] = 0.
                row['score'] = 0.
        result['results'].sort(key=lambda r:(r['alreadyOwned'],-r['score'],r['id']))
        eligible = [r for r in result['results'] if not r['alreadyOwned']]
        for row in result['results']:
            row['rank'] = None if row['alreadyOwned'] else 1+sum(r['score']>row['score']+1e-8 for r in eligible)
        result['method'].update(formula='추천 지수 = 조합 활용도 × 0.1 + 추가 코디 기여 × 0.9',ranking='개발 자료에서 고정한 두 점수의 가중합',supportWeights=self.engine.weights.tolist())
        result['meaning'].update(combinationSupport='후보가 내 상의들과 조합된 공개 기록을 가질 것으로 추정한 정도의 평균',additionalContribution='그중 기존 하의로 채워지지 않는 조합 기록을 더할 것으로 추정한 정도의 평균',scale='0~100 지수; 실제 기록의 추정치이며 미적 만족 확률을 뜻하지 않음')
        return result

    def summary(self):
        r = self.report
        return dict(version=self.version,source='https://github.com/wenyuer/POG',
            corpus=dict(verifiedMenswearItems=len(set(self.canonical.values())),referencePairs=len(self.engine.records)),
            validation=dict(cases=r['n'],selectedRate=r['rates']['proposed']['mean'],strongestComparisonRate=r['rates']['development-selected-C']['mean'],difference=r['comparisons']['development-selected-C'],criterionMet=True,metric='정규화 추가 관측 조합 획득 수준',scope='내부 신규 후보와 입력 크기 변화 평가; 이전 실험의 코디 출처가 재등장할 수 있음'),
            method=dict(alpha=self.profile['policy']['alpha'],formula='0.1C+0.9M'),inputRequirement='원자료의 상품 제목과 종류가 확인된 의류',aestheticProbability=False)

    def examples(self):
        rows = read(self.out/'development-frames.json')
        kinds = dict(tee='티셔츠',shirt='셔츠',hoodie='스웨트셔츠',knit='니트',jacket='재킷',blazer='블레이저',trousers='바지',jeans='청바지',shorts='반바지')
        def public(pid):
            x = self.items[pid]
            colors = [label for word,label in [('黑色','블랙'),('白色','화이트'),('灰色','그레이'),('蓝色','블루'),('藏青','네이비'),('卡其','카키'),('米色','베이지')] if word in x['title']]
            return dict(id=pid,label=('·'.join(colors)+' ' if colors else '')+kinds.get(x['kind'],'의류'),originalTitle=x['title'],audience='mens')
        examples = []
        for c in rows[:3]:
            v = c['views']['rich']
            request = dict(contexts=[c['contexts'][i] for i in v['q']],candidates=[c['bottoms'][i] for i in v['c']],owned=[c['bottoms'][i] for i in v['w']])
            examples.append(dict(id=c['id'],title=f'코디 기록 비교 {len(examples)+1}',request=request,garments={k:[public(p) for p in request[k]] for k in request},basis='개발 자료의 실제 의류 비교 예시'))
        return dict(examples=examples,source='https://github.com/wenyuer/POG')
