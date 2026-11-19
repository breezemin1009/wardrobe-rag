"""Serve the prospectively validated, source-backed history residual.

Owned coverage uses available original co-outfit records, not an estimated
similarity threshold. Unknown candidate relations use the frozen RAG kernel.
"""
from pathlib import Path
from threading import RLock
import json
import numpy as np
from backend.menswear_case_utility import MenswearCaseUtility,SOURCE
from backend.pog_refined_retrieval import Retriever
from backend.pog_cold_study import read,sha

DEFAULT=Path(__file__).parent/'data/menswear-expansion-v3/pog/evidence-history-serving-20261026'

def history_residual(support,history,candidates,contexts,owned):
    """Average predicted new links after removing source-confirmed coverage."""
    ci=list(candidates);qs=list(dict.fromkeys(contexts));wi=list(dict.fromkeys(owned))
    if not ci or not qs:raise ValueError('Candidates and contexts required')
    base=support[np.ix_(ci,qs)]
    if not np.isfinite(base).all() or (base<0).any() or (base>1).any():raise ValueError('Invalid support')
    old=history[np.ix_(wi,qs)] if wi else np.zeros((0,len(qs)),dtype=np.int64)
    if not np.isin(old,[0,1]).all():raise ValueError('Invalid source coverage')
    covered=old.max(0) if wi else np.zeros(len(qs),dtype=np.int64)
    C=base.mean(1);M=(base*(1-covered)).mean(1) if covered.any() else C.copy()
    for j,c in enumerate(ci):
        if c in wi:M[j]=0.
    return C,M,covered

class EvidenceHistoryRetriever(Retriever):
    def __init__(self,out=DEFAULT):
        super().__init__(out);self.confirmed_pairs={(r['top'],r['bottom']) for r in self.records}

    def matrix(self,contexts,candidates,config):
        contexts=list(dict.fromkeys(contexts));candidates=list(candidates)
        S=super().matrix(contexts,candidates,config)
        # Preserve facts for catalogued candidates too. Evaluation candidates
        # are source-cold, so this does not change their frozen test scores.
        for i,b in enumerate(candidates):
            for j,q in enumerate(contexts):
                if (q,b) in self.confirmed_pairs:S[i,j]=1.
        return S

    def features(self,case,config):
        qs=list(dict.fromkeys(case['contexts']));cs=list(case['candidates']);ws=list(dict.fromkeys(case['owned']));bs=cs+[w for w in ws if w not in cs];ix={p:i for i,p in enumerate(bs)}
        S=self.matrix(qs,bs,config).astype(np.float32)
        H=np.asarray([[int((q,b) in self.confirmed_pairs) for q in qs] for b in bs],dtype=np.int64)
        C,M,covered=history_residual(S,H,[ix[p] for p in cs],list(range(len(qs))),[ix[p] for p in ws])
        return {'C':C,'verified-history-residual':M,'confirmedContextMask':covered}

class EvidenceHistoryCaseUtility(MenswearCaseUtility):
    def __init__(self,out=DEFAULT):
        self.out=Path(out);manifest=read(self.out/'deployment-manifest.json')
        for name,digest in manifest['files'].items():
            if sha(self.out/name)!=digest:raise ValueError('Verified history artifact changed: '+name)
        for name,digest in manifest['implementationHashes'].items():
            if sha(Path(__file__).parent/name)!=digest:raise ValueError('Verified history implementation changed: '+name)
        self.report=read(self.out/'report.json');self.protocol=read(self.out/'protocol.json');self.selection=read(self.out/'selection-lock.json')
        if not self.report['adoptionCriterionMet']:raise ValueError('History transfer adoption not verified')
        for name in ['independent-source-audit.json','independent-result-audit.json']:
            if not read(self.out/name)['allPassed']:raise ValueError('History transfer audit failed')
        if self.report['protocolSha256']!=sha(self.out/'protocol.json') or self.report['selectionSha256']!=sha(self.out/'selection-lock.json'):raise ValueError('History transfer binding changed')
        chosen=self.selection['selected'];self.config=read(self.out/'test-feature-lock.json')['configs'][chosen['config']]
        if chosen['method']!='history' or chosen['alpha']!=0:raise ValueError('Unsupported history deployment policy')
        self.lock=RLock();self.engine=EvidenceHistoryRetriever(self.out);self.items=self.engine.items
        self.canonical={raw:p for p,x in self.items.items() for raw in x['aliases'] if p in self.engine.ix}
        self.profile={'policy':{'alpha':0,'marginal':'verified-history-residual'}}
        self.version='case-rag-verified-history-v3-'+sha(self.out/'selection-lock.json')[:12]

    def assess(self,*args,**kwargs):
        result=super().assess(*args,**kwargs);eligible=[r for r in result['results'] if not r['alreadyOwned']]
        for row in result['results']:
            row['eligibleForPurchase']=not row['alreadyOwned']
            if row['alreadyOwned']:row['score']=0.;row['additionalContribution']=0.;row['rank']=None
            else:row['rank']=1+sum(r['score']>row['score']+1e-8 for r in eligible)
        result['results'].sort(key=lambda r:(r['alreadyOwned'],-r['score'],r['id']))
        result['method'].update(verifiedExistingHistory=True,formula='기존 옷의 코디 기록으로 이미 지원하는 상의를 구분하고, 나머지 상의에 대한 구매 후보의 지원을 평균',historySource='현재 검색 자료에 실제로 기록된 상의·기존 하의 연결')
        result['meaning'].update(combinationSupport='구매 후보가 내 상의들과 조합되는 근거의 평균',additionalContribution='기존 옷의 코디 기록이 채우지 못한 상의에 후보가 더하는 조합 근거',scale='0~100 근거 지수; 구매 만족 확률이 아님')
        return result

    def summary(self):
        r=self.report
        return dict(version=self.version,source=SOURCE,corpus=dict(verifiedMenswearItems=len(set(self.canonical.values())),referencePairs=len(self.engine.records),ageScopeExclusions=len(self.items)-len(set(self.canonical.values()))),
            validation=dict(cases=r['parents'],garmentDisjoint=r['parents'],sourceOutfitDisjoint=True,selectedRate=r['rates']['proposed']['mean'],strongestComparisonRate=r['rates']['development-selected-C']['mean'],difference=r['contrasts']['development-selected-C'],criterionMet=r['adoptionCriterionMet'],metric='정규화 추가 관측 조합 획득 수준',candidateCount=4,scope='기존 옷의 코디 기록을 확인할 수 있는 옷장에서 새 구매 후보를 비교한 고정 설정 시험'),
            method=dict(retrieval=self.config,alpha=0,verifiedExistingHistory=True,formula='기존 옷으로 이미 지원하는 상의를 구분해 추가 조합 근거를 계산'),inputRequirement='원상품 정보와 기존 코디 기록이 연결된 의류',aestheticProbability=False)

    def examples(self):
        # Reference-only illustrations; never mined from held-out outcomes.
        return read(self.out/'serving-examples.json')
