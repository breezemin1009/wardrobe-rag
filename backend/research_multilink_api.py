"""Source-constructed two-candidate study, with partial visible history allowed."""
from functools import lru_cache
from threading import RLock
from pathlib import Path
import joblib
from fastapi import FastAPI,HTTPException
from backend.research_wardrobe_api import WardrobeResearchUtility,Query
from backend import thesis_multilink_source_20261027 as study
from backend import thesis_repair_models_20261026 as b
from backend.pog_refined_retrieval import Retriever

class MultilinkResearchUtility(WardrobeResearchUtility):
    def __init__(self,out=study.OUT):
        self.out=out;self.selection=b.read(out/'selection-lock.json');self.report=b.read(out/'test-results.json')
        if self.report['selectionSha256']!=b.sha(out/'selection-lock.json'):raise ValueError('Selection changed')
        if not b.read(out/'raw-source-audit.json')['allPassed']:raise ValueError('Source audit missing')
        if not b.read(out/'result-audit.json')['allPassed']:raise ValueError('Result audit missing')
        self.profile=self.selection['profiles']['proposed'];path=out/(self.profile['tag']+'.joblib')
        if b.sha(path)!=self.profile['modelSha256']:raise ValueError('Model changed')
        self.model=joblib.load(path);self.engine=Retriever(out);self.items=self.engine.items;self.lock=RLock()
        self._cached_support=lru_cache(maxsize=65536)(self._point_support)
        self.canonical={a:p for p,x in self.items.items() if p in self.engine.ix for a in [p]+x['aliases']}
        self.record_lookup={(r['top'],r['bottom']):r for r in self.engine.records};self.confirmed=set(self.record_lookup)
        # An owned top can lack visible reference outfits. Its withheld target
        # relationships remain labels, never a prerequisite supplied to inference.
        self.known_tops={p for p,x in self.items.items() if x['side']=='top' and p in self.engine.ix}
        self.visible_tops={q for q,w in self.confirmed}
        self.version='source-rich-multilink-v1-'+b.sha(out/'selection-lock.json')[:12]

    def assess(self,candidates,contexts,owned,include_evidence=True):
        response=super().assess(candidates,contexts,owned,include_evidence)
        response['mode']='research-source-constructed-multilink-comparison'
        response['meaning']['input']='실제 상품의 보유 상의·기존 하의와 새 후보; 보유 상의의 공개 기록은 일부만 보일 수 있음'
        response['visibleReferenceContextMask']=[int(p in self.visible_tops) for p in response['contexts']]
        return response

@lru_cache(maxsize=1)
def engine():return MultilinkResearchUtility()

app=FastAPI(title='Source Constructed Multilink Research API')
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
