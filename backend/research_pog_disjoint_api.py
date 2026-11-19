"""Frozen POG source-disjoint comparison; its support model is also used by the confirmed-attribute app path."""
from functools import lru_cache
from fastapi import FastAPI,HTTPException
from backend.research_multilink_api import MultilinkResearchUtility
from backend.research_wardrobe_api import Query
from backend import thesis_pog_disjoint_study_20261027 as study
from backend import thesis_repair_models_20261026 as b
from backend.wardrobe_case_explanation_v2 import explain_case
SOURCE='https://github.com/wenyuer/POG'

class POGDisjointUtility(MultilinkResearchUtility):
    def __init__(self,out=study.OUT):
        super().__init__(out)
        self.version='pog-full-source-disjoint-v1-'+b.sha(out/'selection-lock.json')[:12]

    def evidence(self,candidate,contexts):
        rows=super().evidence(candidate,contexts)
        for row in rows:row['source']=SOURCE
        return rows

    def assess(self,candidates,contexts,owned,include_evidence=True):
        result=super().assess(candidates,contexts,owned,include_evidence)
        result['mode']='research-pog-source-disjoint-comparison'
        result['method']['source']=SOURCE
        return result

@lru_cache(maxsize=1)
def engine():return POGDisjointUtility()

app=FastAPI(title='POG Disjoint Replication Research API')
@app.get('/study')
def summary():return dict(**engine().summary(),explanationPolicy='v2-evidence-presence-guard')
@app.post('/compare')
def compare(request:Query):
    try:return engine().assess(request.candidates,request.contexts,request.owned)
    except ValueError as e:raise HTTPException(422,str(e)) from None
@app.post('/explain')
def explain(request:Query):
    try:return dict(**explain_case(engine(),request),explanationPolicy='v2-evidence-presence-guard')
    except ValueError as e:raise HTTPException(422,str(e)) from None
