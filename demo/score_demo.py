"""Offline example: run the ORIGINAL aggregation with supplied support values.

The binary support matrix below is an explanatory fixture, not an ML prediction
or a replication of the paper's experiments. No dataset or external API is used.
"""
import json
from threading import RLock
from types import SimpleNamespace
import numpy as np
from backend.photo_case_assessment import PhotoCaseAssessment

TOPS = ['top-1', 'top-2', 'top-3', 'top-4']
OWNED = ['owned-1', 'owned-2']
SUPPORT = {
    'candidate-a': [1, 1, 1, 0],
    'candidate-b': [1, 0, 0, 1],
    'owned-1': [1, 1, 0, 0],
    'owned-2': [0, 0, 1, 0],
}

class FixtureAssessment(PhotoCaseAssessment):
    def prepare(self, garments, encoder=None):
        ids = {g['id']: g['id'] for g in garments}
        maps = {g['id']: {'mode':'exact','products':[{'id':g['id'], 'title':g['name'], 'similarity':1.0}]}
                for g in garments}
        return self.base, ids, maps

    def matrix(self, engine, tops, bottoms):
        return np.asarray([[SUPPORT[p][TOPS.index(t)] for t in tops] for p in bottoms], dtype=float)

def run():
    utility = SimpleNamespace(
        engine=SimpleNamespace(), lock=RLock(), version='offline-fixture-v1',
        canonical={}, confirmed={('top-1','owned-1'),('top-2','owned-1'),('top-3','owned-2')},
        evidence=lambda candidate, contexts: [],
    )
    service = FixtureAssessment(utility)
    garment = lambda pid, side: {'id':pid,'referenceProductId':pid,'name':pid,'category':side,'color':'화이트','pattern':'무지'}
    wardrobe = [garment(t,'상의') for t in TOPS] + [garment(w,'하의') for w in OWNED]
    results = []
    for candidate in ['candidate-a','candidate-b']:
        result=service.assess(garment(candidate,'하의'), wardrobe)
        results.append({'candidate':candidate,'C':result['combination'],'M':result['contribution']})
    return results

if __name__ == '__main__':
    print(json.dumps({'mode':'explanatory-fixture','results':run()},ensure_ascii=False,indent=2))
