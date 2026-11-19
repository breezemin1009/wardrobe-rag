"""Deterministic sensitivity audit; neither confidence nor aesthetic accuracy.

One source family is removed at a time. Scores are compared only on the same
computed group mask and design scope; lost coverage is reported separately.
"""
import hashlib
import json
from functools import lru_cache
from backend.config import ALPHA, MIN_COVERAGE, VERSION
from backend.design_reasoning import rulebook
from backend.scoring import evaluate

ALPHAS = (.5, ALPHA, .8)
COVERAGES = (.5, MIN_COVERAGE, 1.0)


def family(record):
    return record.get('sourceFamily', record['sourceId'])


class EvidenceSubset:
    def __init__(self, retriever, excluded_family):
        self.scoped = getattr(retriever, 'scoped', False)
        self.chunks = [r for r in retriever.chunks if r['collection'] != 'style' or family(r) != excluded_family]
        self.manifest = retriever.manifest
        material = json.dumps({'parent': retriever.index['chunksSha256'], 'excludedFamily': excluded_family}, sort_keys=True)
        self.index = {**retriever.index, 'chunksSha256': hashlib.sha256(material.encode()).hexdigest()}


@lru_cache(maxsize=16)
def without_family(retriever, excluded_family):
    return EvidenceSubset(retriever, excluded_family)


def computed_mask(result):
    return result['assessmentScope'], tuple(sorted(r['groupId'] for r in result['groups'] if r['gain'] is not None))


def comparison(base, alternative):
    same_scope = computed_mask(base) == computed_mask(alternative)
    comparable = base['score'] is not None and alternative['score'] is not None and same_scope
    return {'score': alternative['score'], 'status': alternative['status'],
            'coverage': alternative['coverage'], 'gainCoverage': alternative['gainCoverage'],
            'comparableGroups': alternative['counts']['comparableGroups'],
            'sameComputedScope': same_scope, 'comparableToBase': comparable,
            'scoreDelta': round(alternative['score'] - base['score'], 1) if comparable else None}


def validate_snapshot(snapshot, retriever):
    if not snapshot.get('input') or 'groups' not in snapshot:
        raise ValueError('의류 평가 기록을 선택해주세요.')
    if (snapshot.get('version') != VERSION or snapshot.get('corpusSha256') != retriever.index['chunksSha256']
            or snapshot.get('indexRevision') != retriever.index['revision']
            or snapshot.get('method', {}).get('designRulesSha256') != rulebook()[1]):
        raise ValueError('자료·계산 버전이 바뀌었어요. 현재 옷장으로 다시 평가한 후 확인해주세요.')


def audit_evaluation(snapshot, retriever):
    validate_snapshot(snapshot, retriever)
    request = snapshot['input']
    candidate, wardrobe, occasion = request['candidate'], request['wardrobe'], request['occasion']
    method = snapshot['method']
    def calculate(source=retriever, alpha=method['alpha'], threshold=method['minimumCoverage']):
        return evaluate(candidate, wardrobe, occasion, source, method['retrieval'], alpha, threshold)
    base = calculate()
    if base['evaluationFingerprint'] != snapshot.get('evaluationFingerprint') or base['score'] != snapshot.get('score'):
        raise ValueError('저장한 평가를 같은 버전에서 재현하지 못했어요. 다시 평가해주세요.')
    removals = []
    families = sorted({family(r) for r in retriever.chunks if r['collection'] == 'style'})
    for name in families:
        alternative = calculate(without_family(retriever, name))
        removals.append({'removedFamily': name, **comparison(base, alternative)})
    parameters = []
    for threshold in COVERAGES:
        for alpha in ALPHAS:
            parameters.append({'alpha': alpha, 'minimumCoverage': threshold,
                               **comparison(base, calculate(alpha=alpha, threshold=threshold))})
    comparable = [r for r in removals if r['comparableToBase']]
    return {'runId': snapshot['runId'], 'version': VERSION,
            'evaluationFingerprint': base['evaluationFingerprint'], 'corpusSha256': retriever.index['chunksSha256'],
            'designRulesSha256': rulebook()[1], 'scope': 'deterministic-engineering-sensitivity',
            'base': {'score': base['score'], 'status': base['status'], 'coverage': base['coverage'],
                     'gainCoverage': base['gainCoverage'], 'comparableGroups': base['counts']['comparableGroups']},
            'sourceRemoval': removals, 'parameters': parameters,
            'summary': {'familiesTested': len(removals), 'comparableRemovals': len(comparable),
                        'scopeChangedRemovals': sum(not r['sameComputedScope'] for r in removals),
                        'withheldRemovals': sum(r['score'] is None for r in removals),
                        'newlyWithheldRemovals': sum(base['score'] is not None and r['score'] is None for r in removals),
                        'maxAbsoluteComparableDelta': max((abs(r['scoreDelta']) for r in comparable), default=None)},
            'limitations': ['원 출처 계열을 한 번에 하나씩 제외한 민감도 검사이며 신뢰구간이나 어울림 정확도가 아닙니다.',
                            '점수 차이는 같은 의류 그룹·디자인 범위에서만 비교합니다. 범위가 바뀌거나 보류되면 차이를 계산하지 않습니다.',
                            '가중치·보류 기준은 사전에 정한 개발 비교 값이며 가장 좋은 값이라고 선택한 결과가 아닙니다.',
                            '출처 사이의 재인용·공통 편향과 여러 출처가 동시에 없어지는 경우까지 검사하지 않습니다.']}
