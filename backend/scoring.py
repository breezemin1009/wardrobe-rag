"""Experimental utility, not aesthetic truth or a probability. Unknown != zero."""
from collections import defaultdict
import hashlib
import json
import math
from functools import lru_cache
from backend.config import ALPHA, MIN_COVERAGE, VERSION
from backend.design import canonical as canonical_design
from backend.design_reasoning import assess_design,rulebook

LABELS = {'recommend': 1.0, 'conditional': 0.5, 'conflict': 0.0}

def applicable(evidence, first, second, occasion):
    if evidence.get('requiresDesign'):return False  # Applied only by typed joint reasoning, never by colour alone.
    if first['category'] == second['category']:
        return False
    top, bottom = (first, second) if first['category'] == '상의' else (second, first)
    return (top['color'] in evidence.get('topColors', []) and bottom['color'] in evidence.get('bottomColors', [])
            and top['pattern'] in evidence.get('topPatterns', evidence.get('patterns', []))
            and bottom['pattern'] in evidence.get('bottomPatterns', evidence.get('patterns', []))
            and occasion in evidence.get('occasions', []))

def combine_support(records):
    return support_details(records)['support']

def support_details(records):
    # One vote per publisher / original-source family. Concrete pair evidence
    # takes precedence over a generalized colour rule within that family.
    families = defaultdict(list)
    for item in records:
        if item.get('label') in LABELS:
            families[item.get('sourceFamily', item['sourceId'])].append(item)
    votes = []
    specificity = {'direct_pair': 3, 'partial_outfit': 2, 'color_rule': 1, 'capsule_palette': 1}
    for family, items in sorted(families.items()):
        level = max(specificity.get(i.get('scope', 'direct_pair'), 1) for i in items)
        selected = [i for i in items if specificity.get(i.get('scope', 'direct_pair'), 1) == level]
        # Conservative when equally specific statements disagree.
        votes.append({'family': family, 'support': min(LABELS[i['label']] for i in selected),
                      'evidenceIds': sorted({i['id'] for i in selected}), 'specificity': level})
    value = math.fsum(v['support'] for v in votes) / len(votes) if votes else None
    return {'support': value, 'families': votes, 'familyCount': len(votes),
            'directFamilyCount': sum(v['specificity'] == 3 for v in votes),
            'evidenceCount': len({i['id'] for i in records})}

@lru_cache(maxsize=2048)
def pair_records(candidate_key, owned_key, occasion, retriever, mode):
    candidate = dict(zip(('category', 'color', 'pattern'), candidate_key))
    owned = dict(zip(('category', 'color', 'pattern'), owned_key))
    query = f"{candidate['category']} {candidate['color']} {candidate['pattern']} {owned['category']} {owned['color']} {owned['pattern']} {occasion} 코디 조합"
    # Filter before top-k, and exhaust eligible annotations. An unrelated corpus
    # expansion must not push matching evidence out and silently change a score.
    if hasattr(retriever, 'chunks'):
        # Structured assessment must not depend on a document's lexical score or
        # semantic ranking. Exhaust the condition-matched facts; RAG ranks prose
        # in the separate search/explanation path.
        retrieved = sorted((r for r in retriever.chunks if r['collection']=='style'
                            and applicable(r,candidate,owned,occasion)),key=lambda r:r['id'])
    else:
        retrieved = retriever.search(query, collection='style', mode=mode, limit=10000, source_dedup=False,
                                     predicate=lambda r: applicable(r, candidate, owned, occasion))
    return tuple(r for r in retrieved if applicable(r, candidate, owned, occasion))

def pair_support(candidate, owned, occasion, retriever, mode):
    records = list(pair_records(basic_key(candidate), basic_key(owned), occasion, retriever, mode))
    return combine_support(records), records

@lru_cache(maxsize=32)
def design_sources(retriever):
    # A rule is usable only while its originating source is in this corpus.
    # Legacy test adapters without a corpus retain their explicit rule fixture.
    return frozenset(r['sourceId'] for r in retriever.chunks if r['collection']=='style') if hasattr(retriever,'chunks') else None

def pair_assessment(candidate,owned,occasion,retriever,mode):
    if not candidate.get('design') and not owned.get('design'):
        support,records=pair_support(candidate,owned,occasion,retriever,mode)
        return support,records,None
    return assess_design(candidate,owned,occasion,
        lambda a,b:pair_records(basic_key(a),basic_key(b),occasion,retriever,mode),combine_support,design_sources(retriever))

def basic_key(garment):return (garment['category'],garment['color'],garment['pattern'])

def group_key(garment):
    # Conservative exact attribute groups, not visual embedding similarity.
    return (*basic_key(garment),canonical_design(garment.get('design')) or '')

def group_id(key):
    return ':'.join(key[:3])+((':design-'+hashlib.sha256(key[3].encode()).hexdigest()[:16]) if key[3] else '')

def incremental_support(support, existing):
    """A known alternative can prove zero gain without knowing the exact max.

    H >= max(known alternatives, 0). If E <= that lower bound, max(E-H,0)
    is exactly zero even while other alternatives remain unknown.
    """
    known=[value for value in existing if value is not None]
    lower=max(known,default=0.0)
    prior=0.0 if not existing else 1.0 if 1.0 in known else max(known) if len(known)==len(existing) else None
    if support is None:return prior,None,{'type':'unknown','knownExistingLowerBound':lower}
    if prior is not None:return prior,max(support-prior,0.0),{'type':'known-maximum','knownExistingLowerBound':lower}
    if support<=lower:return prior,0.0,{'type':'covered-by-existing-alternative','knownExistingLowerBound':lower}
    return prior,None,{'type':'unknown','knownExistingLowerBound':lower}

def evaluate(candidate, wardrobe, occasion, retriever, mode='hybrid', alpha=ALPHA, minimum_coverage=MIN_COVERAGE):
    if getattr(retriever, 'scoped', False):
        from backend.audience import validate_audiences
        validate_audiences(candidate, wardrobe)
    if not 0 <= alpha <= 1 or not 0 <= minimum_coverage <= 1:
        raise ValueError('Weights and coverage must be between zero and one')
    owned = [g for g in wardrobe if g['role'] == 'owned']
    groups = defaultdict(list)
    for garment in owned:
        if garment['category'] != candidate['category']:
            groups[group_key(garment)].append(garment)
    baseline_map = {group_key(g): g for g in owned if g['category'] == candidate['category']}
    baseline = [baseline_map[k] for k in sorted(baseline_map)]
    rows, evidence, reviews = [], {}, defaultdict(set)
    def review_inputs(audit,first,second):
        if audit:
            for gap in audit['inputGaps']:
                garment=first if gap['side']=='candidate' else second
                reviews[garment['id']].update(gap['fields'])
    for key, garments in sorted(groups.items()):
        representative = sorted(garments, key=lambda g: g['id'])[0]
        support, records,design_audit = pair_assessment(candidate, representative, occasion, retriever, mode)
        review_inputs(design_audit,candidate,representative)
        evidence.update({r['id']: r for r in records})
        existing, existing_rows = [], []
        for garment in baseline:
            value, refs,baseline_design = pair_assessment(garment, representative, occasion, retriever, mode)
            evidence.update({r['id']: r for r in refs})
            existing.append(value)
            existing_rows.append({'groupId': group_id(group_key(garment)), 'support': value,'designAudit':baseline_design,
                                  'evidenceIds': sorted(r['id'] for r in refs)})
        # Maximum is known if every baseline is known OR one establishes the upper bound 1.
        prior,gain,gain_proof=incremental_support(support,existing)
        if support is not None and gain is None:
            for garment,entry in zip(baseline,existing_rows):
                review_inputs(entry['designAudit'],garment,representative)
        grounds=support_details(records)
        if design_audit:
            grounds['support']=support;grounds['directFamilyCount']=0;grounds['derived']=True
        rows.append({'groupId': group_id(key), 'garmentIds': [g['id'] for g in garments], 'label': f'{key[1]} · {key[2]}',
                     'support': support, 'existingSupport': prior, 'gain': gain, 'evidenceIds': sorted(r['id'] for r in records),
                     'grounds': grounds, 'baselines': existing_rows,'designAudit':design_audit,'gainProof':gain_proof})
    supported = [row['support'] for row in rows if row['support'] is not None]
    gains = [row['gain'] for row in rows if row['gain'] is not None]
    coverage = len(supported) / len(rows) if rows else 0
    gain_coverage = len(gains) / len(rows) if rows else 0
    combination = sum(supported) / len(supported) if supported and coverage >= minimum_coverage else None
    # C and N must be over the SAME covered groups. No zero imputation.
    comparable = [r for r in rows if r['gain'] is not None]
    contribution = sum(gains) / len(gains) if gains and gain_coverage >= minimum_coverage else None
    score = None
    if contribution is not None:
        combination = sum(r['support'] for r in comparable) / len(comparable)
        score = 100 * (alpha * combination + (1 - alpha) * contribution)
    status = 'scored' if score is not None else 'partial' if supported else 'insufficient'
    fronts=[r['designAudit']['front']['support'] if r['designAudit'] else r['support'] for r in rows]
    front_supported=[s for s in fronts if s is not None]
    detailed=any(r['designAudit'] for r in rows)
    base_known=sum(r['designAudit']['front']['baseSupport'] is not None for r in rows if r['designAudit'])
    if score is None and front_supported:status='partial'
    if score is None and base_known:status='partial'
    reason = ('적용 가능한 근거와 기존 의류 비교로 계산했습니다.' if score is not None
              else '조합 근거는 있지만 기존 의류 비교 또는 전체 그룹의 근거가 부족해 종합 점수를 보류합니다.' if supported
              else '반대 종류의 보유 옷이 필요합니다.' if not rows
              else '현재 근거 자료에서 속성·상황 조건이 맞는 사례를 찾지 못했습니다.')
    if detailed and score is None and front_supported:
        reason='앞면에서 확인 가능한 조합은 아래에 표시했어요. 뒷면 또는 기존 대안의 상세 근거를 더 확인해야 전체 점수를 계산할 수 있어요.'
    elif detailed and score is None and base_known:
        reason=('기본 색상·무늬의 조합 근거는 있어요. 보유 옷과 후보의 장식·앞뒤 디자인 정보를 확인해야 상세 활용도를 계산할 수 있어요.' if reviews
                else '입력한 디자인 정보는 확인됐어요. 기본 색상 근거 외에 이 무늬·장식 조합의 조건을 뒷받침할 사례가 더 필요해요.')
    direct_groups = sum(r['grounds']['directFamilyCount'] > 0 for r in rows if r['support'] is not None)
    rule_only_groups = len(supported) - direct_groups
    quality = {'directGroups': direct_groups, 'ruleOnlyGroups': rule_only_groups,
               'level': 'rule-based' if supported and not direct_groups else 'mixed' if rule_only_groups else 'case-backed' if supported else 'unknown',
               'label': '색상·팔레트 규칙 중심' if supported and not direct_groups else '착장 사례와 조건부 규칙 혼합' if rule_only_groups else '구체적 착장 사례 포함' if supported else '적용 근거 없음'}
    if detailed and supported and not direct_groups:quality['label']='디자인·색상 조건부 추론 중심'
    if detailed and not supported and base_known:quality['label']='기본 색상·무늬 근거만 확인 · 디자인 확인 필요' if reviews else '기본 색상 근거만 확인 · 해당 디자인 근거 부족'
    gaps = [{'groupId':r['groupId'],'label':r['label'],
             'missingDesign':bool((r['designAudit'] and r['designAudit']['inputGaps']) or
                 (r['support'] is not None and any(b['support'] is None and b['designAudit'] and b['designAudit']['inputGaps'] for b in r['baselines']))),
             'missingCandidate':r['support'] is None,
             'missingBaselineGroups':[b['groupId'] for b in r['baselines'] if b['support'] is None] if r['support'] is not None and r['gain'] is None else []}
            for r in rows if r['support'] is None or r['gain'] is None]
    canonical = {'candidate': group_key(candidate), 'oppositeGroups': sorted(groups), 'baselineGroups': sorted(baseline_map),
                 'occasion': occasion, 'mode': mode, 'alpha': alpha, 'minimumCoverage': minimum_coverage, 'version': VERSION,
                 'corpus': getattr(retriever, 'index', {}).get('chunksSha256', 'test'),'designRulesSha256':rulebook()[1]}
    fingerprint = hashlib.sha256(json.dumps(canonical, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    return {'candidateId': candidate['id'], 'status': status, 'score': round(score, 1) if score is not None else None,
            'evidenceQuality':quality, 'gaps':gaps,
            'inputReviews':[{'garmentId':gid,'fields':sorted(fields)} for gid,fields in sorted(reviews.items())],
            'assessmentScope':'front-back-design' if detailed else 'basic-attributes',
            'partialAssessment':{'frontSupportedGroups':len(front_supported),'baseSupportedGroups':base_known,'totalGroups':len(rows),
                'frontCombination':round(100*math.fsum(front_supported)/len(front_supported),1) if front_supported else None,
                'label':'확인된 앞면 조합만의 조건부 참고값 · 전체 활용도 아님' if detailed else '근거가 있는 기본 속성 조합만의 참고값'},
            'evaluationFingerprint': fingerprint, 'counts': {'totalGroups': len(rows), 'supportedGroups': len(supported),
                'comparableGroups': len(comparable), 'baselineGroups': len(baseline), 'uniqueFamilies': len({r.get('sourceFamily', r['sourceId']) for r in evidence.values()})},
            'combination': round(100 * combination, 1) if combination is not None else None,
            'contribution': round(100 * contribution, 1) if contribution is not None else None,
            'coverage': round(100 * coverage, 1), 'gainCoverage': round(100 * gain_coverage, 1), 'groups': rows,
            'evidence': [evidence[k] for k in sorted(evidence)], 'message': reason, 'method': {'version': VERSION, 'alpha': alpha,
            'minimumCoverage': minimum_coverage, 'retrieval': mode, 'grouping': '종류·색상·무늬·앞뒤 디자인의 동일 속성 그룹',
            'designRulesVersion':rulebook()[0]['version'],'designRulesSha256':rulebook()[1],
            'formula': 'U = 100 × (αC + (1−α)N)', 'annotation': getattr(retriever,'manifest',{}).get('annotationVersion','test'),
            'aggregation': '원 출처 계열별 한 표 · 구체적 사례 우선 · 같은 수준에서 충돌 시 보수적 집계'},
            'limitations': ['개발용 주석과 가중치로 계산한 실험 지수이며 만족 확률이 아닙니다.',
                             '사진 AI 제안에서 사용자가 확인한 의류 속성을 사용하며 실제 색상 톤·소재·핏은 따로 확인해야 합니다.',
                             '부분 근거의 점수는 계산 가능한 그룹에 한정됩니다. 다른 후보와 근거 범위가 다를 수 있습니다.']}
