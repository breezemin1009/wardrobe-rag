"""Bounded, deterministic composition rules. No LLM-generated aesthetic scores."""
import hashlib,json
from functools import lru_cache
from backend.config import DATA
from backend.design import profile,back_view,ViewDesign,describe

@lru_cache(maxsize=1)
def rulebook():
    raw=(DATA/'design-rules.json').read_bytes()
    data=json.loads(raw)
    assert data['graphicDenim']['supportCeiling']==.5
    assert data['source']['url'].startswith('https://')
    return data,hashlib.sha256(raw).hexdigest()

def legacy_view(g):
    return ViewDesign(baseColor=g['color'],pattern=g['pattern'],uncertain=['accents']).model_dump()

def missing_fields(view):
    fields=set(view['uncertain'])
    if view['pattern']=='무지' and not view['accents']:fields-={'contrast','patternScale','patternColors'}
    if view['baseColor']=='기타':fields.add('baseColor')
    if view['pattern']=='기타':fields.add('pattern')
    if view['pattern']!='무지':
        if not view['patternColors'] or '기타' in view['patternColors']:fields.add('patternColors')
        if view['patternScale'] in {'none','unknown'}:fields.add('patternScale')
        if view['contrast']=='unknown':fields.add('contrast')
    if any(a['size']=='unknown' or not a['colors'] or '기타' in a['colors'] for a in view['accents']):fields.add('accents')
    return sorted(fields)

def input_gaps(one,two):
    decorated=any(v['accents'] for p in [one,two] if p for v in [p['front'],back_view(p)] if v)
    gaps=[]
    for side,p in [('candidate',one),('owned',two)]:
        fields=[]
        if not p:fields=['design']
        else:
            fields+=['front.'+key for key in missing_fields(p['front'])]
            back=back_view(p)
            if back:fields+=['back.'+key for key in missing_fields(back)]
            else:fields.append('back')
            if decorated and p['garmentType']=='unknown':fields.append('garmentType')
        if fields:gaps.append({'side':side,'fields':sorted(set(fields))})
    return gaps

def view_result(first,second,first_view,second_view,occasion,lookup,combine,types,allowed_sources=None):
    a={**first,'color':first_view['baseColor'],'pattern':first_view['pattern']}
    b={**second,'color':second_view['baseColor'],'pattern':second_view['pattern']}
    base=list(lookup(a,b));value=combine(base)
    refs={r['id']:r for r in base};rules=[];reasons=[]
    views=[first_view,second_view]
    for v in views:
        uncertain=set(v['uncertain'])
        if v['pattern']=='무지' and not v['accents']:uncertain-={'contrast','patternScale','patternColors'}
        if uncertain:reasons.append('사진에서 확실하지 않은 디자인 항목을 직접 확인해주세요.')
        if v['baseColor']=='기타' or v['pattern']=='기타':reasons.append('분류되지 않은 색상·무늬의 근거가 필요해요.')
        if v['pattern']!='무지' and (not v['patternColors'] or v['patternScale'] in {'none','unknown'} or v['contrast']=='unknown'):
            reasons.append('반복 무늬의 색·폭·대비를 함께 확인해주세요.')
        for accent in v['accents']:
            if accent['size']=='unknown' or not accent['colors'] or '기타' in accent['colors']:
                reasons.append('로고·장식의 크기와 색을 확인해주세요.')
    busy=[v['pattern']!='무지' or bool(v['accents']) for v in views]
    if all(busy):reasons.append('양쪽 모두 무늬·장식이 있는 조합은 별도 사례 근거가 필요해요.')
    if reasons:
        value=None
    elif not any(busy):
        rules=['confirmed-plain-pair'] if value is not None else []
    else:
        index=busy.index(True);v=views[index];quiet=views[1-index]
        if v['accents']:
            data,_=rulebook();rule=data['graphicDenim']
            top,bottom=(first_view,second_view) if first['category']=='상의' else (second_view,first_view)
            top_type,bottom_type=types if first['category']=='상의' else types[::-1]
            source_available=allowed_sources is None or data['source']['sourceId'] in allowed_sources
            matches=(source_available and top_type==rule['topType'] and bottom_type==rule['bottomType']
                and top['baseColor'] in rule['topBaseColors'] and bottom['baseColor'] in rule['bottomBaseColors']
                and occasion in rule['occasions'] and top['pattern']=='무지' and bool(top['accents'])
                and bottom['pattern']=='무지' and not bottom['accents'] and value is not None)
            if matches:
                refs[data['source']['id']]=data['source'];rules=['graphic-tee-confirmed-denim']
                value=min(value,rule['supportCeiling'])
            else:
                value=None;reasons.append('장식이 있는 옷은 종류·색·상황이 맞는 그래픽 조합 근거가 필요해요.')
        else:
            # Secondary colour roles are not pooled with the fabric background.
            # Every repeated colour needs a supported solid-palette relationship
            # AND an existing pattern-to-plain source for the original base pair.
            palette=[]
            for colour in v['patternColors']:
                x={**(a if index==0 else b),'color':colour,'pattern':'무지'}
                y={**(b if index==0 else a),'color':quiet['baseColor'],'pattern':'무지'}
                records=list(lookup(x,y));palette.append(combine(records));refs.update({r['id']:r for r in records})
            if value is None or not palette or any(p is None for p in palette):
                value=None;reasons.append('바탕색과 각 무늬색을 함께 뒷받침하는 조합 근거가 부족해요.')
            else:
                value=min([.5,value,*palette]);rules=['pattern-base-and-secondary-with-plain']
    if value is None and not reasons:reasons.append('현재 조건에 적용되는 원문 근거가 없어요.')
    return {'support':value,'baseSupport':combine(base),'summary':[describe(v) for v in views],
            'reasons':sorted(set(reasons)),'ruleIds':rules,'evidenceIds':sorted(refs)},list(refs.values())

def assess_design(first,second,occasion,lookup,combine,allowed_sources=None):
    one,two=profile(first.get('design')),profile(second.get('design'))
    front,refs=view_result(first,second,one['front'] if one else legacy_view(first),
        two['front'] if two else legacy_view(second),occasion,lookup,combine,
        (one['garmentType'] if one else 'unknown',two['garmentType'] if two else 'unknown'),allowed_sources)
    back=None
    if one and two and back_view(one) and back_view(two):
        back,more=view_result(first,second,back_view(one),back_view(two),occasion,lookup,combine,(one['garmentType'],two['garmentType']),allowed_sources)
        refs+=more
    complete=back is not None
    # An unseen back is never copied from the front implicitly. The minimum
    # rule prevents a small front logo masking an unsupported large rear print.
    support=min(front['support'],back['support']) if complete and front['support'] is not None and back['support'] is not None else None
    return support,list({r['id']:r for r in refs}.values()),{
        'scope':'front-back' if complete else 'front-only','front':front,'back':back,
        'legacyInputs':[name for name,p in [('candidate',one),('owned',two)] if not p],
        'inputGaps':input_gaps(one,two),
        'missingBack':not complete,'aggregation':'앞·뒤 중 낮은 지원을 사용 · 미확인 면은 추정하지 않음',
        'ruleVersion':rulebook()[0]['version']}
