"""Scope is a garment's declared product line, never the wearer's gender."""
ALLOWED_AUDIENCES = frozenset({'mens', 'unisex'})
SCOPE = 'adult-menswear-and-verified-unisex-v1'


def validate_audiences(candidate, wardrobe):
    invalid = [g for g in [candidate, *wardrobe] if g.get('audience') not in ALLOWED_AUDIENCES]
    if invalid:
        names = '、'.join(g.get('name', g.get('id', '의류')) for g in invalid[:3])
        raise ValueError(f'{names}: 의류 구분을 남성복 또는 공용으로 확인해주세요. 미확인 의류는 정보 수정이나 보관함 이동 후 평가할 수 있어요.')


def eligible_evidence(record):
    return record.get('scopeVerified') is True and record.get('audience') in ALLOWED_AUDIENCES
