"""Conservative extraction of explicitly named clothing conditions; no LLM labels."""
import re

COLOURS={'화이트':'화이트','흰색':'화이트','흰':'화이트','크림':'크림','오프화이트':'크림',
         '블루':'블루','파란':'블루','파랑':'블루','네이비':'블루','블랙':'블랙','검정':'블랙','검은':'블랙',
         '베이지':'베이지','그레이':'그레이','회색':'그레이','올리브':'올리브'}
PATTERNS=['무지','스트라이프','체크','프린트']

def explicit_conditions(query):
    result={}
    colour_rx='|'.join(sorted(COLOURS,key=len,reverse=True))
    marker=re.compile(r'상의|티셔츠|셔츠|스웨터|니트|폴로|하의|바지|팬츠|데님(?!\s*셔츠)')
    previous=0
    for match in marker.finditer(query):
        prefix=query[max(previous,match.start()-24):match.start()]
        colours=list(re.finditer(colour_rx,prefix))
        side='top' if match.group() in {'상의','티셔츠','셔츠','스웨터','니트','폴로'} else 'bottom'
        if colours:
            result[side+'Color']=COLOURS[colours[-1].group()]
            patterns=[(prefix.rfind(p),p) for p in PATTERNS if p in prefix]
            if patterns:result[side+'Pattern']=max(patterns)[1]
        previous=match.end()
    # A single colour or an ambiguous list is not a complete outfit query.
    if not all(result.get(k) for k in ['topColor','bottomColor']):return {}
    for occasion in ['데일리','출근','나들이']:
        if occasion in query:result['occasion']=occasion;break
    return result

def matches_conditions(record,conditions):
    for side in ['top','bottom']:
        if conditions[side+'Color'] not in record.get(side+'Colors',[]):return False
        pattern=conditions.get(side+'Pattern')
        if pattern and pattern not in record.get(side+'Patterns',record.get('patterns',[])):return False
    return not conditions.get('occasion') or conditions['occasion'] in record.get('occasions',[])
