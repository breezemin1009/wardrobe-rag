"""Typed visual descriptions. Logo identity/brand prestige is outside this schema."""
import json
from typing import Literal
from pydantic import BaseModel,ConfigDict,Field,model_validator

Colour=Literal['화이트','크림','블루','블랙','베이지','그레이','올리브','레드','핑크','브라운','옐로','오렌지','퍼플','그린','기타']
Pattern=Literal['무지','스트라이프','체크','프린트','기타']
Scale=Literal['none','fine','medium','bold','unknown']
GarmentType=Literal['tshirt','shirt','knit','jacket','jeans','trousers','shorts','skirt','other','unknown']

class Accent(BaseModel):
    model_config=ConfigDict(extra='forbid')
    kind:Literal['logo','text','graphic','patch','other']
    size:Literal['small','medium','large','unknown']
    placement:Literal['chest','center','upper','lower','side','all-over','other']
    colors:list[Colour]=Field(default_factory=list,max_length=6)

class ViewDesign(BaseModel):
    model_config=ConfigDict(extra='forbid')
    baseColor:Colour
    pattern:Pattern
    patternColors:list[Colour]=Field(default_factory=list,max_length=6)
    patternScale:Scale='unknown'
    contrast:Literal['low','medium','high','unknown']='unknown'
    accents:list[Accent]=Field(default_factory=list,max_length=6)
    uncertain:list[Literal['baseColor','pattern','patternColors','patternScale','contrast','accents']]=Field(default_factory=list,max_length=6)
    @model_validator(mode='after')
    def normalise(self):
        self.patternColors=sorted(set(self.patternColors)-{self.baseColor})
        self.uncertain=sorted(set(self.uncertain))
        if self.pattern=='무지':self.patternScale='none';self.patternColors=[]
        for accent in self.accents:accent.colors=sorted(set(accent.colors))
        return self

class DesignProfile(BaseModel):
    model_config=ConfigDict(extra='forbid')
    version:Literal[1]=1
    garmentType:GarmentType='unknown'
    front:ViewDesign
    back:ViewDesign|None=None
    backStatus:Literal['unknown','observed','same-as-front','plain-confirmed']='unknown'
    @model_validator(mode='after')
    def back_is_explicit(self):
        if (self.backStatus=='observed') != (self.back is not None):
            raise ValueError('Observed back requires its own description; unobserved back cannot contain inferred details')
        return self

class DesignProposal(BaseModel):
    model_config=ConfigDict(extra='forbid')
    category:Literal['상의','하의']
    garmentType:GarmentType='unknown'
    garmentPresent:bool
    singleGarment:bool
    front:ViewDesign
    back:ViewDesign|None=None
    pairCheck:Literal['single','consistent','different','uncertain']='single'
    uncertain:list[str]=Field(default_factory=list,max_length=12)

def profile(value):
    return DesignProfile.model_validate(value).model_dump() if value else None

def back_view(design):
    if design['backStatus']=='observed':return design['back']
    if design['backStatus']=='same-as-front':return design['front']
    if design['backStatus']=='plain-confirmed':
        return ViewDesign(baseColor=design['front']['baseColor'],pattern='무지',contrast='low').model_dump()
    return None

def canonical(design):
    value=profile(design)
    if not value:return None
    # Equivalent observed/explicitly-confirmed back descriptions share a group.
    # Provenance stays in the saved profile, not in visual duplicate counting.
    value['back']=back_view(value);value['backStatus']='confirmed' if value['back'] else 'unknown'
    for view in [value['front'],value['back']]:
        if view:
            view['accents']=sorted(view['accents'],key=lambda x:json.dumps(x,sort_keys=True,ensure_ascii=False))
            if view['pattern']=='무지' and not view['accents']:
                view['contrast']='low';view['uncertain']=[k for k in view['uncertain'] if k not in {'contrast','patternScale','patternColors'}]
    return json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'))

def legacy_colour(value):
    return value if value in {'화이트','크림','블루','블랙','베이지','그레이','올리브'} else '기타'

def describe(view):
    colours=' / '.join([view['baseColor']]+view['patternColors'])
    scales={'none':'','fine':'가는·작은','medium':'중간','bold':'굵은·큰','unknown':'크기 미확인'}
    kinds={'logo':'로고','text':'문자','graphic':'그래픽','patch':'패치','other':'장식'}
    sizes={'small':'작은','medium':'중간','large':'큰','unknown':'크기 미확인'}
    parts=[f'{colours} · {scales[view["patternScale"]]} {view["pattern"]}'.replace('  ',' ')]
    parts += [f'{sizes[a["size"]]} {kinds[a["kind"]]} ({" / ".join(a["colors"]) or "색 미확인"})' for a in view['accents']]
    return ' · '.join(parts)
