"""Conservative, source-title-grounded adult menswear projection of POG.

Opaque category hashes are retained, never decoded by guesswork. Inferences
from explicit title words are separately recorded. Outfit membership is the
publisher's observed association, not an aesthetic rating or a negative label.
Raw and projected data stay local (the release states no redistribution license).
"""
from pathlib import Path
from collections import Counter,defaultdict
from urllib.parse import urlparse
import hashlib,io,json,re,zipfile
from backend.data_io import write_json_atomic as write

ROOT=Path(__file__).parent/'data/menswear-expansion-v3/pog'
RULE_VERSION='explicit-adult-menswear-title-v1'
EXCLUDE=re.compile(r'女|童|儿童|宝宝|婴|孕|亲子|小学生|初中生|中学生|高中生|青少年|校服|睡衣|睡裤|泳|内裤|保暖内衣|家居服|浴袍|情趣|戏服|演出服|舞蹈|汉服|古装|项链|毛衣链|耳环|手链|胸针|衣架|袖扣')
TOP=[('blazer',r'西装|西服|休闲西'),('shirt',r'衬衫|衬衣'),('hoodie',r'卫衣'),('knit',r'毛衣|针织衫|羊毛衫|羊绒衫|毛衫|开衫'),('tee',r'T恤|t恤|Ｔ恤|体恤|短T|短t|长T|长t|polo|POLO|打底衫'),('jacket',r'夹克|棒球服|牛仔衣|牛仔外套|棉衣|棉服|羽绒|冲锋衣|皮衣|风衣|大衣|外套')]
BOTTOM=[('shorts',r'短裤|五分裤|七分裤|九分短裤|中裤|沙滩裤'),('jeans',r'牛仔裤'),('trousers',r'休闲裤|西裤|长裤|运动裤|卫裤|工装裤|小脚裤|直筒裤|阔腿裤|束脚裤|哈伦裤|皮裤')]

def classify(title):
    if not re.search(r'男',title):return None,'not-explicit-menswear'
    if EXCLUDE.search(title):return None,'excluded-gender-age-or-special-purpose'
    if any(x in title for x in ['套装','两件套','三件套','二件套','连体','连身']):return None,'multi-garment-set'
    tops=[(k,re.search(pattern,title)) for k,pattern in TOP];bottoms=[(k,re.search(pattern,title)) for k,pattern in BOTTOM]
    tops=[(k,m.group()) for k,m in tops if m];bottoms=[(k,m.group()) for k,m in bottoms if m]
    if bool(tops)==bool(bottoms):return None,'ambiguous-or-non-apparel'
    matches=tops or bottoms
    # Hierarchical words such as "short denim pants" have a defined precedence.
    kind,word=matches[0]
    return dict(side='top' if tops else 'bottom',kind=kind,audience='mens',audienceBasis=dict(type='explicit-original-title',value=title,token='男'),categoryBasis=dict(type='explicit-original-title-rule',token=word,version=RULE_VERSION)),'accepted'

def norm_title(t):return re.sub(r'\s+','',t).casefold()
def image_key(url):
    u=urlparse(url);return u.netloc.lower()+u.path

def main():
    items={};counts=Counter();categories=defaultdict(Counter);ambiguous=[];seen={};id_conflicts=set();families={}
    with zipfile.ZipFile(ROOT/'item_data.txt.zip') as z:
        with z.open('item_data.txt') as f:
            for line in io.TextIOWrapper(f,encoding='utf8'):
                if not line.strip():continue
                counts['sourceRows']+=1
                fields=line.rstrip('\r\n').split(',',3)
                if len(fields)!=4:counts['malformed']+=1;continue
                pid,category,url,title=fields;meta,reason=classify(title);counts[reason]+=1
                if meta is None:continue
                signature=(category,image_key(url),norm_title(title))
                if pid in seen:
                    if seen[pid]!=signature:id_conflicts.add(pid)
                    else:counts['duplicateAcceptedItemId']+=1
                    continue
                seen[pid]=signature
                # Exact normalized image URLs define image-identity families.
                family=hashlib.sha256(image_key(url).encode()).hexdigest()[:24]
                items[pid]=dict(id=pid,title=title,imageUrl=url,sourceCategory=category,family=family,**meta)
                categories[category][meta['side']+'/'+meta['kind']]+=1
    for pid in id_conflicts:items.pop(pid,None)
    # Identical source photographs cannot be counted as independent products.
    family_side=defaultdict(set)
    for x in items.values():family_side[x['family']].add(x['side'])
    badfamilies={k for k,v in family_side.items() if len(v)>1}
    items={k:v for k,v in items.items() if v['family'] not in badfamilies}
    print('items',len(items),dict(counts),flush=True)
    pairs={};outfits=[];outcounts=Counter();allids=set(items)
    with zipfile.ZipFile(ROOT/'outfit_data.txt.zip') as z:
        with z.open('outfit_data.txt') as f:
            for line in io.TextIOWrapper(f,encoding='utf8'):
                if not line.strip():continue
                outcounts['sourceOutfits']+=1
                oid,rawids=line.strip().split(',',1);ids=rawids.split(';')
                accepted=[i for i in ids if i in allids];tops=[i for i in accepted if items[i]['side']=='top'];bottoms=[i for i in accepted if items[i]['side']=='bottom']
                if not tops or not bottoms:continue
                # Projection keeps explicitly male garments; unknown accessories
                # do not become confirmed male data by co-occurrence.
                outfits.append(dict(id=oid,tops=tops,bottoms=bottoms,acceptedItems=accepted,originalItemCount=len(ids)))
                for top in tops:
                    for bottom in bottoms:
                        identity=(items[top]['family'],items[bottom]['family'])
                        if identity in pairs:pairs[identity]['outfits'].append(oid)
                        else:pairs[identity]=dict(top=top,bottom=bottom,outfits=[oid])
    used={x[s] for x in pairs.values() for s in ['top','bottom']}
    write(ROOT/'menswear-items.json',items)
    write(ROOT/'menswear-pairs.json',dict(pairs=list(pairs.values()),outfits=outfits))
    audit=dict(ruleVersion=RULE_VERSION,sourceCounts=dict(counts),outfitCounts=dict(outcounts),acceptedItems=len(items),usedItems=len(used),acceptedOutfits=len(outfits),canonicalPairs=len(pairs),conflictingItemIds=len(id_conflicts),crossSidePhotoFamilies=len(badfamilies),byKind=dict(Counter(x['side']+'/'+x['kind'] for x in items.values())),usedByKind=dict(Counter(items[k]['side']+'/'+items[k]['kind'] for k in used)),categoryEvidence={k:dict(v) for k,v in categories.items()},sampleTitles=[items[k] for k in sorted(used)[:20]],sourceSemantics='Observed co-outfit association, not proof that missing pairs are incompatible; gender is explicit original title, not appearance inference',redistribution='Private research cache only; no redistribution license asserted')
    write(ROOT/'menswear-audit.json',audit);print(json.dumps({k:v for k,v in audit.items() if k not in ['sampleTitles','categoryEvidence']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
