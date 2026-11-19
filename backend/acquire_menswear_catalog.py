"""Download the public dataset through its owner's ordinary Kaggle endpoint.

Only numeric, metadata-verified adult menswear/unisex apparel photographs are
extracted. Product catalog records are not treated as outfit compatibility facts.
"""
from pathlib import Path
import argparse,csv,io,json,hashlib,zipfile,re
from collections import Counter
import requests
from PIL import Image
from backend.data_io import write_json_atomic

TOPS={'Shirts','Tshirts','Sweatshirts','Sweaters','Jackets','Blazers','Waistcoat'}
BOTTOMS={'Jeans','Trousers','Track Pants','Shorts'}


def title_scope_conflict(title):
    # Source gender metadata contains mislabeled Kidswear rows. Possessives,
    # singulars and compounds must not escape an exact-word plural-only check.
    return bool(re.search(r"\b(?:wom[ae]n(?:[’']s|s)?|female|lad(?:y|ies)|girl(?:s|[’']s)?|boy(?:s|[’']s)?|kid(?:s|[’']s)?(?:wear)?|child(?:ren)?(?:s|[’']s)?|junior|toddler|infant|baby|babies|hot\s*pants)\b",title,re.I))


def multi_garment_title(title):
    return bool(re.search(r'\b(?:track.?suit|suit set|set of|pack of|combo|two.?piece|twin.?pack)\b',title,re.I))


def acquire(root):
    source=root/'source/myntra';source.mkdir(parents=True,exist_ok=True)
    session=requests.Session();owner='paramaggarwal/fashion-product-images-small'
    metadata=session.get('https://www.kaggle.com/api/v1/datasets/view/'+owner,timeout=30);metadata.raise_for_status()
    (source/'kaggle-metadata.json').write_text(json.dumps(metadata.json(),ensure_ascii=False,indent=2),encoding='utf8')
    archive=source/'fashion-product-images-small.zip'
    if not archive.exists():
        partial=source/'fashion-product-images-small.zip.partial'
        response=session.get('https://www.kaggle.com/api/v1/datasets/download/'+owner,stream=True,timeout=(20,60));response.raise_for_status()
        if 'zip' not in response.headers.get('Content-Type',''):raise ValueError('Public ZIP is unavailable; no alternate credential or access route used')
        received=0;next_progress=100_000_000
        with partial.open('wb') as stream:
            for chunk in response.iter_content(1024*1024):
                if not chunk:continue
                stream.write(chunk);received+=len(chunk)
                if received>=next_progress:print(json.dumps(dict(downloadedMB=received//1_000_000)),flush=True);next_progress+=100_000_000
        expected=response.headers.get('Content-Length')
        if expected and received!=int(expected):raise ValueError('Incomplete original archive')
        partial.replace(archive)
    accepted=[];rejected=Counter();photos=root/'catalog-images';photos.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as package:
        names=package.namelist();csv_name=next(name for name in names if name.rsplit('/',1)[-1]=='styles.csv')
        raw=package.read(csv_name);(source/'styles.csv').write_bytes(raw)
        rows=csv.reader(io.StringIO(raw.decode('utf-8-sig')));header=next(rows);images={}
        for name in names:
            basename=name.rsplit('/',1)[-1]
            if re.fullmatch(r'\d+\.jpg',basename):images[basename[:-4]]=name
        for values in rows:
            if len(values)!=len(header):rejected['malformedCSV']+=1;continue
            row=dict(zip(header,values));pid=row['id'];gender=row['gender'];kind=row['articleType']
            if gender not in {'Men','Unisex'}:rejected['audienceOutsideScope']+=1;continue
            if row['masterCategory']!='Apparel' or kind not in TOPS|BOTTOMS:rejected['categoryOutsideScope']+=1;continue
            if title_scope_conflict(row['productDisplayName']):rejected['titleScopeConflict']+=1;continue
            if multi_garment_title(row['productDisplayName']):rejected['multiGarmentProduct']+=1;continue
            if pid not in images:rejected['missingImage']+=1;continue
            try:
                data=package.read(images[pid])
                with Image.open(io.BytesIO(data)) as picture:
                    picture.load();width,height=picture.size
                    if width<40 or height<40:raise ValueError('Unusable photograph')
            except (ValueError,OSError,zipfile.BadZipFile):rejected['invalidImage']+=1;continue
            (photos/(pid+'.jpg')).write_bytes(data)
            accepted.append({**row,'id':'myntra-'+pid,'productId':pid,'audience':'mens' if gender=='Men' else 'unisex',
                'audienceBasis':dict(type='source-metadata',field='gender',value=gender,source='myntra-kaggle'),
                'category':'상의' if kind in TOPS else '하의','photoSha256':hashlib.sha256(data).hexdigest(),
                'width':width,'height':height,'sourceUrl':'https://www.kaggle.com/datasets/'+owner})
    write_json_atomic(root/'catalog.json',accepted)
    sha=hashlib.sha256()
    with archive.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):sha.update(chunk)
    report=dict(source='https://www.kaggle.com/datasets/'+owner,declaredLicense=metadata.json().get('licenseName'),
        archiveSha256=sha.hexdigest(),accepted=len(accepted),audiences=dict(Counter(r['audience'] for r in accepted)),
        categories=dict(Counter(r['articleType'] for r in accepted)),rejected=dict(rejected),outfitLabels=False)
    write_json_atomic(root/'catalog-audit.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=Path('backend/data/menswear-rebuild-v1'));a=p.parse_args();acquire(a.output)
