"""Local foreground extraction. Never writes uploaded photos or calls a cloud service."""
import base64
import hashlib
import io
import threading
from collections import OrderedDict
from pathlib import Path
import numpy as np
from PIL import Image
from scipy import ndimage
from backend.vision import prepare_photo, VisionFailure

MODEL_PATH = Path(__file__).resolve().parent / 'data/models/isnet-general-use.onnx'
MODEL_MD5 = 'fc16ebd8b0c10d971d3513d564d01e29'
MODEL_SHA256 = '60920e99c45464f2ba57bee2ad08c919a52bbf852739e96947fbb4358c0d964a'
_session = None
_revision = None
_lock = threading.Lock()
_cache = OrderedDict()

def _network_mask(image):
    global _session, _revision
    if _session is None:
        if not MODEL_PATH.is_file():
            raise VisionFailure('배경 분리 모델 준비가 필요해요. 원본 사진과 수동 입력을 사용할 수 있습니다.',503)
        blob=MODEL_PATH.read_bytes()
        if hashlib.sha256(blob).hexdigest()!=MODEL_SHA256:
            raise VisionFailure('배경 분리 모델 파일을 확인해주세요. 원본 사진은 유지됩니다.',503)
        _revision=hashlib.sha256(blob).hexdigest()
        del blob
        import onnxruntime as ort
        options=ort.SessionOptions();options.intra_op_num_threads=4;options.inter_op_num_threads=1
        _session=ort.InferenceSession(str(MODEL_PATH),sess_options=options,providers=['CPUExecutionProvider'])
    # ISNet preprocessing/postprocessing follows rembg 2.0.85 DisSession.
    # Load ONNX directly to avoid importing unrelated matting/JIT models.
    pixels=np.asarray(image.convert('RGB').resize((1024,1024),Image.Resampling.LANCZOS),dtype=np.float32)
    pixels=pixels/max(float(pixels.max()),1e-6)-.5
    tensor=pixels.transpose(2,0,1)[None,...]
    prediction=_session.run(None,{_session.get_inputs()[0].name:tensor})[0][0,0]
    minimum,maximum=float(prediction.min()),float(prediction.max())
    alpha=(prediction-minimum)/max(maximum-minimum,1e-6)
    return Image.fromarray((alpha*255).clip(0,255).astype('uint8')).resize(image.size,Image.Resampling.LANCZOS)


def uniform_background(image,tolerance=4):
    """Find an edge-connected, nearly uniform backdrop using image pixels only.

    Interior regions matching the backdrop are retained when disconnected from
    the image edge. No ground-truth garment mask is used during registration.
    """
    pixels=np.asarray(image.convert('RGB'),dtype=np.int16)
    border=np.concatenate([pixels[0],pixels[-1],pixels[:,0],pixels[:,-1]])
    colour=np.median(border,axis=0)
    if np.quantile(np.max(np.abs(border-colour),axis=1),.95)>4:return None
    close=np.max(np.abs(pixels-colour),axis=2)<=tolerance
    labels,_=ndimage.label(close)
    edges=np.unique(np.concatenate([labels[0],labels[-1],labels[:,0],labels[:,-1]]));edges=edges[edges!=0]
    background=np.isin(labels,edges);visible=~background;fraction=float(visible.mean())
    if not .04<=fraction<=.9:return None
    return dict(background=background,visible=visible,colour=colour)


def _predict_mask(image):
    original=_network_mask(image);backdrop=uniform_background(image)
    if backdrop is None:return original
    visible=backdrop['visible'];first=np.asarray(original)>=128
    agreement=float((first&visible).sum()/max((first|visible).sum(),1))
    pixels=np.asarray(image.convert('RGB'));fg=pixels[first]
    colour_contrast=float(np.linalg.norm(np.median(fg,axis=0)-backdrop['colour'])) if len(fg) else 0.
    if agreement>=.94 and colour_contrast>=65:return original
    # A neutral-background second pass repairs white/white and black/black
    # bias. Only the inferred backdrop is changed for model input; output
    # cutouts always retain the original garment RGB pixels.
    # Exact edge-connected background preserves near-white highlights and
    # near-black folds that would otherwise merge into a tolerance band.
    precise=uniform_background(image,tolerance=0) or backdrop
    visible=precise['visible'];agreement=float((first&visible).sum()/max((first|visible).sum(),1))
    neutral=np.array(image.convert('RGB'));neutral[precise['background']]=[144,150,158]
    second=_network_mask(Image.fromarray(neutral));other=np.asarray(second)>=128
    second_agreement=float((other&visible).sum()/max((other|visible).sum(),1))
    return second if second_agreement>agreement+.015 else original

def inspect_mask(image,mask):
    alpha=np.asarray(mask,dtype=np.float32)/255
    foreground=alpha>=.5
    fraction=float(foreground.mean())
    labels,count=ndimage.label(foreground)
    sizes=np.bincount(labels.ravel())[1:]
    significant=int((sizes>=max(64,foreground.sum()*.15)).sum()) if count else 0
    edge=np.concatenate([foreground[0,:],foreground[-1,:],foreground[:,0],foreground[:,-1]])
    warnings=[]
    if fraction<.04:warnings.append('옷 영역을 충분히 찾지 못했어요. 옷과 대비되는 배경에서 다시 촬영해주세요.')
    if fraction>.92:warnings.append('배경과 옷을 구분하기 어려워요. 옷 주변에 여백을 두고 다시 촬영해주세요.')
    if float(edge.mean())>.1:warnings.append('옷이 사진 가장자리에 닿아 일부가 잘렸을 수 있어요.')
    if significant>1:warnings.append('여러 물체가 보일 수 있어요. 옷 한 벌만 놓고 확인해주세요.')
    pixels=np.asarray(image,dtype=np.float32)
    fg=pixels[foreground];bg=pixels[alpha<.1]
    contrast=float(np.linalg.norm(np.median(fg,axis=0)-np.median(bg,axis=0))) if len(fg) and len(bg) else 0
    if contrast<55:warnings.append('옷과 배경의 색이 비슷해요. 소매와 밑단이 남아 있는지 확인해주세요.')
    backdrop=uniform_background(image,tolerance=0);retention=None;agreement=None
    if backdrop is not None:
        visible=backdrop['visible'];retention=float((foreground&visible).sum()/max(visible.sum(),1));agreement=float((foreground&visible).sum()/max((foreground|visible).sum(),1))
        if retention<.94 or agreement<.85:warnings.append('사진에 보이는 옷의 일부가 분리 결과에서 빠졌을 수 있어요. 원본과 소매·밑단을 비교해주세요.')
    return {'foregroundFraction':round(fraction,4),'foregroundBackgroundContrast':round(contrast,1),
            'visibleRegionRetention':round(retention,4) if retention is not None else None,'uniformBackdropAgreement':round(agreement,4) if agreement is not None else None,
            'significantComponents':significant,'needsReview':bool(warnings),'warnings':warnings,
            'usable':.04<=fraction<=.92}

def extract(data_uri):
    # Use the same bounded decoder, orientation and metadata removal as vision.
    normalised=prepare_photo(data_uri)
    digest=hashlib.sha256(('isnet-v3-detail-preserving-backdrop:'+normalised).encode()).hexdigest()
    image=Image.open(io.BytesIO(base64.b64decode(normalised.split(',',1)[1]))).convert('RGB')
    if float(np.asarray(image,dtype=np.float32).std(axis=(0,1)).max())<4:
        return {'usable':False,'cutout':None,'analysisImage':None,'quality':{'needsReview':True,'warnings':['옷이 없는 사진이거나 경계가 보이지 않아요.']},
                'message':'옷이 있는지 확인하고 배경과 구분되도록 다시 촬영해주세요. 원본은 유지됩니다.','model':'isnet-general-use','reviewRequired':True}
    with _lock:
        if digest in _cache:
            _cache.move_to_end(digest)
            return {**_cache[digest],'cached':True}
        mask=_predict_mask(image)
        quality=inspect_mask(image,mask)
        result={'usable':quality['usable'],'cutout':None,'analysisImage':None,'quality':quality,
                'model':'isnet-general-use','modelSha256':_revision,'reviewRequired':True,'cached':False,
                'message':'옷의 소매·밑단·무늬를 원본과 비교한 뒤 적용해주세요.'}
        if quality['usable']:
            rgba=image.convert('RGBA');rgba.putalpha(mask)
            bounds=mask.point(lambda x:255 if x>=24 else 0).getbbox()
            if bounds:
                x0,y0,x1,y1=bounds
                rgba=rgba.crop((max(0,x0-20),max(0,y0-20),min(image.width,x1+20),min(image.height,y1+20)))
                buffer=io.BytesIO();rgba.save(buffer,format='PNG',optimize=True)
                result['cutout']='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()
                neutral=Image.new('RGBA',rgba.size,(144,150,158,255));neutral.alpha_composite(rgba)
                buffer=io.BytesIO();neutral.convert('RGB').save(buffer,format='JPEG',quality=90)
                result['analysisImage']='data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode()
        else:result['message']='옷의 윤곽을 안전하게 분리하지 못했어요. 대비되는 배경에서 다시 촬영하거나 원본을 사용해주세요.'
        _cache[digest]=result
        if len(_cache)>8:_cache.popitem(last=False)
        return result
