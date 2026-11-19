"""Publish a complete immutable index, then atomically switch one small pointer."""
from pathlib import Path
import json, os, re, tempfile
from backend.config import DATA
from backend.audience import SCOPE


def active_index_path(data=DATA):
    profile=json.loads((data/'active-profile.json').read_text(encoding='utf8'))
    relative=profile.get('indexDirectory','')
    if profile.get('scope')!=SCOPE or not re.fullmatch(r'menswear-rebuild-v1/(?:index|indexes/[a-f0-9]{16})',relative):
        raise RuntimeError('검증한 남성복 인덱스 경로를 확인해주세요.')
    target=(data/relative).resolve()
    if not target.is_relative_to(data.resolve()):raise RuntimeError('Index is outside the data directory')
    return target


def activate_index(index_root,data=DATA):
    from backend.retrieval import Retriever
    relative=index_root.resolve().relative_to(data.resolve()).as_posix()
    if not re.fullmatch(r'menswear-rebuild-v1/indexes/[a-f0-9]{16}',relative):
        raise ValueError('Only immutable namespaced indexes can be published')
    checked=Retriever(index_root)
    profile=dict(scope=SCOPE,indexDirectory=relative,version='menswear-v1',corpusSha256=checked.index['chunksSha256'])
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf8',dir=data,prefix='active-profile-',suffix='.tmp',delete=False) as stream:
        json.dump(profile,stream);stream.flush();os.fsync(stream.fileno());temporary=Path(stream.name)
    temporary.replace(data/'active-profile.json')
    return profile
