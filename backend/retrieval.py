import hashlib
import json
import math
import re
import threading
from collections import Counter
from functools import lru_cache
import numpy as np
from backend.config import DATA, RRF_K
from backend.style_constraints import explicit_conditions, matches_conditions

def tokens(text):
    terms = re.findall(r'[a-z0-9]+|[가-힣]+', text.lower())
    return terms + [word[i:i+2] for word in terms if re.search('[가-힣]', word) for i in range(len(word)-1)]

def rrf(rankings, k=RRF_K):
    scores = Counter()
    for ranking in rankings:
        for position, doc_id in enumerate(ranking, 1):
            scores[doc_id] += 1 / (k + position)
    return scores

class Retriever:
    def __init__(self, data_root=None):
        if data_root is None and (DATA/'active-profile.json').exists():
            from backend.index_profile import active_index_path
            data_root=active_index_path()
        root = data_root or DATA
        self.scoped = data_root is not None
        path = root / 'chunks.jsonl'
        self.chunks = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
        self.manifest = json.loads((root / 'manifest.json').read_text(encoding='utf-8'))
        self.index = json.loads((root / 'index.json').read_text(encoding='utf-8'))
        if self.scoped:
            from backend.audience import eligible_evidence, SCOPE
            if self.manifest.get('scope') != SCOPE or any(c['collection']=='style' and not eligible_evidence(c) for c in self.chunks):
                raise RuntimeError('남성복 검색 자료의 범위와 출처 표기를 확인해주세요.')
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.index['chunksSha256']:
            raise RuntimeError('문서가 변경되었습니다. 검색 인덱스를 다시 구축해주세요.')
        for name, field in [('embeddings.npy','embeddingsSha256'),('manifest.json','manifestSha256')]:
            if field in self.index and hashlib.sha256((root/name).read_bytes()).hexdigest()!=self.index[field]:
                raise RuntimeError('검색 자료 파일의 식별값이 다릅니다: '+name)
        self.vectors = np.load(root / 'embeddings.npy', allow_pickle=False)
        if len(self.vectors) != self.index['rows'] or len(self.index['owners']) != len(self.vectors):
            raise RuntimeError('인덱스와 임베딩 개수가 다릅니다.')
        if self.vectors.ndim!=2 or self.vectors.shape[1]!=self.index['dimension'] or not np.isfinite(self.vectors).all() or any(not isinstance(i,int) or i<0 or i>=len(self.chunks) for i in self.index['owners']):
            raise RuntimeError('임베딩 차원·값·문서 연결을 확인해주세요.')
        self.tf = [Counter(tokens(c['title'] + ' ' + c['text'])) for c in self.chunks]
        self.df = Counter(term for counts in self.tf for term in counts)
        self.average_length = sum(sum(c.values()) for c in self.tf) / max(len(self.tf), 1)
        self._model = None
        self.lock = threading.Lock()

    def model(self):
        if self._model is None:
            with self.lock:
                if self._model is None:
                    from sentence_transformers import SentenceTransformer
                    self._model = SentenceTransformer(self.index['model'], revision=self.index['revision'], device='cpu', trust_remote_code=False, local_files_only=True)
        return self._model

    @lru_cache(maxsize=512)
    def query_vector(self, query):
        with self.lock:
            return self._model.encode([query], normalize_embeddings=True)[0]

    def search(self, query, collection='research', mode='hybrid', limit=5, source_dedup=True, include_historical=False, predicate=None):
        # Explicit garment conditions are requirements, independent of the
        # ranking method. Semantic similarity cannot override a stated color,
        # pattern, wearing occasion or garment direction.
        conditions=explicit_conditions(query) if collection=='style' else {}
        indices = [i for i, c in enumerate(self.chunks) if c['collection'] == collection
                   and (collection != 'research' or include_historical or c['kind'] == 'paper' or c.get('currentScope'))
                   and (predicate is None or predicate(c))
                   and (not conditions or matches_conditions(c,conditions))]
        if not indices:
            return []
        pool_limit = len(indices) if predicate else 40
        terms = Counter(tokens(query))
        lexical = {}
        for i in indices:
            counts = self.tf[i]
            length = sum(counts.values())
            score = 0
            for term in terms:
                f = counts[term]
                if f:
                    idf = math.log(1 + (len(self.chunks) - self.df[term] + 0.5) / (self.df[term] + 0.5))
                    score += idf * f * 2.2 / (f + 1.2 * (0.25 + 0.75 * length / self.average_length))
            if score > 0:
                lexical[i] = score
        lexical_rank = sorted(lexical, key=lambda i: (-lexical[i], self.chunks[i]['id']))[:pool_limit]
        semantic = {}
        if mode != 'keyword':
            self.model()
            similarities = self.vectors @ self.query_vector(query)
            allowed = set(indices)
            for row, owner in enumerate(self.index['owners']):
                if owner in allowed:
                    semantic[owner] = max(semantic.get(owner, -1), float(similarities[row]))
        semantic_rank = sorted(semantic, key=lambda i: (-semantic[i], self.chunks[i]['id']))[:pool_limit]
        ranks = [lexical_rank] if mode == 'keyword' else [semantic_rank] if mode == 'vector' else [lexical_rank, semantic_rank]
        combined = rrf(ranks)
        if mode in {'hybrid','conditioned'} and collection == 'research':
            # Lexical acronym anchors favour their primary paper, without consulting qrels.
            anchors = {'rrf':'rrf_2009', 'rag':'rag_2020', 'beir':'beir_2021', 'fashionclip':'fashionclip_2022',
                       'outfittransformer':'outfittransformer_2023', 'text2outfit':'text2outfit_2025'}
            preferred = {source for term, source in anchors.items() if term in terms}
            for i in combined:
                if self.chunks[i]['sourceId'] in preferred:
                    combined[i] *= 1.5
        selected, seen = [], set()
        for i in sorted(combined, key=lambda i: (-combined[i], self.chunks[i]['id'])):
            chunk = self.chunks[i]
            if source_dedup and chunk['sourceId'] in seen:
                continue
            seen.add(chunk['sourceId'])
            selected.append({**chunk, 'retrieval': {'bm25': round(lexical.get(i, 0), 4), 'cosine': round(semantic.get(i, 0), 4), 'rrf': round(combined[i], 6)}})
            if len(selected) >= limit:
                break
        return selected
