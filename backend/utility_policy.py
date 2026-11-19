"""Explicit, auditable ranking policies over paired-case retrieval features.

These weights belong to the paired-description schema. A policy from this
schema is not silently reused for the mobile application's style annotations.
"""
from dataclasses import asdict,dataclass
import numpy as np

FEATURE_SCHEMA='paired-description-utility-v2'

@dataclass(frozen=True)
class UtilityPolicy:
    alpha: float = 2/3
    relative_floor: float = 0

    def __post_init__(self):
        if not np.isfinite(self.alpha) or not 0 <= self.alpha <= 1:
            raise ValueError('alpha must be in [0,1]')
        if not np.isfinite(self.relative_floor) or not 0 <= self.relative_floor <= 1:
            raise ValueError('relative_floor must be in [0,1]')

    def to_dict(self):return {**asdict(self),'featureSchema':FEATURE_SCHEMA}

def utility_features(candidate_pairs,existing_pairs):
    matrix=np.asarray(candidate_pairs,float)
    existing=np.asarray(existing_pairs,float)
    if matrix.ndim!=2 or matrix.shape[0]==0 or matrix.shape[1]==0:
        raise ValueError('A nonempty candidate/context matrix is required')
    if existing.size==0:existing=np.empty((0,matrix.shape[1]))
    elif existing.ndim==1:existing=existing.reshape(1,-1)
    if existing.ndim!=2 or existing.shape[1]!=matrix.shape[1]:
        raise ValueError('Existing items must use the same contexts')
    if not np.isfinite(matrix).all() or not np.isfinite(existing).all() or (matrix<0).any() or (existing<0).any():
        raise ValueError('Similarity must be finite and nonnegative')
    best=existing.max(axis=0) if len(existing) else np.zeros(matrix.shape[1])
    return matrix.mean(axis=1),np.maximum(matrix-best,0).mean(axis=1)

def policy_scores(combination,marginal,policy):
    c,m=np.asarray(combination,float),np.asarray(marginal,float)
    if c.ndim!=1 or c.size==0 or c.shape!=m.shape or not np.isfinite(c).all() or not np.isfinite(m).all():
        raise ValueError('Finite feature vectors with matching lengths are required')
    if (c<0).any() or (m<0).any() or (m>c+1e-10).any():
        raise ValueError('Marginal support must be between zero and combination support')
    eligible=c>=policy.relative_floor*c.max()-1e-12
    scores=policy.alpha*c+(1-policy.alpha)*m
    return np.where(eligible,scores,-np.inf)

def tie_aware_metrics(scores,observed_pairs,existing_pairs):
    scores=np.asarray(scores,float);pairs=np.asarray(observed_pairs,float);existing=np.asarray(existing_pairs,float)
    if pairs.ndim!=2 or pairs.shape[0]!=scores.size or existing.shape!=(pairs.shape[1],):
        raise ValueError('Reference dimensions do not match candidate scores')
    top=np.isclose(scores,scores.max(),rtol=0,atol=1e-7)
    if not top.any():raise ValueError('No eligible candidate')
    new=np.maximum(pairs-existing,0).sum(axis=1)
    total=pairs.sum(axis=1)
    gain=float(new[top].mean())
    return {'newLinks':gain,'totalLinks':float(total[top].mean()),
            'regret':float(new.max()-gain),'topTies':int(top.sum()>1),
            'topIndices':np.flatnonzero(top).tolist()}
