"""Point-in-time selection and chronological validation primitives."""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Vintage:
    observed_at: int
    available_at: int
    value: float


def asof_value(rows, observed_at, decision_at):
    selected = {}
    for row in rows:
        if not np.isfinite([row.observed_at,row.available_at,row.value]).all():
            raise ValueError('nonfinite vintage')
        if row.observed_at != observed_at or row.available_at > decision_at:
            continue
        if row.available_at in selected and selected[row.available_at] != row.value:
            raise ValueError('conflicting value at same availability time')
        selected[row.available_at] = row.value
    return selected[max(selected)] if selected else None


def purged_walk_forward(times, label_end, min_train, test_size, gap=0):
    """Expanding folds; labels must END strictly before test start minus gap.

    gap uses the same time unit as times. No future training rows, so no
    post-test embargo is needed here. Symmetric CV would require one.
    """
    t,e = np.asarray(times),np.asarray(label_end)
    if t.ndim!=1 or t.shape!=e.shape or len(t)<2 or not np.isfinite(t).all() or not np.isfinite(e).all():
        raise ValueError('finite aligned one-dimensional times required')
    if np.any(np.diff(t)<=0) or np.any(e<t) or min_train<1 or test_size<1 or gap<0:
        raise ValueError('invalid fold geometry')
    start=min_train
    while start<len(t):
        train=np.flatnonzero((np.arange(len(t))<start)&(e<t[start]-gap))
        if len(train)>=min_train:
            yield train,np.arange(start,min(start+test_size,len(t)))
            start+=test_size
        else:
            start+=1
