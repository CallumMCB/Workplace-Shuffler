# worker_distributor_shuffle.py
import pandas as pd

from worker_distributor_base import BaseWorkerDistributor
from worker_pool import *

@dataclass(slots=True)
class WorkplaceShuffler(BaseWorkerDistributor):
    exp:      pd.Series    = field(init=False)
    _labels:  List[str]    = field(init=False)
    _weights: np.ndarray   = field(init=False)

    def __post_init__(self):
        # CORRECT CALL: pass self into the base initializer
        BaseWorkerDistributor.__post_init__(self)

        # Cache positive-expected distance buckets once
        exp = self.area_pool.distance_counts['expected'].loc[lambda s: s>0]
        object.__setattr__(self, '_labels',  exp.index.tolist())
        object.__setattr__(self, '_weights', exp.values.astype(float))

    def new_proposal(self, industry: str) -> Tuple[Optional[float], Optional[WorkplaceContainer]]:
        labels  = self._labels.copy()
        weights = self._weights.copy()

        while labels and weights.sum()>0:
            probs = weights / weights.sum()
            idx   = np.random.choice(len(labels), p=probs)
            bucket = labels.pop(idx)
            weights = np.delete(weights, idx)

            if bucket == 'Mainly work from home':
                dists_10, caps, conts = self.network.get_containers_by_iz_code(
                    self.area_pool.iz11, industry
                )
                dists_10 = np.zeros(len(conts), dtype=float)
            elif bucket == 'Other - No fixed place of work or working outside the UK':
                return None, None
            else:
                r_min, r_max = dist_lookup[bucket]
                dists_10, _, conts = self.network.query_nearby(
                    self.area_pool.area_coord, r_min, r_max, industry
                )

            valid = [(d,c) for d,c in zip(dists_10, conts) if c.deficit>0]
            if not valid:
                continue

            ds, cs = zip(*valid)
            cap_arr = np.array([c.target_capacity for c in cs], dtype=float)
            j = np.random.choice(len(cs), p=cap_arr/cap_arr.sum())

            return ds[j], cs[j]
        else: return np.nan, None
