import sys
import logging
from itertools import count
from dataclasses import dataclass, field, fields
from typing import List, Dict, Optional, Tuple
import yaml

import apportionment.methods as app
import numpy as np
import pandas as pd
import random

from workplace_container import WorkplaceContainer
from workplace_network import WorkplaceNetwork
from worker import Worker

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    force=True,
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

# Load configuration and precompute constants
with open('workers_config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

CONVERSION: Dict[str, List[str]] = {
    k: (v if isinstance(v, list) else [v])
    for k, v in cfg['conversion'].items()
}
wfh = cfg['wfh']
nwp = cfg['nwp']
dist_lookup = cfg['dist_conversion']

@dataclass(slots=True)
class OutputAreaWorkerPool:
    area_id: np.uint32
    iz11: str
    area_coord: np.ndarray
    industry_counts: pd.Series
    distance_age_counts: pd.DataFrame
    method_age_counts: pd.DataFrame
    network: WorkplaceNetwork

    # automatically‐initialized fields
    _id_counter: count = field(init=False, default_factory=lambda: count(1))
    workers: Dict[np.uint16, Worker] = field(init=False, default_factory=dict)
    _worker_ids: List[int] = field(init=False, default_factory=list)

    # combined expected vs sampled distance‐cxounts
    distance_labels: List[str] = field(init=False)
    distance_weights: np.ndarray = field(init=False)
    distance_counts: pd.DataFrame = field(init=False)

    def __post_init__(self):
        # 1) compute “expected” distance distribution via Hamilton apportionment
        total = int(self.industry_counts.sum())
        dist_idx = self.distance_age_counts.index
        float_seats = self.distance_age_counts.sum(axis=1).values.tolist()

        if total > 0:
            # only apportion when there’s at least one seat
            alloc_list = app.compute(
                "hamilton",
                float_seats,
                total,
                dist_idx.tolist(),
                verbose=False
            )
            expected = pd.Series(alloc_list, index=dist_idx, dtype=int)
        else:
            # no seats ⇒ all expected counts zero
            expected = pd.Series(0, index=dist_idx, dtype=int)

        # 2) build the DataFrame: expected counts + blank sampled column
        self.distance_counts = pd.DataFrame({
            "expected": expected,
            "sampled": pd.NA
        })
        exp = self.distance_counts.loc[:, 'expected'].loc[lambda s: s > 0]
        self.distance_labels = exp.index.tolist()
        self.distance_weights = exp.values.astype(np.uint16)

    def add_worker(
        self,
        local_id: int,
        industry_code: str,
        container: Optional[WorkplaceContainer] = None,
        distance_10: Optional[np.uint16] = None
    ) -> Tuple[int, int]:
        self._worker_ids.append(local_id)
        global_id = self.area_id_root + local_id

        self.workers[local_id] = Worker(
            global_id=global_id,
            industry_code=industry_code,
            distance_10=distance_10,
            workplace_container=container,
        )
        if container is not None:
            container.add_employee(global_id)

        return local_id, global_id

    def remove_worker(self, local_id: int) -> None:
        assn = self.workers.pop(local_id)
        self._worker_ids.remove(local_id)
        iz_i = self.network.iz_idx_map[assn.iz_code]
        ind_i = self.network.ind_idx_map[assn.industry_code]
        self.network.containers[iz_i][ind_i].remove_employee(assn.global_id)

    def get_random_worker(self) -> Worker:
        if not self._worker_ids:
            return None
        local_id = random.choice(self._worker_ids)
        return self.workers[local_id]

    def initialize_sampled_distances(self) -> pd.Series:
        sampled = self._compute_sampled_distances()
        self.distance_counts["sampled"] = sampled
        return sampled

    @property
    def sampled_distance_counts(self) -> pd.Series:
        if pd.isna(self.distance_counts["sampled"]).all():
            self.initialize_sampled_distances()
        return self.distance_counts["sampled"]

    def _compute_sampled_distances(self) -> pd.Series:
        labels = self.distance_counts.index
        cat = pd.Categorical(
            [w.distance_bin for w in self.workers.values()],
            categories=labels,
            ordered=True
        )
        counts = pd.Series(cat).value_counts(sort=False).astype(int)
        return counts

    @property
    def area_id_root(self):
        return self.area_id * 10_000

    def __getstate__(self):
        return tuple(getattr(self, f.name) for f in fields(self))

    def __setstate__(self, state):
        for f, val in zip(fields(self), state):
            object.__setattr__(self, f.name, val)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}\n"
            f"  area_id:         {self.area_id}\n"
            f"  industry counts: {self.industry_counts}\n"
            f"  distance counts:\n{self.distance_counts}\n"
        )
