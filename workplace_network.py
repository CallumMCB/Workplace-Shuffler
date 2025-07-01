import sys
import logging
from dataclasses import dataclass, field, fields
from typing import List, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

from workplace_container import WorkplaceContainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    force=True,
    stream=sys.stderr
)
logger = logging.getLogger(__name__)

@dataclass(slots=True)
class WorkplaceNetwork:
    """
    Optimized network: integer indexing, tuned BallTree, and vectorized industry filtering.
    Supports nearest and multi-industry queries.
    """
    workplace_pop: pd.DataFrame           # index: iz_code, cols: industry_code
    iz_centroids: pd.DataFrame            # index: iz_code, ['x','y'] coords
    industry_subset: Optional[List[str]] = None
    leaf_size: int = 5

    # post-init fields (must be declared for slots)
    iz_codes: List[str] = field(init=False)
    iz_idx_map: Dict[str,int] = field(init=False)
    ind_codes: List[str] = field(init=False)
    ind_idx_map: Dict[str,int] = field(init=False)
    containers: List[List[Optional[WorkplaceContainer]]] = field(init=False)
    _coords: np.ndarray = field(init=False)
    _iz_idxs: np.ndarray = field(init=False)
    _ind_idxs: np.ndarray = field(init=False)
    _caps: np.ndarray = field(init=False)
    tree_all: BallTree = field(init=False)

    def __post_init__(self):
        # 1. Map IZ codes and industry codes to integer indices
        self.iz_codes = list(self.workplace_pop.index)
        self.iz_idx_map = {iz: i for i, iz in enumerate(self.iz_codes)}

        all_inds = list(self.workplace_pop.columns)
        self.ind_codes = [ind for ind in all_inds if (not self.industry_subset or ind in self.industry_subset)]
        self.ind_idx_map = {ind: i for i, ind in enumerate(self.ind_codes)}

        # 2. Build containers matrix and flatten lists
        n_iz, n_ind = len(self.iz_codes), len(self.ind_codes)
        self.containers = [[None] * n_ind for _ in range(n_iz)]
        coords_list, iz_idxs, ind_idxs = [], [], []

        for iz in self.iz_codes:
            iz_i = self.iz_idx_map[iz]
            centroid = tuple(self.iz_centroids.loc[iz])
            for ind in self.ind_codes:
                ind_i = self.ind_idx_map[ind]
                cap = int(self.workplace_pop.at[iz, ind])
                if cap > 0:
                    cont = WorkplaceContainer(iz, ind, cap)
                    self.containers[iz_i][ind_i] = cont
                    coords_list.append(centroid)
                    iz_idxs.append(iz_i)
                    ind_idxs.append(ind_i)

        # 3. Convert to numpy arrays
        self._coords   = np.array(coords_list)
        self._iz_idxs  = np.array(iz_idxs, dtype=np.int32)
        self._ind_idxs = np.array(ind_idxs, dtype=np.int32)
        self._caps     = np.array([c.target_capacity for row in self.containers for c in row if c is not None], dtype=np.int32)

        # 4. Build global BallTree
        self.tree_all = BallTree(self._coords, metric='euclidean', leaf_size=self.leaf_size)

    @property
    def _flat_containers(self) -> List[WorkplaceContainer]:
        """
        Flattened list of all non-None WorkplaceContainer instances, in sync with coords.
        """
        return [cont for row in self.containers for cont in row if cont is not None]

    def query_nearby(
        self,
        coords: Union[np.ndarray, List[float]],
        min_dist_m: float,
        max_dist_m: float,
        industry_code: Optional[Union[str, List[str]]] = None
    ) -> Tuple[np.ndarray, np.ndarray, List[WorkplaceContainer]]:
        coords = np.atleast_2d(np.asarray(coords))
        idxs_arr, dists_arr = self.tree_all.query_radius(coords, r=max_dist_m, return_distance=True)
        idxs, dists = idxs_arr[0], dists_arr[0]

        # Filter by distance
        valid = (dists >= min_dist_m)
        idxs, dists = idxs[valid], dists[valid]

        # Filter by industry if requested
        if industry_code is not None:
            inds = industry_code if isinstance(industry_code, list) else [industry_code]
            allowed = [self.ind_idx_map[i] for i in inds if i in self.ind_idx_map]
            mask = np.isin(self._ind_idxs[idxs], allowed)
            idxs, dists = idxs[mask], dists[mask]

        # Gather outputs
        distances_10 = (np.rint(dists / 10)).astype(np.uint16)
        capacities = self._caps[idxs]
        containers = [self._flat_containers[i] for i in idxs]
        return distances_10, capacities, containers

    def get_containers_by_iz_code(
            self,
            iz_code: str,
            industry_code: Optional[Union[str, List[str]]] = None
    ) -> List[Tuple[WorkplaceContainer, int]]:
        """
        Returns all WorkplaceContainer instances in the given IZ code
        matching optional industries, paired with their target_capacity.
        """
        iz_i = self.iz_idx_map.get(iz_code)
        if iz_i is None:
            return []

        capacities = []
        containers = []
        row = self.containers[iz_i]
        if industry_code is None:
            for cont in row:
                if cont is not None:
                    containers.append(cont)
                    capacities.append(cont.target_capacity)
            return np.zeros_like(containers), containers, capacities

        codes = industry_code if isinstance(industry_code, list) else [industry_code]
        for code in codes:
            ind_i = self.ind_idx_map.get(code)
            if ind_i is not None:
                cont = row[ind_i]
                if cont is not None:
                    containers.append(cont)
                    capacities.append(cont.target_capacity)
        return np.zeros_like(containers), capacities, containers

    def __getstate__(self):
        return tuple(getattr(self, f.name) for f in fields(self))

    def __setstate__(self, state):
        for f, val in zip(fields(self), state):
            object.__setattr__(self, f.name, val)
