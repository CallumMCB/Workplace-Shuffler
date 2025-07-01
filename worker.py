from dataclasses import dataclass, fields
from typing import Optional, Tuple
from workplace_container import WorkplaceContainer
from bisect import bisect_left
import numpy as np
import math

# Class‐level bin edges and labels
BIN_EDGES: Tuple[float, ...] = (
    0, 2_000, 5_000, 10_000, 20_000, 30_000, 40_000, 60_000
)
BIN_LABELS: Tuple[str, ...] = (
    "Mainly work from home",  # exactly zero
    "Less than 2km",  # (0, 2 km
    "2km to less than 5km",
    "5km to less than 10km",
    "10km to less than 20km",
    "20km to less than 30km",
    "30km to less than 40km",
    "40km to less than 60km",
    "60km and over",
)
NAN_LABEL: str = "Other - No fixed place of work or working outside the UK"

def distance_bin_calculator(d):
    """
     - NaN/None → NAN_LABEL
     - 0.0      → BIN_LABELS[0]
     - >0       → pick via bisect_left and clamp to the last label
    """
    # 1) missing
    if d is None:
        return NAN_LABEL

    # 2) exactly zero
    if d == 0:
        return BIN_LABELS[0]

    # 3) positive: find insertion point among edges
    idx = bisect_left(BIN_EDGES, d)
    # bisect_left returns:
    #   1..len(BIN_EDGES)-1 for values within the defined edges,
    #   len(BIN_EDGES) for values beyond the last edge
    if idx >= len(BIN_LABELS):
        return BIN_LABELS[-1]
    return BIN_LABELS[idx]

@dataclass(slots=True)
class Worker:
    global_id: np.uint32
    industry_code: np.uint8
    distance_10: Optional[np.uint16] = None
    workplace_container: Optional[WorkplaceContainer] = None

    @property
    def distance(self) -> Optional[int]:
        """Actual distance in metres (rounded to nearest 10m)."""
        if self.distance_10 is None:
            return None
        # int() just casts the numpy.uint16 → Python int
        return int(self.distance_10) * 10

    @property
    def distance_bin(self) -> str:
        return distance_bin_calculator(self.distance)

    def __getstate__(self):
        return tuple(getattr(self, f.name) for f in fields(self))

    def __setstate__(self, state):
        for f, val in zip(fields(self), state):
            object.__setattr__(self, f.name, val)

    def __repr__(self):
        return (
            f"{self.global_id}: \n"
            f"      Industry Code: {self.industry_code}\n"
            f"      Distance:      {self.distance/1000}km\n"
            f"      Workplace:     {self.workplace_container}"
        )

