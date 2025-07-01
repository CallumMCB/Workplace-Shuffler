from dataclasses import dataclass, field, fields
from typing import Tuple, Any
from worker_pool import OutputAreaWorkerPool
from workplace_network import WorkplaceNetwork

@dataclass(slots=True)
class BaseWorkerDistributor:
    area_pool: OutputAreaWorkerPool

    network:      WorkplaceNetwork = field(init=False)
    counter:      Any              = field(init=False)
    area_id_root: int              = field(init=False)

    def __post_init__(self):
        object.__setattr__(self, 'network',      self.area_pool.network)
        object.__setattr__(self, 'counter',      self.area_pool._id_counter)
        object.__setattr__(self, 'area_id_root', self.area_pool.area_id_root)

    def __getstate__(self) -> Tuple:
        return tuple(getattr(self, f.name) for f in fields(self))

    def __setstate__(self, state: Tuple) -> None:
        for f, val in zip(fields(self), state):
            object.__setattr__(self, f.name, val)

