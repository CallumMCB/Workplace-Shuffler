from dataclasses import dataclass, field, fields
from typing import Tuple, Set
import random
import numpy as np

@dataclass(slots=True)
class WorkplaceContainer:
    iz_code: str
    industry_code: str
    target_capacity: np.uint32
    employees: Set[int] = field(default_factory=set)

    def add_employee(self, person_id: int) -> None:
        self.employees.add(person_id)

    def remove_employee(self, person_id: int) -> None:
        self.employees.remove(person_id)

    @property
    def num_employees(self) -> int:
        return len(self.employees)

    @property
    def deficit(self) -> int:
        return self.target_capacity - self.num_employees

    def pop_random_employee(self) -> Tuple[int, int, int]:
        emp_id = random.choice(tuple(self.employees))
        self.employees.remove(emp_id)

        divisor = 10_000
        area_id, worker_id = divmod(emp_id, divisor)
        return area_id, worker_id, emp_id

    def __getstate__(self):
        return tuple(getattr(self, f.name) for f in fields(self))

    def __setstate__(self, state):
        for f, val in zip(fields(self), state):
            object.__setattr__(self, f.name, val)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__} ("
            f"iz_code={self.iz_code!r}, "
            f"industry={self.industry_code!r}, "
            f"headcount={self.num_employees}/{self.target_capacity}, "
            f"deficit={self.deficit}"
            f")"
        )