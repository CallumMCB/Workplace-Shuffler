from dataclasses import dataclass
from tqdm import tqdm
import numpy as np

@dataclass(slots=True)
class CountContainer:
    iz_code: str
    industry_code: str
    target_capacity: np.uint32
    num_employees: np.uint32

    def add_employee(self, _):      # Ignore the Global ID
        self.num_employees += 1

    def remove_employee(self, _):
        self.num_employees -= 1

    @property
    def deficit(self):
        return self.target_capacity - self.num_employees

def convert_workplace_containers(master_containers):
    """
    Replace each full WorkplaceContainer with a slim CountContainer,
    showing progress over each IZ row.
    """
    total_rows = len(master_containers)
    for iz, row in enumerate(tqdm(master_containers, total=total_rows, desc="Converting IZ rows")):
        for ind_i, cont in enumerate(row):
            if cont is None:
                continue
            slim = CountContainer(
                iz_code         = cont.iz_code,
                industry_code   = cont.industry_code,
                target_capacity = cont.target_capacity,
                num_employees   = cont.num_employees,
            )
            master_containers[iz][ind_i] = slim