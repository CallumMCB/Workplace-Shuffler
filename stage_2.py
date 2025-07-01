#!/usr/bin/env python3
import logging
import math
import pickle
import random
import time
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
from multiprocessing import Manager, cpu_count
import argparse

from worker_distributor_shuffle import WorkplaceShuffler
from workplace_network import WorkplaceNetwork
from workplace_container_counter import *
from worker_pool import OutputAreaWorkerPool
from worker import distance_bin_calculator

from tqdm import tqdm

import warnings
warnings.filterwarnings(
    'ignore',
    'Pickle, copy, and deepcopy support will be removed from itertools',
    category=DeprecationWarning
)

# -----------------------------------------------------------------------------
# 1) Logger setup
# -----------------------------------------------------------------------------
def setup_logger(level: int = logging.DEBUG) -> logging.Logger:
    logging.basicConfig(
        level=level,
        format="%(message)s",
        force=True,
    )
    return logging.getLogger(__name__)

logger = setup_logger()

# -----------------------------------------------------------------------------
# 2) Labels and types
# -----------------------------------------------------------------------------
NAN_LABEL = 'Other - No fixed place of work or working outside the UK'
HOME_LABEL = 'Mainly work from home'
ContainerKey = Optional[Tuple[str, str]]
moveType = Tuple[int, ContainerKey, ContainerKey]

# -----------------------------------------------------------------------------
# 3) Load / save helpers
# -----------------------------------------------------------------------------
def load_state(test_name: str) -> Dict:
    initial = Path(f"run_outputs/{test_name}_initial_state.pkl")
    intermediate = Path(f"run_outputs/{test_name}_intermediate_state.pkl")
    path = intermediate if intermediate.exists() else initial
    conversion_req = True if path == initial else False
    logger.info(f"Loading state from {path}")
    return pickle.loads(path.read_bytes()), conversion_req


def save_state(state: Dict, test_name: str, final: bool = True) -> None:
    suffix = 'final' if final else 'intermediate'
    path = Path(f"run_outputs/{test_name}_{suffix}_state.pkl")
    logger.info(f"Saving state to {path}")
    path.write_bytes(pickle.dumps(state))

# -----------------------------------------------------------------------------
# 4) Wait-for-file helper
# -----------------------------------------------------------------------------
def wait_for_file(path: str, interval: float = 60):
    elapsed = 0
    while not os.path.exists(path):
        logger.info(f"[⏳] waiting for file `{path}` ({elapsed}s elapsed)...")
        time.sleep(interval)
        elapsed += interval
    logger.info(f"[✅] detected `{path}` after {elapsed}s, proceeding.")

# -----------------------------------------------------------------------------
# 5) χ² + Metropolis helpers
# -----------------------------------------------------------------------------
def delta_chi2(obs: float, exp: float, delta: int, bias: int = 1) -> float:
    if exp == 0:
        return (1 + 2 * delta * obs) * 10
    return (1 - 2 * delta * (exp - obs)) / exp * bias


def metropolis_accept(delta_chi2_val: float, temperature: float) -> bool:
    if delta_chi2_val <= 0:
        return True
    return random.random() < math.exp(-0.5 * delta_chi2_val / temperature)

# -----------------------------------------------------------------------------
# 6) Merge net-moves into master network
# -----------------------------------------------------------------------------
def merge_moves(
    master_net: WorkplaceNetwork,
    moves: Dict[int, Tuple[ContainerKey, ContainerKey]],
) -> None:
    lookup = {(c.iz_code, c.industry_code): c for c in master_net._flat_containers}
    for gid, (orig, final) in moves.items():
        if orig:
            lookup[orig].remove_employee(gid)
        if final:
            lookup[final].add_employee(gid)

# -----------------------------------------------------------------------------
# 7) Worker-batch MH on just sub-pools
# -----------------------------------------------------------------------------
def run_mh_batch_net(
    pools: Dict[int, OutputAreaWorkerPool],
    area_ids: List[int],
    n_steps: int,
    T0: float,
    cooling_rate: float,
    distance_weight: float
) -> Tuple[Dict[int, Tuple[ContainerKey, ContainerKey]], Dict[int, OutputAreaWorkerPool]]:
    # pools contains only the sub-pools for this batch
    shufflers = {aid: WorkplaceShuffler(pools[aid]) for aid in area_ids}
    pool_items = list(pools.items())

    first_last: Dict[int, Tuple[ContainerKey, ContainerKey]] = {}
    T = T0
    for _ in range(n_steps):
        aid, pool = random.choice(pool_items)
        worker = pool.get_random_worker()
        if worker is None:
            continue
        gid = worker.global_id
        orig_site = worker.workplace_container
        orig_dist = worker.distance_bin

        raw_dist_10, prop_site = shufflers[aid].new_proposal(worker.industry_code)
        if raw_dist_10 != raw_dist_10:
            continue
        dist = None if raw_dist_10 is None else int(raw_dist_10) * 10
        prop_dist = distance_bin_calculator(dist)

        delta = 0.0
        if prop_dist != orig_dist:
            exp_o, obs_o = pool.distance_counts.loc[orig_dist]
            exp_p, obs_p = pool.distance_counts.loc[prop_dist]
            b_o = 2 if orig_dist in (NAN_LABEL, HOME_LABEL) else 1
            b_p = 2 if prop_dist in (NAN_LABEL, HOME_LABEL) else 1
            delta += (delta_chi2(obs_o, exp_o, -1, b_o)
                      + delta_chi2(obs_p, exp_p, +1, b_p)) * distance_weight
        if orig_site:
            delta += delta_chi2(orig_site.num_employees, orig_site.target_capacity, -1)
        if prop_site:
            delta += delta_chi2(prop_site.num_employees, prop_site.target_capacity, +1)

        if not metropolis_accept(delta, T):
            T *= cooling_rate
            continue

        if orig_site:
            orig_site.remove_employee(gid)
            pool.distance_counts.at[orig_dist, 'sampled'] -= 1
            from_key = (orig_site.iz_code, orig_site.industry_code)
        else:
            from_key = None
        if prop_site:
            prop_site.add_employee(gid)
            pool.distance_counts.at[prop_dist, 'sampled'] += 1
            to_key = (prop_site.iz_code, prop_site.industry_code)
        else:
            to_key = None

        worker.workplace_container = prop_site
        worker.distance_10 = raw_dist_10
        first = first_last.get(gid, (from_key, None))[0]
        first_last[gid] = (first, to_key)
        T *= cooling_rate

    # return updated pools
    return first_last, pools

# -----------------------------------------------------------------------------
# 8) Orchestrator: send only sub-pools to workers
# -----------------------------------------------------------------------------
def main_parallel_dynamic(
    test_name: str,
    steps_per_batch: int = 50_000,
    T0: float = 1.0,
    cooling_rate: float = 1 - 1e-7,
    distance_weight: float = 2.0,
    batch_size: int = 10,
    max_workers: int = None,
    num_batches: int = None
):
    initial_file = f"run_outputs/{test_name}_initial_state.pkl"
    wait_for_file(initial_file)

    state, conversion_req = load_state(test_name)
    master_net: WorkplaceNetwork = state['network']
    master_pools: Dict[int, OutputAreaWorkerPool] = state['area_pools']
    master_lads = state['lads']
    if conversion_req:
        convert_workplace_containers(master_net.containers)

    area_ids = list(master_pools.keys())
    mgr = Manager()
    area_queue = mgr.Queue()
    for aid in area_ids:
        area_queue.put(aid)

    if max_workers is None:
        max_workers = cpu_count()

    completed = 0
    total = num_batches or float('inf')

    print(max_workers)

    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        futures = set()
        # launch initial batches
        for _ in range(min(max_workers, len(area_ids), int(total - completed))):
            batch = [area_queue.get() for _ in range(min(batch_size, area_queue.qsize()))]
            if not batch:
                break
            sub_pools = {aid: master_pools[aid] for aid in batch}
            futures.add(
                exe.submit(
                    run_mh_batch_net,
                    sub_pools,
                    batch,
                    steps_per_batch,
                    T0,
                    cooling_rate,
                    distance_weight
                )
            )

        with tqdm(total=total, desc="Batches completed") as pbar:
            while futures and completed < total:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for fut in done:
                    futures.remove(fut)
                    moves, local_pools = fut.result()
                    merge_moves(master_net, moves)
                    # update master pools with returned sub-pools
                    for aid, pool in local_pools.items():
                        pool.network = master_net
                        master_pools[aid] = pool
                    for pool in master_pools.values():
                        pool.initialize_sampled_distances()

                    completed += 1
                    pbar.update(1)

                    # periodic save
                    if completed % max_workers == 0:
                        save_state({'lads': master_lads, 'network': master_net, 'area_pools': master_pools}, test_name, final=False)

                    # schedule next batch
                    if area_queue.qsize() >= batch_size and completed < total:
                        next_batch = [area_queue.get() for _ in range(batch_size)]
                        sub_pools = {aid: master_pools[aid] for aid in next_batch}
                        futures.add(
                            exe.submit(
                                run_mh_batch_net,
                                sub_pools,
                                next_batch,
                                steps_per_batch,
                                T0,
                                cooling_rate,
                                distance_weight
                            )
                        )

    save_state({'lads': master_lads, 'network': master_net, 'area_pools': master_pools}, test_name, final=True)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the parallel dynamic test harness"
    )
    parser.add_argument(
        "--test_name",
        type=str,
        default="Test 3",
        help="Name of this test run"
    )
    parser.add_argument(
        "--steps_per_batch",
        type=int,
        default=5000,
        help="Number of steps per batch"
    )
    parser.add_argument(
        "--T0",
        type=float,
        default=0.5,
        help="Initial temperature"
    )
    parser.add_argument(
        "--cooling_rate",
        type=float,
        default=1 - 1e-7,
        help="Multiplicative cooling rate per step"
    )
    parser.add_argument(
        "--distance_weight",
        type=float,
        default=2.0,
        help="Weighting factor for distance in the cost function"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=400,
        help="Number of samples per batch"
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Maximum number of worker threads/processes"
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=100,
        help="Total number of batches to process"
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    main_parallel_dynamic(
        test_name=args.test_name,
        steps_per_batch=args.steps_per_batch,
        T0=args.T0,
        cooling_rate=args.cooling_rate,
        distance_weight=args.distance_weight,
        batch_size=args.batch_size,
        max_workers=args.max_workers,
        num_batches=args.num_batches
    )
