import pickle
import math
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from data_loader import WorkDataLoader
from typing import List
import random
import yaml

test_name = 'Test 3'

with open('workers_config.yaml', 'r') as f:
    cfg = yaml.safe_load(f)
industry_names = cfg['industries']

def load_pre_shuffled_state(test_name: str):
    """Load the network and area pools from a pickle file."""
    # fname = f'run_outputs/{test_name}_initial_state.pkl'
    # fname = f'run_outputs/{test_name}_intermediate_state.pkl'
    # fname = f'run_outputs/{test_name}_final_state.pkl'
    fname = f'hcp_outputs/{test_name}_final_state.pkl'

    with open(fname, 'rb') as f:
        state = pickle.load(f)
    return state['lads'], state['network'], state['area_pools']

def extract_raw_distances(area_pools):
    """
    Return a list of all worker distances, preserving None/NaN for no usual workplace.
    """
    distances = [
        w.distance
        for pool in area_pools.values()
        for w in pool.workers.values()
    ]
    return distances

def categorize_distances(distances):
    """
    Separate distances into counts for:
    - no fixed workplace (None or NaN)
    - work from home (0)
    - positive distances
    Returns (nan_count, wfh_count, positive_list).
    """
    nan_count = sum(d is None or (isinstance(d, float) and math.isnan(d)) for d in distances)
    wfh_count = sum(d == 0 for d in distances if d is not None)
    positive = [d for d in distances if d is not None and d > 0]
    return nan_count, wfh_count, positive

def plot_distance_histogram(distances):
    """
    Plot histogram with:
    - a separate bar for 'No usual workplace'
    - a separate bar for 'Work from home'
    - 1 km bins for positive distances
    """
    nan_count, wfh_count, positive = categorize_distances(distances)

    # Define 1 km bin edges up to the maximum observed positive distance
    max_dist = max(positive) if positive else 0
    last_edge = math.ceil(max_dist / 1000) * 1000
    bin_edges = np.arange(0, last_edge + 1000, 1000)

    counts, _ = np.histogram(positive, bins=bin_edges)

    # Labels: first nan, then WFH, then each 1 km bin
    labels = ['No fixed workplace', 'Work from home'] + [f'{i}-{i+1} km' for i in range(len(counts))]
    values = [nan_count, wfh_count] + counts.tolist()

    plt.figure(figsize=(12, 6))
    plt.bar(range(len(values)), values, edgecolor='black')
    plt.xticks(range(len(values)), labels, rotation=45, ha='right')
    plt.xlabel('Distance Category')
    plt.ylabel('Number of Workers')
    plt.title('Worker Distances to Work\n(No workplace, WFH, then 1 km bins)')
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.show()

def load_observed_proportions(lads: List[str], ordered_cats):
    """Load and normalize observed proportions for given distance categories."""
    loader = WorkDataLoader(lads)
    data = loader.load_all()
    df = data['distance_age']
    sum_by_distance = df.groupby(level=1).sum().sum(axis=1)
    obs_vals = sum_by_distance.reindex(ordered_cats).fillna(0).values
    total = obs_vals.sum()
    return obs_vals / total if total > 0 else obs_vals

def compute_sampled_counts(distances, ordered_cats):
    """Bin sampled distances into predefined categories and normalize counts."""
    thresholds = {
        'Mainly work from home': lambda d: d == 0,
        'Less than 2km':        lambda d: 0 < d < 2000,
        '2km to less than 5km': lambda d: 2000 <= d < 5000,
        '5km to less than 10km':lambda d: 5000 <= d < 10000,
        '10km to less than 20km':lambda d:10000 <= d < 20000,
        '20km to less than 30km':lambda d:20000 <= d < 30000,
        '30km to less than 40km':lambda d:30000 <= d < 40000,
        '40km to less than 60km':lambda d:40000 <= d < 60000,
        '60km and over':        lambda d: d >= 60000,
    }
    counts = {cat: 0 for cat in ordered_cats}
    for d in distances:
        if d is None or (isinstance(d, float) and math.isnan(d)):
            counts['Other - No fixed place of work or working outside the UK'] += 1
        else:
            for cat, cond in thresholds.items():
                if cond(d):
                    counts[cat] += 1
                    break
    vals = np.array([counts[cat] for cat in ordered_cats], dtype=float)
    total = vals.sum()
    return vals / total if total > 0 else vals

def plot_normalized_comparison(obs_norm, pred_norm, display_labels):
    """Plot side-by-side bar chart of normalized observed vs. sampled proportions."""
    x = np.arange(len(display_labels))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, obs_norm, width, label='Observed', edgecolor='black')
    plt.bar(x + width/2, pred_norm, width, label='Sampled', edgecolor='black')
    plt.xticks(x, display_labels, rotation=45, ha='right')
    plt.xlabel('Distance Category')
    plt.ylabel('Proportion of Workers')
    plt.title('Observed vs. Sampled Proportions by Distance Category')
    plt.legend()
    plt.grid(axis='y', linestyle='--')
    plt.tight_layout()
    plt.show()

def plot_capacity_vs_employees(network):
    """Log–log scatter of target capacity vs. employee count per industry."""
    all_containers = [cont for sublist in network.containers for cont in sublist if cont]
    groups = defaultdict(list)
    for cont in all_containers:
        code_num = cont.industry_code
        groups[code_num].append(cont)

    codes_sorted = sorted(groups)
    cols, rows = 3, math.ceil(len(codes_sorted)/3)
    fig, axs = plt.subplots(rows, cols, figsize=(cols*5, rows*4), sharex=True, sharey=True)

    for idx, code in enumerate(codes_sorted):
        r, c = divmod(idx, cols)
        ax = axs[r, c] if rows>1 else axs[c]
        conts = groups[code]
        caps = [c.target_capacity for c in conts]
        emps = [c.num_employees for c in conts]

        ax.scatter(caps, emps, marker='x')
        ax.set_xscale('log'); ax.set_yscale('log')
        maxv = max(caps+emps); mp = 10**math.ceil(math.log10(maxv))
        ticks = [10**e for e in range(int(math.log10(mp))+1)]

        ax.plot(ticks, ticks, linestyle='--')
        ax.set_xlim(1, mp); ax.set_ylim(1, mp)
        ax.set_xticks(ticks); ax.set_yticks(ticks)
        ax.set_xlabel('Target Capacity'); ax.set_ylabel('Number of Employees')
        ax.set_title(f"{code}: {industry_names[conts[0].industry_code]}")
        ax.grid(True, which='both', linestyle=':')

    # clean up any empty subplots
    total = rows*cols
    for i in range(len(codes_sorted), total):
        fig.delaxes(axs.flatten()[i])

    plt.tight_layout()
    plt.show()

def print_random_areas(area_pools: dict, samples: int = 10):
    # pool_items: List[Tuple[area_id, pool]]
    sampled = random.sample(list(area_pools.items()), samples)
    for area_id, pool in sampled:
        print(f"Area {area_id}:")
        print(pool)
        # for id, worker in pool.workers.items():
        #     print(f"{worker}")


def main():
    lads, network, area_pools = load_pre_shuffled_state(test_name)

    # print_random_areas(area_pools)

    # Plot distances histogram including NaN category
    distances = extract_raw_distances(area_pools)
    plot_distance_histogram(distances)

    # Observed vs sampled proportions
    ordered_cats = [
        'Mainly work from home',
        'Less than 2km', '2km to less than 5km', '5km to less than 10km',
        '10km to less than 20km', '20km to less than 30km',
        '30km to less than 40km', '40km to less than 60km',
        '60km and over', 'Other - No fixed place of work or working outside the UK'
    ]
    label_map = {
        'Mainly work from home': 'Work from home',
        'Other - No fixed place of work or working outside the UK': 'No fixed place',
        'Less than 2km': '0 km–2 km', '2km to less than 5km': '2 km–5 km',
        '5km to less than 10km': '5 km–10 km', '10km to less than 20km': '10 km–20 km',
        '20km to less than 30km': '20 km–30 km', '30km to less than 40km': '30 km–40 km',
        '40km to less than 60km': '40 km–60 km', '60km and over': '60 km+'
    }
    display_labels = [label_map[cat] for cat in ordered_cats]

    obs_norm = load_observed_proportions(lads, ordered_cats)
    pred_norm = compute_sampled_counts(distances, ordered_cats)
    plot_normalized_comparison(obs_norm, pred_norm, display_labels)

    # Capacity vs employees scatter
    plot_capacity_vs_employees(network)

if __name__ == '__main__':
    main()
