
import numpy as np
import argparse
import random
import warnings
import itertools
warnings.filterwarnings("error")

import sys
sys.path.append("..")
from functions.pricing_functions import *
from functions.algorithm import *
from functions.alternative_consumer_decision import *
from functions.baseline import *
from functions.consumer_decision import *
from functions.overall import *

parser = argparse.ArgumentParser()
parser.add_argument("-r", "--repetitions", type=int, help="Number of repetitions", default =15 )
parser.add_argument("-e", "--seed", type=int, help="Seed", default = 0)
parser.add_argument("-s", "--size", type=int, help="Size", default = 4)
parser.add_argument("-p", "--proportion", type=float, help="Proportion", default = 0.5)
parser.add_argument("-cls", "--clustersize", type=int, help="Cluster Size", default = 8)
parser.add_argument("-m", "--maxscale", type=int, help="Max Scale", default = 9)

args = parser.parse_args()

seed = args.seed
random.seed(seed)
repetitions = args.repetitions
size = args.size
proportion = args.proportion
cluster_size = args.clustersize
max_scale = args.maxscale


data_PID = np.loadtxt("../data/Electronics_PID.csv", delimiter=",", dtype=str)
whole_items = [i for i in data_PID]
np_wtp = np.loadtxt("../data/Electronics_wtp.csv", delimiter=",")


# Random 100 consumers 
random_indices = np.random.choice(np_wtp.shape[0], size=100, replace=False)
np_wtp = np_wtp[random_indices]

bundle_sizes = []
print("Experiment Scalability - Bundles")

# Initialize the results dictionary
comparison_results = { 'overall_time': []}

scale_start_time = process_time()
for scale in range(max_scale):
    scale_end_time = process_time()
    overall_time_list = []

    # Initialize the results dictionary
    for _ in range(repetitions):
        target_items = random.sample(whole_items, size + scale)
        indices = [whole_items.index(i) for i in target_items]

        n1 = np.arange(len(np_wtp))
        temp_wtp = np.round(np_wtp[n1[:,None],indices], 2)
        temp_whole_items = target_items.copy()

        total_coverage = np.sum(temp_wtp)


        # Generate powerset
        powerset = set()
        for r in range(2, len(target_items) + 1):  # Start from 2 to exclude single-item sets
            for combination in itertools.combinations(target_items, r):
                powerset.add(frozenset(tuple(combination)))

        # Sample proportion p
        num_to_sample = int(len(powerset) * proportion)
        sampled_set = set(random.sample(list(powerset), num_to_sample))  # Convert set to list

        # Add single sets
        for item in target_items:
            sampled_set.add(frozenset({item}))

        bundles = sampled_set
        
        clusters = int(len(bundles) / cluster_size)

        # Overall
        start = process_time()
        try:
            refined_revenue, unrefined_revenue = overall_heuristic_split(
                temp_whole_items, bundles, temp_wtp, clusters,
                refinement=True, prune=True,
                bmkc=True, comp_ind=False, lattice=True, theta =  0.0001
            )
            end = process_time()
            overall_time_list.append(end - start)
        except Exception as e:
            print("An error occurred:", e)
            import traceback
            traceback.print_exc()
            continue  # Skip the current iteration and proceed to the next one
    bundle_sizes.append(len(bundles))
    comparison_results['overall_time'].append(np.mean(overall_time_list))


format_string = "{:<25}{:<25}"

# Print aggregated results
print(format_string.format("Bundles","Time"))
for i in range(len(comparison_results['overall_time'])):
    print(format_string.format(f"{bundle_sizes[i]}",f"{comparison_results['overall_time'][i]:.2f}"))
