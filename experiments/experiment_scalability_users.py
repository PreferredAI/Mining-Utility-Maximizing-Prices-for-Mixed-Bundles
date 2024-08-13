
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
parser.add_argument("-r", "--repetitions", type=int, help="Number of repetitions", default = 15)
parser.add_argument("-e", "--seed", type=int, help="Seed", default = 0)
parser.add_argument("-cl", "--cluster", type=int, help="Clusters", default = 8)
parser.add_argument("-m", "--maxscale", type=int, help="Max Scale", default = 10)
parser.add_argument("-co", "--components", type=int, help="Components", default = 8)
parser.add_argument("-n", "--bundles", type=int, help="Number of Bundles", default = 50)
parser.add_argument("-st", "--start", type=int, help="Starting number of users", default = 50)
parser.add_argument("-i", "--increment", type=int, help="Increment number of users", default = 50)

args = parser.parse_args()


seed = args.seed
random.seed(seed)
repetitions = args.repetitions
clusters = args.cluster
max_scale = args.maxscale
components = args.components
num_bundles = args.bundles
start_users = args.start
increment = args.increment
max_users = start_users + increment *  (max_scale - 1)

data_PID = np.loadtxt("../data/Electronics_PID.csv", delimiter=",", dtype=str)
whole_items = [i for i in data_PID]
np_wtp = np.loadtxt("../data/Electronics_wtp.csv", delimiter=",")


user_counts = range(start_users, max_users + 1, increment)

print("Experiment Scalability - Users")

# Initialize the results dictionary
comparison_results = { 'overall_time': []}

scale_start_time = process_time()
for scale in range(max_scale):
    scale_end_time = process_time()

    overall_time_list = []
    # print(start_users + increment * scale)
    np_wtp_scaled = np_wtp[:start_users + increment * scale, :]
    

    for _ in range(repetitions):

        target_items = random.sample(whole_items, components)
        indices = [whole_items.index(i) for i in target_items]

        n1 = np.arange(len(np_wtp_scaled))
        temp_wtp = np.round(np_wtp_scaled[n1[:,None],indices], 2)
        temp_whole_items = target_items.copy()

        total_coverage = np.sum(temp_wtp)


        # Generate powerset
        powerset = set()
        for r in range(2, len(target_items) + 1):  # Start from 2 to exclude single-item sets
            for combination in itertools.combinations(target_items, r):
                powerset.add(frozenset(tuple(combination)))

        # Sample proportion p
        num_to_sample = num_bundles - components if num_bundles - components > 0 else 0
        sampled_set = set(random.sample(list(powerset), num_to_sample))  # Convert set to list

        # Add single sets
        for item in target_items:
            sampled_set.add(frozenset({item}))

        bundles = sampled_set

        
        # Overall
        start = process_time()
        try:
            refined_revenue, unrefined_revenue = overall_heuristic_split(
                temp_whole_items, bundles, temp_wtp, clusters,
                refinement=True, prune=True,
                bmkc=True, comp_ind=False, lattice=True
            )
            end = process_time()
            overall_time_list.append(end - start)
        except Exception as e:
            print("An error occurred:", e)
            import traceback
            traceback.print_exc()
            continue  # Skip the current iteration and proceed to the next one

    comparison_results['overall_time'].append(np.mean(overall_time_list))
        
        
format_string = "{:<25}{:<25}"

# Print aggregated results
print(format_string.format("Users","Time"))
for i in range(len(comparison_results['overall_time'])):
    print(format_string.format(f"{user_counts[i]}",f"{comparison_results['overall_time'][i]:.2f}"))
