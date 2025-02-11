# Comparing Lattice vs Simple Graph

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
parser.add_argument("-r", "--repetitions", type=int, help="Number of repetitions", default = 30)
parser.add_argument("-e", "--seed", type=int, help="Seed", default = 0)
parser.add_argument("-co", "--components", type=int, help="Components", default = 7)
parser.add_argument("-n", "--bundles", type=int, help="Number of Bundles", default = 100)

args = parser.parse_args()

seed = args.seed
random.seed(seed)
repetitions = args.repetitions
components = args.components
num_bundles = args.bundles


data_PID = np.loadtxt("../data/Electronics_PID.csv", delimiter=",", dtype=str)
whole_items = [i for i in data_PID]
np_wtp = np.loadtxt("../data/Electronics_wtp.csv", delimiter=",")

print("Experiment on Lattice Compared With Simple Graph")


clusters_list = range(8, 17, 2)

for clusters in clusters_list:

    # Random 100 consumers 
    random_indices = np.random.choice(np_wtp.shape[0], size=100, replace=False)
    np_wtp = np_wtp[random_indices]

    # Define different configurations for the ablation study
    configurations = {
        'Lattice': {'refinement': True, 'prune': True, 'bmkc': True, 'comp_ind': False, 'lattice': True},
        'Simple_Graph': {'refinement': True, 'prune': True, 'bmkc': True, 'comp_ind': False, 'lattice': False}
    }

    results = {key: {'refined': [], 'unrefined': [], 'time': []} for key in configurations}


    # Initialize the results dictionary

    for _ in range(repetitions):
        target_items = random.sample(whole_items, components)
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
        num_to_sample = num_bundles - components if num_bundles - components > 0 else 0
        sampled_set = set(random.sample(list(powerset), num_to_sample))  # Convert set to list

        # Add single sets
        for item in target_items:
            sampled_set.add(frozenset({item}))

        bundles = sampled_set

        for config_name, config in configurations.items():

            start = process_time()
            try:
                refined_revenue, unrefined_revenue = overall_heuristic_split(
                    temp_whole_items, bundles, temp_wtp, clusters,
                    refinement=config['refinement'], prune=config['prune'],
                    bmkc=config['bmkc'], comp_ind=config['comp_ind'], lattice=config['lattice'],
                    theta =  0.0001
                )
                end = process_time()
            except Exception as e:
                print("An error occurred:", e)
                import traceback
                traceback.print_exc()
                continue  # Skip the current iteration and proceed to the next one

            results[config_name]['refined'].append(refined_revenue/total_coverage)
            results[config_name]['unrefined'].append(unrefined_revenue/total_coverage)
            results[config_name]['time'].append(end - start)


    print(f"Number of clusters (k) = {clusters}")
    for config_name, config_results in results.items():
        print(f"Configuration {config_name}:")
        print(f"Revenue: {np.mean(config_results['refined']):.5f} ± {np.std(config_results['refined']):.5f}")
        print(f"Time: {np.mean(config_results['time']):.5f} ± {np.std(config_results['time']):.5f}")
        print()
