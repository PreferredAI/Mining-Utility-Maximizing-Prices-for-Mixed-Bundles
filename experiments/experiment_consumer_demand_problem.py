# with rmp pricing, how do alternative consumer deicsion methods ocmpare?
# updated packages and refactored code
# we take price as a given


import numpy as np
import argparse
import random
from time import time
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



parser = argparse.ArgumentParser()
parser.add_argument("-r", "--repetitions", type=int, help="Number of repetitions", default = 50)
parser.add_argument("-e", "--seed", type=int, help="Seed", default = 0)
parser.add_argument("-p", "--proportion", type=float, help="Proportion", default = 0.1)
parser.add_argument("-m", "--maxscale", type=int, help="Max Scale", default = 16)



args = parser.parse_args()

seed = args.seed
random.seed(seed)
repetitions = args.repetitions
proportion = args.proportion
max_scale = args.maxscale

data_PID = np.loadtxt("../data/Electronics_PID.csv", delimiter=",", dtype=str)
whole_items = [i for i in data_PID]
np_wtp = np.loadtxt("../data/Electronics_wtp.csv", delimiter=",")


print("Consumer Decision Experiment")

# Initialize the results dictionary
results = {method: {n: [] for n in range(5, 5 + max_scale)} for method in [
    'max_price', 'max_size', 'max_surplus_by_weight', 'components_baseline']}



for n in range(5, 5 + max_scale):
    for _ in range(repetitions):
        target_items = random.sample(whole_items, n)
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

        components = set()
        # Add single sets
        for item in target_items:
            components.add(frozenset({item}))
        


        start = time()
        # RMP bundle prices
        rmp_revenue, rmp_prices = rmp_baseline_return_price(temp_whole_items, temp_wtp, bundles )
        end = time()

        # RMP Component prices
        component_revenue, component_prices = rmp_baseline_return_price(temp_whole_items, temp_wtp, components )

        # Comparison methods
        methods = {
            'max_price': max_price,
            'max_size': max_size,
            'max_surplus_by_weight': max_weighted_packing,
            'components_baseline': min_price,
            'single_bundle_plus_components_greedy': None
        }

        for method_name, method_func in methods.items():
            start = time()
            if method_name == 'components_baseline':
                revenue, revenue_dict, surplus = calculate_alternative_revenue(temp_whole_items, temp_wtp, components, component_prices, method_func)
            elif method_name == 'single_bundle_plus_components_greedy':
                revenue, revenue_dict, surplus = calculate_alternative_revenue_sbpc_greedy(temp_whole_items, temp_wtp, bundles, rmp_prices)
            else:
                revenue, revenue_dict, surplus = calculate_alternative_revenue(temp_whole_items, temp_wtp, bundles, rmp_prices, method_func)
            end = time()
            results[method_name][n].append((surplus, surplus / total_coverage * 100, end - start))


# Calculate and print averages
for n in range(5, 5 + max_scale):
    print(f"Averages for number of bundles={int((2**n-1-n)*proportion)+n}:")
    for method_name in results:
        if results[method_name][n]:
            avg_percentage = np.mean([res[1] for res in results[method_name][n]])
            print(f"  {method_name.replace('_', ' ').title()}:")
            print(f"    Average Percentage: {avg_percentage:.3f}")


