#compare overall against baseline

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
parser.add_argument("-s", "--size", type=int, help="Size", default = 4)
parser.add_argument("-p", "--proportion", type=float, help="Proportion", default = 0.5)
parser.add_argument("-cls", "--clustersize", type=int, help="Cluster Size", default = 5)
parser.add_argument("-m", "--maxscale", type=int, help="Max Scale", default = 6)
args = parser.parse_args()

seed = args.seed
random.seed(seed)
repetitions = args.repetitions
size = args.size
proportion = args.proportion
cluster_size = args.clustersize
max_scale = args.maxscale


print("Comparison Against Baseline")

datasets = ["Electronics", "Sports and Outdoors", "UEL"]


for dataset in datasets:

    if dataset == "Electronics":
        data_PID = np.loadtxt("../data/Electronics_PID.csv", delimiter=",", dtype=str)
        np_wtp = np.loadtxt("../data/Electronics_wtp.csv", delimiter=",")
    elif dataset == "Sports and Outdoors":
        data_PID = np.loadtxt("../data/SportsOutdoors_PID.csv", delimiter=",", dtype=str)
        np_wtp = np.loadtxt("../data/SportsOutdoors_wtp.csv", delimiter=",")
    elif dataset == "UEL":
        data_PID = np.loadtxt("../data/UEL_PID.csv", delimiter=",", dtype=str)
        np_wtp = np.loadtxt("../data/UEL_wtp.csv", delimiter=",")



    whole_items = [i for i in data_PID]

    # Random 100 consumers 
    random_indices = np.random.choice(np_wtp.shape[0], size=100, replace=False)
    np_wtp = np_wtp[random_indices]


    # Initialize the results dictionary
    comparison_results = {'overall_revenue': [], 'rmp_revenue': [], 'baseline_revenue':[], 'hill_revenue': [], 'closed_form_revenue': []}
    bundle_sizes = []



    scale_start_time = process_time()
    for scale in range(max_scale):
        scale_end_time = process_time()
        overall_revenue_list = []
        overall_time_list = []
        rmp_revenue_list = []
        rmp_time_list = []
        baseline_revenue_list = []
        hill_revenue_list = []
        closed_form_revenue_list = []
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


            components_set = set()
            # Add single sets
            for item in target_items:
                components_set.add(frozenset({item}))
            

            clusters = int(len(bundles) / cluster_size)
            if clusters == 0:
                clusters = 1

            # Overall
            start = time.time()
            try:
                refined_revenue, unrefined_revenue = overall_heuristic_split_irefine(
                    temp_whole_items, bundles, temp_wtp, clusters,
                    refinement=True, prune=True,
                    bmkc=False, comp_ind=False, lattice=True, partition_method = 'balanced_kmeans'
                )
                end = time.time()
                overall_revenue_list.append(refined_revenue/total_coverage)
                overall_time_list.append(end - start)
            except Exception as e:
                print("An error occurred:", e)
                import traceback
                traceback.print_exc()
                continue  # Skip the current iteration and proceed to the next one
            
            algo_time = end - start
            # Ind-Pricing
            
            start = process_time()
            rmp_revenue = rmp_baseline(temp_whole_items, temp_wtp, bundles )
            end = process_time()
            rmp_revenue_list.append(rmp_revenue/total_coverage)
            rmp_time_list.append(end - start)
            
            # RMP Component prices (Baseline)
            component_revenue, component_prices = rmp_baseline_return_price(temp_whole_items, temp_wtp, components_set )
            baseline_revenue_list.append(component_revenue/total_coverage)
            
            hill_revenue, _ = hill_climbing_baseline(temp_whole_items, temp_wtp, bundles, max_time = algo_time)
            hill_revenue_list.append(hill_revenue / total_coverage)

            closed_form_revenue_normal_simple, _ = closed_form_extension_baseline_normal(temp_whole_items, temp_wtp, bundles)
            closed_form_revenue_list.append(closed_form_revenue_normal_simple / total_coverage)

        bundle_sizes.append(len(bundles))

        comparison_results['overall_revenue'].append(np.mean(overall_revenue_list))
        comparison_results['rmp_revenue'].append(np.mean(rmp_revenue_list))
        comparison_results['baseline_revenue'].append(np.mean(baseline_revenue_list))
        comparison_results['hill_revenue'].append(np.mean(hill_revenue_list))
        comparison_results['closed_form_revenue'].append(np.mean(closed_form_revenue_list))


        format_string = "{:<25}{:<25}{:<30}{:<30}{:<30}{:<30}{:<30}"

    # Print aggregated results
    print(f"{dataset} Dataset Results:")

    print(format_string.format("Items", "Bundles", "IPM", "Ind-Pricing", "Components", "HillClimbing", "ClosedFormExtended"))
    for s, (overall_revenue, rmp_revenue, baseline_revenue, hill_revenue, closed_form_simple_normal_revenue) in enumerate(
        zip(
            comparison_results['overall_revenue'],
            comparison_results['rmp_revenue'],
            comparison_results['baseline_revenue'],
            comparison_results['hill_revenue'],
            comparison_results['closed_form_revenue']
        )
    ):
        print(format_string.format(s+size, bundle_sizes[s], f"{overall_revenue*100:.2f}%", f"{rmp_revenue*100:.2f}%", f"{baseline_revenue*100:.2f}%",  f"{hill_revenue*100:.2f}%", f"{closed_form_simple_normal_revenue*100:.2f}%"))



