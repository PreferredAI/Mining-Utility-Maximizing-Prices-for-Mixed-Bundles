#compare alpha beta

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
import os
# Visualize or save the results
import pandas as pd
# import ace_tools as tools  # To display results for analysis


parser = argparse.ArgumentParser()
parser.add_argument("-r", "--repetitions", type=int, help="Number of repetitions", default = 15)
parser.add_argument("-e", "--seed", type=int, help="Seed", default = 0)
parser.add_argument("-s", "--size", type=int, help="Size", default = 7)
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



# Generate reasonable alpha and beta values
alphas = np.arange(0, 2.25, 1)  # Alpha values: 0, 0.25, 0.5, ..., 2
betas = np.arange(0, 2.25, 1)   # Beta values: 0, 0.25, 0.5, ..., 2

# Initialize results dictionary for storing performance metrics
alpha_beta_results = {
    "alpha": [],
    "beta": [],
    "repetition": [],
    "revenue": [],
    "time": []
}



datasets = ["Electronics"]

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


    for repetition in range(repetitions):  # Loop over repetitions
        print(f"Repetition {repetition + 1}/{repetitions}")

        # Random 100 consumers 
        random_indices = np.random.choice(np_wtp.shape[0], size=100, replace=False)
        np_wtp = np_wtp[random_indices]

        # Prepare data for this repetition
        target_items = random.sample(whole_items, size)
        indices = [whole_items.index(i) for i in target_items]

        n1 = np.arange(len(np_wtp))
        temp_wtp = np.round(np_wtp[n1[:, None], indices], 2)
        temp_whole_items = target_items.copy()

        # Generate powerset
        powerset = set()
        for r in range(2, len(target_items) + 1):  # Start from 2 to exclude single-item sets
            for combination in itertools.combinations(target_items, r):
                powerset.add(frozenset(tuple(combination)))

        # Sample proportion
        num_to_sample = int(len(powerset) * proportion)
        sampled_set = set(random.sample(list(powerset), num_to_sample))  # Convert set to list

        # Add single sets
        for item in target_items:
            sampled_set.add(frozenset({item}))

        bundles = sampled_set

        clusters = int(len(bundles) / cluster_size)
        if clusters == 0:
            clusters = 1

        for alpha in alphas:
            for beta in betas:
                print(f"  Testing for alpha={alpha:.2f}, beta={beta:.2f}")

                # Run the heuristic split with varying alpha and beta
                start = process_time()
                try:
                    refined_revenue, unrefined_revenue = overall_heuristic_split_irefine(
                        temp_whole_items, bundles, temp_wtp, clusters,
                        refinement=True, prune=True,
                        bmkc=True, comp_ind=False, lattice=True,
                        alpha=alpha, beta=beta
                    )
                    end = process_time()
                    normalized_revenue = refined_revenue / np.sum(temp_wtp)  # Normalize revenue
                    elapsed_time = end - start

                    # Append results for this alpha-beta combination and repetition
                    alpha_beta_results["alpha"].append(alpha)
                    alpha_beta_results["beta"].append(beta)
                    alpha_beta_results["repetition"].append(repetition + 1)
                    alpha_beta_results["revenue"].append(normalized_revenue)
                    alpha_beta_results["time"].append(elapsed_time)

                except Exception as e:
                    print(f"    An error occurred at alpha={alpha:.2f}, beta={beta:.2f}: {e}")
                    # Append failed runs with 0 values
                    alpha_beta_results["alpha"].append(alpha)
                    alpha_beta_results["beta"].append(beta)
                    alpha_beta_results["repetition"].append(repetition + 1)
                    alpha_beta_results["revenue"].append(0)
                    alpha_beta_results["time"].append(0)

# Print results in a clean, grouped tabular format
print("\nAlpha-Beta Experiment Results:")
print("{:<10} {:<10} {:<12} {:<15} {:<15}".format("Alpha", "Beta", "Repetition", "Revenue", "Time"))

# Sort results for clean printing
sorted_results = sorted(zip(
    alpha_beta_results["alpha"],
    alpha_beta_results["beta"],
    alpha_beta_results["repetition"],
    alpha_beta_results["revenue"],
    alpha_beta_results["time"]
))

current_alpha = None
current_beta = None
for alpha, beta, repetition, revenue, time in sorted_results:
    if alpha != current_alpha or beta != current_beta:
        if current_alpha is not None:
            print("-" * 60)
        current_alpha, current_beta = alpha, beta
    print("{:<10.2f} {:<10.2f} {:<12} {:<15.5f} {:<15.5f}".format(alpha, beta, repetition, revenue, time))

