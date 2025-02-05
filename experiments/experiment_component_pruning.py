#compare theta

import numpy as np
import argparse
import random
import warnings
import itertools
warnings.filterwarnings("error")
import os

import sys
sys.path.append("..")
from functions.pricing_functions import *
from functions.algorithm import *
from functions.alternative_consumer_decision import *
from functions.baseline import *
from functions.consumer_decision import *
from functions.overall import *

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




# Initialize results dictionary for storing performance metrics
theta_results = {
    "theta": [],
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


    thetas = [0.5, 0.9, None]

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


        for theta in thetas:  # Loop over theta values
            # print(f"  Testing for theta={theta if theta is not None else 'None'}")

            # Run the heuristic split with varying theta
            start = process_time()
            try:
                if theta is not None:
                    refined_revenue, unrefined_revenue = overall_heuristic_split_irefine(
                        temp_whole_items, bundles, temp_wtp, clusters,
                        refinement=True, prune=True,
                        bmkc=True, comp_ind=False, lattice=True,
                        theta=theta
                    )
                else:
                    refined_revenue, unrefined_revenue = overall_heuristic_split_irefine(
                        temp_whole_items, bundles, temp_wtp, clusters,
                        refinement=True, prune=False,
                        bmkc=True, comp_ind=False, lattice=True,
                        theta=0
                    )

                end = process_time()
                normalized_revenue = refined_revenue / np.sum(temp_wtp)  # Normalize revenue
                elapsed_time = end - start
                # Append results for this theta and repetition
                theta_results["theta"].append(theta)
                theta_results["repetition"].append(repetition + 1)
                theta_results["revenue"].append(normalized_revenue)
                theta_results["time"].append(elapsed_time)

            except Exception as e:
                print(f"An error occurred at theta={theta}: {e}")
                theta_results["theta"].append(theta)
                theta_results["repetition"].append(repetition + 1)
                theta_results["revenue"].append(0)  # Append 0 revenue for failed runs
                theta_results["time"].append(0)  # Append 0 time for failed runs
                continue  # Skip to the next theta


# Sort results for better readability
sorted_results = sorted(zip(theta_results["repetition"], theta_results["theta"], theta_results["revenue"], theta_results["time"]))

# Print header
print("\nTheta Experiment Results:")
print("{:<12} {:<10} {:<15} {:<15}".format("Repetition", "Theta", "Revenue", "Time"))

# Process and print results, grouping by repetition
current_repetition = None
repetition_data = {}  # To store data for calculating averages and std deviations

for repetition, theta, revenue, time in sorted_results:
    if repetition != current_repetition:
        # Print separator for new repetition group
        if current_repetition is not None:
            print("-" * 60)
        current_repetition = repetition

    # Print the data row
    print("{:<12} {:<10} {:<15.5f} {:<15.5f}".format(
        repetition,
        f"{theta:.5f}" if theta is not None else "None",
        revenue,
        time
    ))

    # Collect data for averages and std deviation
    if repetition not in repetition_data:
        repetition_data[repetition] = {"revenue": [], "time": []}
    repetition_data[repetition]["revenue"].append(revenue)
    repetition_data[repetition]["time"].append(time)

# Print summary for each repetition
print("\nSummary (Averages and Standard Deviations):")
print("{:<12} {:<15} {:<15}".format("Repetition", "Avg Revenue", "Std Revenue"))
print("{:<12} {:<15} {:<15}".format("", "Avg Time", "Std Time"))

for repetition, data in repetition_data.items():
    avg_revenue = np.mean(data["revenue"])
    std_revenue = np.std(data["revenue"], ddof=1)
    avg_time = np.mean(data["time"])
    std_time = np.std(data["time"], ddof=1)

    print("{:<12} {:<15.5f} {:<15.5f}".format(repetition, avg_revenue, std_revenue))
    print("{:<12} {:<15.5f} {:<15.5f}".format("", avg_time, std_time))
