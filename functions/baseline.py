from functions.pricing_functions import *
import random
# from math import prod

# BASELINES

def random_pricing(whole_items, bundle, wtp):
    wtp_bundle = [0 for k in range(len(wtp))]
    for i in bundle:
        item_index = whole_items.index(i)
        for k in range(len(wtp)):
            wtp_bundle[k] += wtp[k][item_index]
    return random.choice(wtp_bundle), None

def no_competition_pricing(whole_items, bundle, wtp):
    # assume that the item sets are equal
    wtp_bundle = [0 for k in range(len(wtp))]
    for i in bundle:
        item_index = whole_items.index(i)
        for k in range(len(wtp)):
            wtp_bundle[k] += wtp[k][item_index]

    max_so_far = 0
    max_price = 0
    sorted_wtp = sorted(wtp_bundle)
    unique_prices = sorted(set(sorted_wtp))
    for i in range(0, len(unique_prices), 5):  # Iterate through every 5 unique prices
        revenue = unique_prices[i] * (len(wtp_bundle) - sorted_wtp.index(unique_prices[i]))
        if revenue >= max_so_far:
            max_so_far = revenue
            max_price = unique_prices[i]
    return max_price, max_so_far

def rmp_baseline(whole_items, wtp, bundles):
    bundle_prices = {bundle: 1e6 for bundle in bundles}
    for bundle in bundles:
        bundle_price, _ = no_competition_pricing(whole_items, bundle, wtp)
        bundle_prices[bundle] = bundle_price
    
    total_revenue, revenue_dict = calculate_revenue(whole_items, wtp, bundles, bundle_prices)
    return total_revenue


def rmp_baseline_return_price(whole_items, wtp, bundles):
    bundle_prices = {bundle: 1e6 for bundle in bundles}
    for bundle in bundles:
        bundle_price, _ = no_competition_pricing(whole_items, bundle, wtp)
        bundle_prices[bundle] = bundle_price
    
    total_revenue, revenue_dict = calculate_revenue(whole_items, wtp, bundles, bundle_prices)
    return total_revenue, bundle_prices

import time

def hill_climbing_baseline(whole_items, wtp, bundles, increment=1.0, max_time=10.0):
    """
    Hill-climbing pricing algorithm for optimizing bundle prices with a time constraint.
    
    Args:
        whole_items: A set of all available items.
        wtp: Willingness-to-pay dictionary mapping customers to their valuations.
        bundles: A list of bundles (each bundle is a tuple of items).
        increment: The amount by which prices are increased or decreased in each step.
        max_time: Maximum time (in seconds) allowed for optimization.

    Returns:
        total_revenue: The optimized total revenue after hill climbing.
        bundle_prices: A dictionary mapping each bundle to its optimized price.
    """
    # Initialize bundle prices with a very high starting value
    bundle_prices = {bundle: 1e6 for bundle in bundles}
    
    # Compute initial prices
    for bundle in bundles:
        bundle_prices[bundle], _ = random_pricing(whole_items, bundle, wtp)

    # Compute initial revenue
    total_revenue, _ = calculate_revenue(whole_items, wtp, bundles, bundle_prices)
    
    # Start time tracking
    start_time = time.time()

    improvement = True
    while improvement:
        # Check if time limit is exceeded
        if time.time() - start_time > max_time:
            # print("Time limit reached, terminating early.")
            break
        
        improvement = False
        best_new_revenue = total_revenue
        best_new_prices = bundle_prices.copy()
        
        # Iterate through all bundles and try increasing or decreasing prices
        for bundle in bundles:
            for change in [-increment, increment]:
                new_prices = bundle_prices.copy()
                new_prices[bundle] += change  # Adjust price
                
                # Ensure price remains non-negative
                if new_prices[bundle] < 0:
                    continue
                
                # Compute new revenue
                new_revenue, _ = calculate_revenue(whole_items, wtp, bundles, new_prices)
                
                # If revenue improves, update best known revenue and prices
                if new_revenue > best_new_revenue:
                    best_new_revenue = new_revenue
                    best_new_prices = new_prices.copy()
                    improvement = True
        
        # Update prices and revenue if improvement was found
        if improvement:
            total_revenue = best_new_revenue
            bundle_prices = best_new_prices
    
    return total_revenue, bundle_prices


from scipy.stats import norm  # Import normal distribution

def closed_form_extension_baseline_normal(whole_items, wtp, bundles):
    """
    Computes closed-form bundle pricing using a normal distribution model for WTP.

    Args:
        whole_items: A set of all available items.
        wtp: Willingness-to-pay dictionary mapping customers to their valuations.
        bundles: A list of bundles (each bundle is a tuple of items)).

    Returns:
        revenue: The total revenue using the computed bundle prices.
        bundle_prices: A dictionary mapping each bundle to its computed closed-form price.
    """
    bundle_prices = {}

    for bundle in bundles:
        if len(bundle) == 1:
            # Extract single item
            item = next(iter(bundle))  
            item_wtp = [wtp[k][whole_items.index(item)] for k in range(len(wtp))]

            # Fit normal distribution (mean and std)
            mu, sigma = np.mean(item_wtp), np.std(item_wtp, ddof=1)

            # Get the 66th percentile value from the normal distribution
            P = norm.ppf(0.66, loc=mu, scale=sigma)

            # Compute price using P = 2R/3 → R = 3P/2
            R = (3 * P) / 2
            bundle_prices[bundle] = (2 * R) / 3  # Back to P
        else:
            # Step 1: Extract 66th percentile WTP modeled as normal for each item
            P_values = []
            for i in bundle:
                item_wtp = [wtp[k][whole_items.index(i)] for k in range(len(wtp))]
                mu, sigma = np.mean(item_wtp), np.std(item_wtp, ddof=1)
                P_values.append(norm.ppf(0.66, loc=mu, scale=sigma))  

            # Step 2: Compute R values (since P = 2R/3 → R = 3P/2)
            R_values = [ (3 * P) / 2 for P in P_values ]

            # Step 3: Compute bundle price using the modified k^(1/k) formulation
            P_b = (1 / 3) * (sum(2 * R for R in R_values) - (len(bundle) ** (1 / len(bundle))) * (prod(R_values) ** (1 / len(bundle))))
            
            # Store bundle price
            bundle_prices[bundle] = P_b

    # Step 4: Calculate revenue
    revenue, _ = calculate_revenue(whole_items, wtp, bundles, bundle_prices)
    
    return revenue, bundle_prices


from functools import reduce
import operator

def prod(iterable):
    """Custom implementation of math.prod() for Python 3.7."""
    return reduce(operator.mul, iterable, 1)

