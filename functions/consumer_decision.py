import numpy as np
import math


# consumer decision function, single consumer, 
# given a set of bundles (set of items), and also their corresponding weights, 
# return the set that gives the maximum weighted set packing. 
def max_weighted_pack(bundles, bundle_weights):
    # Sort bundles by weight/size ratio
    
    sorted_bundles = sorted(bundles, key=lambda x: bundle_weights[x] / math.sqrt(len(x)), reverse=True)
    
    selected_bundles = set()
    selected_items = set()
    total_weight = 0
    
    # Greedily select bundles
    for bundle in sorted_bundles:
        # Check if any item in the current bundle is already in selected bundles
        if not any(item in selected_items for item in bundle):
            selected_bundles.add(bundle) # adds it in as a bundle
            selected_items.update(bundle) # union of two sets

            total_weight += bundle_weights[bundle]
    
    return selected_bundles, total_weight

# wrapper
def max_weighted_packing(bundles, bundle_weights, bundle_prices):
    return max_weighted_pack(bundles, bundle_weights)


# function, to calculate surplus, singlue consumer. 
# given a set of bundles (set of items), and also their corresponding prices,
# return their corresponding weights. 
def calculate_weights(whole_items, wtp_k, bundles, bundle_prices):
    weights = {}
    purchased_bundles = set()
    for bundle in bundles:
        weight = sum(wtp_k[whole_items.index(item)] for item in bundle) - bundle_prices.get(bundle)
        if weight >= 0:
            weights[bundle] = weight 
            purchased_bundles.add(bundle)

    return purchased_bundles, weights

# original
# revenue function, all consumers
# given a set of bundles, and their pricing, return the revenue of that bundle, pricing pair
# given consumer decision, find revenue. of a group of consumers for a given pricing


def calculate_revenue(whole_items, wtp, bundles, bundle_prices):
    revenue_dict = {bundle: 0 for bundle in bundles}  # initialize revenue_dict

    for k in range(len(wtp)):  
        wtp_k = wtp[k]
        # Calculate weights for each bundle
        purchased_bundles, purchased_bundles_weights = calculate_weights(whole_items, wtp_k, bundles, bundle_prices)
        # Select max weighted bundles
        selected_bundles, total_weight = max_weighted_pack(purchased_bundles, purchased_bundles_weights)
        
        for bundle in selected_bundles:
            # Add corresponding price of the bundle to revenue_dict
            revenue_dict[bundle] += bundle_prices.get(bundle)  # Use get to handle missing keys
    
    # Calculate total revenue
    total_revenue = sum(revenue_dict.values())
    
    return total_revenue, revenue_dict



def calculate_bundle_wtp(whole_items, wtp, bundle):
    wtp_bundle = np.sum(wtp[:, [whole_items.index(item) for item in bundle]], axis=1)  # Select relevant columns and sum
    
    # Find the revenue_maximising_price
    sorted_wtp = sorted(wtp_bundle)
    unique_prices = sorted(set(sorted_wtp))
    max_so_far = 0
    max_price = None
    for i in range(0, len(unique_prices), 5):  # Iterate through every 5 unique prices
        revenue = unique_prices[i] * (len(wtp_bundle) - sorted_wtp.index(unique_prices[i]))
        if revenue >= max_so_far:
            max_so_far = revenue
            max_price = unique_prices[i]
    
    # Calculate the standard deviation of wtp_bundle
    std_dev = np.std(wtp_bundle)

    return wtp_bundle, max_price, std_dev





