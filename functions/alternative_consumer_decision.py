# alternative consumer decisions for the consumer decision exp

from functions.consumer_decision import *

#baseline
def min_price(bundles, bundle_weights, bundle_prices):
    sorted_bundles = sorted(bundles, key=lambda x: bundle_prices[x], reverse=False)
    
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


# greedy max price
def max_price(bundles, bundle_weights, bundle_prices):
    sorted_bundles = sorted(bundles, key=lambda x: bundle_prices[x], reverse=True)
    
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

# greedy max surplus approach
def max_surplus_packing(bundles, bundle_weights, bundle_prices):
    # Sort bundles by weight/size ratio
    
    sorted_bundles = sorted(bundles, key=lambda x: bundle_weights[x], reverse=True)
    
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

# max size 
def max_size(bundles, bundle_weights, bundle_prices):
    # Sort bundles by weight/size ratio
    
    sorted_bundles = sorted(bundles, key=lambda x: len(x), reverse=True)
    
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


# revenue function, all consumers
# given a set of bundles, and their pricing, return the revenue of that bundle, pricing pair
# given consumer decision, find revenue. of a group of consumers for a given pricing


def calculate_alternative_revenue(whole_items, wtp, bundles, bundle_prices, packing_function):
    revenue_dict = {bundle: 0 for bundle in bundles}  # initialize revenue_dict
    total_surplus = 0

    for k in range(len(wtp)):  
        wtp_k = wtp[k]
        # Calculate weights for each bundle
        purchased_bundles, purchased_bundles_weights = calculate_weights(whole_items, wtp_k, bundles, bundle_prices)
        # Select max weighted bundles
        selected_bundles, total_weight = packing_function(purchased_bundles, purchased_bundles_weights, bundle_prices)
        total_surplus += total_weight

        for bundle in selected_bundles:
            # Add corresponding price of the bundle to revenue_dict
            revenue_dict[bundle] += bundle_prices.get(bundle)  # Use get to handle missing keys
    
    # Calculate total revenue
    total_revenue = sum(revenue_dict.values())
    
    return total_revenue, revenue_dict, total_surplus




def calculate_alternative_revenue_sbpc_greedy(whole_items, wtp, bundles, bundle_prices):
    revenue_dict = {bundle: 0 for bundle in bundles}  # initialize revenue_dict
    total_surplus = 0

    # Iterate through each consumer's WTP
    for k in range(len(wtp)):  
        wtp_k = wtp[k]  # Consumer's WTP for all items
        max_surplus = float('-inf')  # Initialize max surplus for this consumer
        best_configuration = None  # Track the best configuration for the consumer

        purchased_bundles, purchased_bundles_weights = calculate_weights(whole_items, wtp_k, bundles, bundle_prices)
        sorted_bundles = sorted(purchased_bundles, key=lambda x: purchased_bundles_weights[x] / math.sqrt(len(x)), reverse=True)
        if not sorted_bundles:
            continue  # Skip this consumer if no bundles are available
        chosen_bundle = sorted_bundles[0]
        configuration = [chosen_bundle] + [frozenset({item}) for item in whole_items if item not in chosen_bundle]
        surplus = 0

        for b in configuration:  # Iterate through each bundle in the configuration (bundle + components)

            surplus += max(0, sum(wtp_k[whole_items.index(item)] for item in b) - bundle_prices.get(b))  # WTP for bundle b

        total_surplus += surplus

    total_revenue  = None
    # total_revenue, revenue_dict isnt filled up here
    return total_revenue, revenue_dict, total_surplus



        