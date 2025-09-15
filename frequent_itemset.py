import json
from itertools import combinations
from collections import defaultdict


def load_names():
    # Load data
    foods_data = []
    with open('foodb/Food.json', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    food_item = json.loads(line)
                    foods_data.append(food_item)
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {line[:100]}... Error: {e}")

    names = [food['name'].lower() for food in foods_data]
    return names


def load_recipes():
    with open('ingredient_matching_best_of_both_results.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def frequent_ONE_itemsets(names, data, SUPPORT_THRESHOLD=100):
    # CORRECTED IMPLEMENTATION
    # Step 1: Count support for each individual item (Pass 1 of A-priori)
    item_counts = {}
    total_baskets = len(data)

    print(f"Total number of baskets (recipes): {total_baskets}")

    # Count each item's frequency
    for food in names:
        count = 0
        for recipe in data:
            if food in recipe['matched_ingredients']:
                count += 1
        item_counts[food] = count

    frequent_1_itemsets = {}
    for food, count in item_counts.items():
        if count >= SUPPORT_THRESHOLD:
            support_percentage = (count / total_baskets) * 100
            frequent_1_itemsets[food] = {
                'count': count,
                'support': count / total_baskets,
                'support_percentage': f"{support_percentage:.2f}%"
            }

    print(f"\nFound {len(frequent_1_itemsets)} frequent 1-itemsets out of {len(names)} total items")
    print(f"Support threshold: {SUPPORT_THRESHOLD} baskets")

    # Display top 10 most frequent items
    sorted_frequent = sorted(frequent_1_itemsets.items(),
                             key=lambda x: x[1]['count'],
                             reverse=True)

    print(f"\nTop 10 most frequent items:")
    for i, (food, stats) in enumerate(sorted_frequent[:10], 1):
        print(f"{i:2d}. {food:20s} - Count: {stats['count']:4d}, Support: {stats['support_percentage']}")

    # Save results
    with open('frequent_itemsets_whats-cooking/frequent_1_itemsets.json', 'w', encoding='utf-8') as f:
        json.dump(frequent_1_itemsets, f, ensure_ascii=False, indent=4)

    print(f"\nSaved frequent 1-itemsets to 'frequent_1_itemsets_corrected.json'")

    # Additional statistics
    if frequent_1_itemsets:
        max_support = max(item['count'] for item in frequent_1_itemsets.values())
        min_support = min(item['count'] for item in frequent_1_itemsets.values())
        avg_support = sum(item['count'] for item in frequent_1_itemsets.values()) / len(frequent_1_itemsets)

        print(f"\nFrequent itemsets statistics:")
        print(f"  Maximum support: {max_support}")
        print(f"  Minimum support: {min_support}")
        print(f"  Average support: {avg_support:.1f}")

    return list(frequent_1_itemsets.keys())  # Return L1 for next steps


def load_frequent_ONE():
    # Load your frequent 1-itemsets
    with open('frequent_itemsets_whats-cooking/frequent_1_itemsets.json', 'r', encoding='utf-8') as f:
        frequent_1_itemsets = json.load(f)
    return frequent_1_itemsets


def frequent_TWO_itemsets(data, SUPPORT_THRESHOLD=100):
    frequent_1_itemsets = load_frequent_ONE()

    # Extract just the list of frequent items (L1)
    L1 = list(frequent_1_itemsets.keys())
    print(f"Number of frequent 1-itemsets: {len(L1)}")

    # PASS 2: Generate C2 (candidate 2-itemsets) from L1
    print("Generating candidate 2-itemsets...")
    C2 = list(combinations(L1, 2))  # All pairs of frequent 1-itemsets
    print(f"Number of candidate 2-itemsets: {len(C2)}")

    # Count support for each candidate pair
    print("Counting support for candidate pairs...")
    pair_counts = defaultdict(int)
    total_baskets = len(data)

    # For each recipe (basket)
    for recipe_idx, recipe in enumerate(data):
        if recipe_idx % 5000 == 0:
            print(f"Processed {recipe_idx}/{total_baskets} recipes...")

        recipe_ingredients = set(recipe['matched_ingredients'])

        # Check each candidate pair
        for item1, item2 in C2:
            if item1 in recipe_ingredients and item2 in recipe_ingredients:
                # Store pair in sorted order for consistency
                pair = tuple(sorted([item1, item2]))
                pair_counts[pair] += 1

    print(f"Finished counting. Found {len(pair_counts)} pairs with non-zero support.")

    # Apply support threshold to get L2 (frequent 2-itemsets)
    frequent_2_itemsets = {}
    L2_list = []  # List format for next iteration

    for pair, count in pair_counts.items():
        if count >= SUPPORT_THRESHOLD:
            support_percentage = (count / total_baskets) * 100
            frequent_2_itemsets[f"{pair[0]} + {pair[1]}"] = {
                'items': list(pair),
                'count': count,
                'support': count / total_baskets,
                'support_percentage': f"{support_percentage:.2f}%"
            }
            L2_list.append(list(pair))  # Store as list for candidate generation

    print(f"\nFound {len(frequent_2_itemsets)} frequent 2-itemsets")
    print(f"Support threshold: {SUPPORT_THRESHOLD} baskets")

    # Display top 20 most frequent pairs
    sorted_frequent_pairs = sorted(frequent_2_itemsets.items(),
                                   key=lambda x: x[1]['count'],
                                   reverse=True)

    print(f"\nTop 10 most frequent item pairs:")
    for i, (pair_name, stats) in enumerate(sorted_frequent_pairs[:10], 1):
        items = stats['items']
        print(
            f"{i:2d}. {items[0]:15s} + {items[1]:15s} - Count: {stats['count']:4d}, Support: {stats['support_percentage']}")

    # Save results
    with open('frequent_itemsets_whats-cooking/frequent_2_itemsets.json', 'w', encoding='utf-8') as f:
        json.dump(frequent_2_itemsets, f, ensure_ascii=False, indent=4)

    print(f"\nSaved frequent 2-itemsets to 'frequent_2_itemsets.json'")

    return L2_list  # Return L2 for next iteration


def generate_candidates_k(frequent_k_minus_1, k):
    """
    Generate candidate k-itemsets from frequent (k-1)-itemsets
    Using the F(k-1) x F(k-1) method with pruning
    """
    candidates = []
    frequent_k_minus_1_sorted = [sorted(itemset) for itemset in frequent_k_minus_1]

    print(f"  Generating candidate {k}-itemsets from {len(frequent_k_minus_1)} frequent {k - 1}-itemsets...")

    for i in range(len(frequent_k_minus_1_sorted)):
        for j in range(i + 1, len(frequent_k_minus_1_sorted)):
            itemset1 = frequent_k_minus_1_sorted[i]
            itemset2 = frequent_k_minus_1_sorted[j]

            # Join condition: first k-2 items must be identical
            if itemset1[:-1] == itemset2[:-1]:
                candidate = sorted(itemset1 + [itemset2[-1]])

                # Prune: all (k-1)-subsets must be frequent
                if is_valid_candidate(candidate, frequent_k_minus_1_sorted, k):
                    candidates.append(candidate)

    return candidates


def is_valid_candidate(candidate, frequent_k_minus_1, k):
    """
    Check if all (k-1)-subsets of candidate are frequent
    This implements the pruning step of A-priori
    """
    for i in range(len(candidate)):
        subset = candidate[:i] + candidate[i + 1:]  # Remove item at index i
        if subset not in frequent_k_minus_1:
            return False
    return True


def count_support_k(candidates, data, k):
    """Count support for candidate k-itemsets"""
    candidate_counts = defaultdict(int)
    total_baskets = len(data)

    print(f"  Counting support for {len(candidates)} candidate {k}-itemsets...")

    for recipe_idx, recipe in enumerate(data):
        if recipe_idx % 5000 == 0 and recipe_idx > 0:
            print(f"    Processed {recipe_idx}/{total_baskets} recipes...")

        recipe_ingredients = set(recipe['matched_ingredients'])

        for candidate in candidates:
            if all(item in recipe_ingredients for item in candidate):
                candidate_tuple = tuple(sorted(candidate))
                candidate_counts[candidate_tuple] += 1

    return candidate_counts


def frequent_K_itemsets(data, frequent_k_minus_1, k, SUPPORT_THRESHOLD=100):
    """
    Generic function to find frequent k-itemsets
    """
    print(f"\n{'=' * 50}")
    print(f"PASS {k}: Finding frequent {k}-itemsets")
    print(f"{'=' * 50}")

    # Generate candidates
    candidates = generate_candidates_k(frequent_k_minus_1, k)
    print(f"  Generated {len(candidates)} candidate {k}-itemsets")

    if not candidates:
        print(f"  No candidates generated. Algorithm terminates.")
        return []

    # Count support
    candidate_counts = count_support_k(candidates, data, k)

    # Filter by support threshold
    frequent_k_itemsets = {}
    Lk_list = []  # List format for next iteration
    total_baskets = len(data)

    for itemset, count in candidate_counts.items():
        if count >= SUPPORT_THRESHOLD:
            support_percentage = (count / total_baskets) * 100
            key = " + ".join(itemset)
            frequent_k_itemsets[key] = {
                'items': list(itemset),
                'count': count,
                'support': count / total_baskets,
                'support_percentage': f"{support_percentage:.2f}%"
            }
            Lk_list.append(list(itemset))

    print(f"\nFound {len(frequent_k_itemsets)} frequent {k}-itemsets")

    if frequent_k_itemsets:
        # Display top results
        sorted_frequent = sorted(frequent_k_itemsets.items(),
                                 key=lambda x: x[1]['count'],
                                 reverse=True)

        display_count = min(10, len(sorted_frequent))
        print(f"\nTop {display_count} most frequent {k}-itemsets:")
        for i, (itemset_name, stats) in enumerate(sorted_frequent[:display_count], 1):
            items_str = " + ".join(stats['items'])
            print(f"{i:2d}. {items_str:50s} - Count: {stats['count']:4d}, Support: {stats['support_percentage']}")

        # Save results
        filename = f'frequent_itemsets_whats-cooking/frequent_{k}_itemsets.json'
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(frequent_k_itemsets, f, ensure_ascii=False, indent=4)
        print(f"\nSaved frequent {k}-itemsets to '{filename}'")

        # Statistics
        max_support = max(item['count'] for item in frequent_k_itemsets.values())
        min_support = min(item['count'] for item in frequent_k_itemsets.values())
        avg_support = sum(item['count'] for item in frequent_k_itemsets.values()) / len(frequent_k_itemsets)

        print(f"\nFrequent {k}-itemsets statistics:")
        print(f"  Maximum support: {max_support}")
        print(f"  Minimum support: {min_support}")
        print(f"  Average support: {avg_support:.1f}")

    return Lk_list


def complete_apriori_algorithm(names, recipes, SUPPORT_THRESHOLD=100):
    """
    Complete A-priori algorithm that finds all frequent itemsets
    """
    print("=" * 60)
    print("COMPLETE A-PRIORI ALGORITHM")
    print("=" * 60)
    print(f"Dataset: {len(recipes):,} recipes")
    print(f"Support threshold: {SUPPORT_THRESHOLD}")
    print()

    # Step 1: Find frequent 1-itemsets
    print("PASS 1: Finding frequent 1-itemsets")
    print("-" * 40)
    L1 = frequent_ONE_itemsets(names, recipes, SUPPORT_THRESHOLD)
    current_frequent = [[item] for item in L1]  # Convert to list of lists

    # Step 2: Find frequent 2-itemsets
    print("\n" + "=" * 50)
    print("PASS 2: Finding frequent 2-itemsets")
    print("=" * 50)
    L2 = frequent_TWO_itemsets(recipes, SUPPORT_THRESHOLD)
    current_frequent = L2

    # Step 3+: Find frequent k-itemsets (k >= 3)
    k = 3
    all_frequent_counts = {1: len(L1), 2: len(L2)}

    while current_frequent:
        Lk = frequent_K_itemsets(recipes, current_frequent, k, SUPPORT_THRESHOLD)

        if not Lk:  # No more frequent itemsets found
            print(f"\nAlgorithm terminates at k={k - 1}")
            break

        all_frequent_counts[k] = len(Lk)
        current_frequent = Lk
        k += 1

        # Safety check to avoid infinite loops with very low thresholds
        if k > 10:
            print(f"\nStopping at k={k - 1} to avoid excessive computation")
            break

    # Final summary
    print("\n" + "=" * 60)
    print("ALGORITHM SUMMARY")
    print("=" * 60)
    total_frequent = sum(all_frequent_counts.values())
    print(f"Total frequent itemsets found: {total_frequent}")
    print("\nBreakdown by size:")
    for size, count in all_frequent_counts.items():
        print(f"  {size}-itemsets: {count:,}")

    print(f"\nAll results saved in 'frequent_itemsets_whats-cooking/' directory")


if __name__ == '__main__':
    names = load_names()
    recipes = load_recipes()

    # Run complete A-priori algorithm
    complete_apriori_algorithm(names, recipes, SUPPORT_THRESHOLD=100)
