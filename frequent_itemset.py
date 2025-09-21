import json
import numpy as np
from itertools import combinations
from collections import defaultdict, Counter
import time


class FinalOptimizedApriori:
    def __init__(self, recipes_path, support_threshold=30, max_itemset_size=5, use_multiset=True):
        self.recipes_path = recipes_path
        self.support_threshold = support_threshold
        self.max_itemset_size = max_itemset_size
        self.use_multiset = use_multiset

        self.item_to_id = {}
        self.id_to_item = {}
        self.recipe_vectors = []
        self.frequent_items_set = set()

    def load_and_preprocess(self):
        print("Loading and preprocessing data...")
        start_time = time.time()

        with open(self.recipes_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)

        all_ingredients = set()
        for recipe in raw_data:
            ingredients = [str(item).strip().lower() for item in recipe.get('matched_ingredients', [])]
            all_ingredients.update(ingredients)

        sorted_ingredients = sorted(all_ingredients)
        self.item_to_id = {item: idx for idx, item in enumerate(sorted_ingredients)}
        self.id_to_item = {idx: item for item, idx in self.item_to_id.items()}

        if self.use_multiset:
            self.recipe_vectors = []
            for recipe in raw_data:
                ingredients = [str(item).strip().lower() for item in recipe.get('matched_ingredients', [])]
                ingredient_ids = [self.item_to_id[ing] for ing in ingredients if ing in self.item_to_id]
                if ingredient_ids:
                    self.recipe_vectors.append(Counter(ingredient_ids))
        else:
            self.recipe_vectors = []
            for recipe in raw_data:
                ingredients = [str(item).strip().lower() for item in recipe.get('matched_ingredients', [])]
                ingredient_ids = {self.item_to_id[ing] for ing in ingredients if ing in self.item_to_id}
                if ingredient_ids:
                    self.recipe_vectors.append(ingredient_ids)

        print(f"Loaded {len(self.recipe_vectors)} recipes with {len(all_ingredients)} unique ingredients")
        print(f"Preprocessing took {time.time() - start_time:.2f} seconds")

    def frequent_1_itemsets_fast(self):
        '''Fast frequent 1-itemsets using vectorized counting'''
        print(f"\nPass 1: Finding frequent 1-itemsets...")
        start_time = time.time()

        if self.use_multiset:
            item_counts = defaultdict(int)
            for recipe_counter in self.recipe_vectors:
                for item_id, count in recipe_counter.items():
                    item_counts[item_id] += count
        else:
            item_counts = defaultdict(int)
            for recipe_set in self.recipe_vectors:
                for item_id in recipe_set:
                    item_counts[item_id] += 1

        frequent_1 = {}
        frequent_item_ids = []

        for item_id, count in item_counts.items():
            if count >= self.support_threshold:
                item_name = self.id_to_item[item_id]
                support_pct = (count / len(self.recipe_vectors)) * 100
                frequent_1[item_name] = {
                    'id': item_id,
                    'count': count,
                    'support': count / len(self.recipe_vectors),
                    'support_percentage': f"{support_pct:.2f}%"
                }
                frequent_item_ids.append(item_id)

        self.frequent_items_set = set(frequent_item_ids)

        print(f"Found {len(frequent_1)} frequent 1-itemsets")
        print(f"Pass 1 took {time.time() - start_time:.2f} seconds")

        sorted_items = sorted(frequent_1.items(), key=lambda x: x[1]['count'], reverse=True)
        print(f"\nTop 10 most frequent items:")
        for i, (item, stats) in enumerate(sorted_items[:10], 1):
            print(f"{i:2d}. {item:25s} - Count: {stats['count']:4d} ({stats['support_percentage']})")

        return frequent_1, frequent_item_ids

    def frequent_2_itemsets_ultra_fast(self, frequent_1_ids):
        '''Optimized 2-itemsets using recipe-driven approach'''
        print(f"\nPass 2: Finding frequent 2-itemsets (optimized)...")
        start_time = time.time()

        frequent_set = set(frequent_1_ids)
        pair_counts = defaultdict(int)

        print(f"Counting pairs from {len(self.recipe_vectors)} recipes...")

        for recipe_idx, recipe in enumerate(self.recipe_vectors):
            if recipe_idx % 10000 == 0 and recipe_idx > 0:
                print(f"  Processed {recipe_idx:,} recipes...")

            if self.use_multiset:
                recipe_items = []
                for item_id, count in recipe.items():
                    if item_id in frequent_set:
                        recipe_items.extend([item_id] * count)

                for i in range(len(recipe_items)):
                    for j in range(i + 1, len(recipe_items)):
                        pair = tuple(sorted([recipe_items[i], recipe_items[j]]))
                        pair_counts[pair] += 1

                item_counter = Counter(recipe_items)
                for item_id, count in item_counter.items():
                    if count >= 2:
                        self_pair_count = count * (count - 1) // 2
                        pair = (item_id, item_id)
                        pair_counts[pair] += self_pair_count
            else:
                recipe_items = [item_id for item_id in recipe if item_id in frequent_set]

                for i in range(len(recipe_items)):
                    for j in range(i + 1, len(recipe_items)):
                        pair = tuple(sorted([recipe_items[i], recipe_items[j]]))
                        pair_counts[pair] += 1

        print(f"Found {len(pair_counts)} unique pairs")

        frequent_2 = {}
        frequent_2_ids = []

        for pair, count in pair_counts.items():
            if count >= self.support_threshold:
                support_pct = (count / len(self.recipe_vectors)) * 100

                item1_name = self.id_to_item[pair[0]]
                item2_name = self.id_to_item[pair[1]]

                if pair[0] == pair[1]:
                    key = f"{item1_name} (×2)"
                    items_list = [pair[0], pair[0]]
                else:
                    key = f"{item1_name} + {item2_name}"
                    items_list = list(pair)

                frequent_2[key] = {
                    'items': [item1_name, item2_name] if pair[0] != pair[1] else [item1_name, item1_name],
                    'count': count,
                    'support': count / len(self.recipe_vectors),
                    'support_percentage': f"{support_pct:.2f}%"
                }
                frequent_2_ids.append([item1_name, item2_name] if pair[0] != pair[1] else [item1_name, item1_name])

        print(f"Found {len(frequent_2)} frequent 2-itemsets")
        print(f"Pass 2 took {time.time() - start_time:.2f} seconds")

        if frequent_2:
            sorted_pairs = sorted(frequent_2.items(), key=lambda x: x[1]['count'], reverse=True)
            print(f"\nTop 10 most frequent pairs:")
            for i, (pair_name, stats) in enumerate(sorted_pairs[:10], 1):
                print(f"{i:2d}. {pair_name:40s} - Count: {stats['count']:4d} ({stats['support_percentage']})")

        return frequent_2, frequent_2_ids

    def frequent_k_itemsets_recipe_driven(self, frequent_k_minus_1_ids, k):
        """Recipe-driven k-itemsets to avoid candidate explosion"""
        print(f"\nPass {k}: Finding frequent {k}-itemsets (recipe-driven)...")
        start_time = time.time()

        frequent_tuples = set()
        for itemset in frequent_k_minus_1_ids:
            id_itemset = tuple(sorted([self.item_to_id[item_name] for item_name in itemset]))
            frequent_tuples.add(id_itemset)
        k_itemset_counts = defaultdict(int)

        print(f"Scanning recipes for {k}-itemsets...")

        for recipe_idx, recipe in enumerate(self.recipe_vectors):
            if recipe_idx % 10000 == 0 and recipe_idx > 0:
                print(f"  Processed {recipe_idx:,} recipes...")

            if self.use_multiset:
                recipe_items = []
                for item_id, count in recipe.items():
                    if item_id in self.frequent_items_set:
                        recipe_items.extend([item_id] * min(count, k))

                if len(recipe_items) >= k:
                    for combo in combinations(recipe_items, k):
                        sorted_combo = tuple(sorted(combo))

                        if self.all_subsets_frequent(sorted_combo, frequent_tuples, k):
                            k_itemset_counts[sorted_combo] += 1
            else:
                recipe_items = [item_id for item_id in recipe if item_id in self.frequent_items_set]

                if len(recipe_items) >= k:
                    for combo in combinations(sorted(recipe_items), k):
                        if self.all_subsets_frequent(combo, frequent_tuples, k):
                            k_itemset_counts[combo] += 1

        print(f"Found {len(k_itemset_counts)} potential {k}-itemsets")

        frequent_k = {}
        frequent_k_ids = []

        for itemset_tuple, count in k_itemset_counts.items():
            if count >= self.support_threshold:
                support_pct = (count / len(self.recipe_vectors)) * 100

                item_names = [self.id_to_item[item_id] for item_id in itemset_tuple]

                if self.use_multiset:
                    item_counter = Counter(item_names)
                    display_parts = []
                    for item, freq in sorted(item_counter.items()):
                        if freq == 1:
                            display_parts.append(item)
                        else:
                            display_parts.append(f"{item}(×{freq})")
                    key = " + ".join(display_parts)
                else:
                    key = " + ".join(sorted(set(item_names)))

                frequent_k[key] = {
                    'items': item_names,
                    'count': count,
                    'support': count / len(self.recipe_vectors),
                    'support_percentage': f"{support_pct:.2f}%"
                }
                frequent_k_ids.append(item_names)

        print(f"Found {len(frequent_k)} frequent {k}-itemsets")
        print(f"Pass {k} took {time.time() - start_time:.2f} seconds")

        if frequent_k:
            sorted_itemsets = sorted(frequent_k.items(), key=lambda x: x[1]['count'], reverse=True)
            display_count = min(5, len(sorted_itemsets))
            print(f"\nTop {display_count} most frequent {k}-itemsets:")
            for i, (itemset_name, stats) in enumerate(sorted_itemsets[:display_count], 1):
                print(f"{i:2d}. {itemset_name:60s} - Count: {stats['count']:4d}")

        return frequent_k, frequent_k_ids

    def all_subsets_frequent(self, itemset, frequent_set, k):
        """Check if all (k-1)-subsets of itemset are frequent"""
        itemset_list = list(itemset)

        for i in range(min(3, len(itemset_list))):
            subset = tuple(sorted(itemset_list[:i] + itemset_list[i + 1:]))
            if subset not in frequent_set:
                return False

        return True

    def run_optimized(self):
        '''Run the complete recipe-driven A-priori algorithm'''
        print("=" * 70)
        print(f"RECIPE-DRIVEN A-PRIORI ALGORITHM ({'MULTISET' if self.use_multiset else 'STANDARD'})")
        print("=" * 70)
        print(f"Support threshold: {self.support_threshold}")
        print(f"Max itemset size: {self.max_itemset_size}")
        print(f"Multiset support: {'Enabled' if self.use_multiset else 'Disabled'}")

        total_start = time.time()

        self.load_and_preprocess()

        frequent_1, frequent_1_ids = self.frequent_1_itemsets_fast()
        all_results = {'frequent_1_itemsets': frequent_1}
        all_frequent_counts = {1: len(frequent_1)}

        if self.max_itemset_size == 1:
            self._save_and_summarize(all_results, all_frequent_counts, total_start)
            return all_results

        frequent_2, frequent_2_ids = self.frequent_2_itemsets_ultra_fast(frequent_1_ids)
        all_results['frequent_2_itemsets'] = frequent_2
        all_frequent_counts[2] = len(frequent_2)

        if self.max_itemset_size == 2:
            self._save_and_summarize(all_results, all_frequent_counts, total_start)
            return all_results

        current_frequent_ids = frequent_2_ids

        for k in range(3, self.max_itemset_size + 1):
            if not current_frequent_ids:
                print(f"\nNo frequent {k - 1}-itemsets found. Terminating.")
                break

            frequent_k, frequent_k_ids = self.frequent_k_itemsets_recipe_driven(current_frequent_ids, k)

            if not frequent_k:
                print(f"\nAlgorithm terminates at k={k - 1} (no more frequent itemsets)")
                break

            all_results[f'frequent_{k}_itemsets'] = frequent_k
            all_frequent_counts[k] = len(frequent_k)
            current_frequent_ids = frequent_k_ids

        self._save_and_summarize(all_results, all_frequent_counts, total_start)
        return all_results

    def _save_and_summarize(self, all_results, all_frequent_counts, total_start):
        '''Save results and print summary'''
        import os

        total_time = time.time() - total_start

        output_dir = "frequent_db_itemsets"
        os.makedirs(output_dir, exist_ok=True)

        saved_files = []
        for k in sorted(all_frequent_counts.keys()):
            itemset_key = f'frequent_{k}_itemsets'
            if itemset_key in all_results:
                clean_data = {}
                for display_name, stats in all_results[itemset_key].items():
                    if k == 1:
                        items_as_strings = [str(display_name)]
                    else:
                        items_as_strings = [str(item) for item in stats['items']]

                    clean_data[display_name] = {
                        "items": items_as_strings,
                        "count": stats['count'],
                        "support": stats['support'],
                        "support_percentage": stats['support_percentage']
                    }

                suffix = "_multiset" if self.use_multiset else "_standard"
                filename = f'{output_dir}/frequent_{k}_itemsets{suffix}.json'

                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(clean_data, f, ensure_ascii=False, indent=4)

                saved_files.append(filename)
                print(f"Saved {len(clean_data)} frequent {k}-itemsets to '{filename}'")

        summary_filename = f'{output_dir}/summary{"_multiset" if self.use_multiset else "_standard"}.json'
        summary_data = {
            'metadata': {
                'algorithm': 'Recipe-Driven A-priori with Multiset Support',
                'support_threshold': self.support_threshold,
                'max_itemset_size': self.max_itemset_size,
                'use_multiset': self.use_multiset,
                'total_recipes': len(self.recipe_vectors),
                'total_runtime_seconds': total_time,
                'itemset_counts': all_frequent_counts
            },
            'files_generated': saved_files
        }

        with open(summary_filename, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, ensure_ascii=False, indent=4)

        print(f"\n{'=' * 70}")
        print("PERFORMANCE SUMMARY")
        print(f"{'=' * 70}")
        print(f"Total runtime: {total_time:.2f} seconds")
        print(f"Algorithm type: {'Multiset' if self.use_multiset else 'Standard'}")
        print(f"Total frequent itemsets: {sum(all_frequent_counts.values())}")
        print(f"\nBreakdown by size:")
        for size in sorted(all_frequent_counts.keys()):
            count = all_frequent_counts[size]
            print(f"  {size}-itemsets: {count:,}")
        print(f"\nResults saved in '{output_dir}/' directory:")
        for filename in saved_files:
            print(f"  - {filename}")
        print(f"  - {summary_filename}")


def run_recipe_driven_apriori(recipes_path, support_threshold=30, max_itemset_size=5, use_multiset=True):
    '''
    Run the recipe-driven A-priori algorithm

    recipes_path: Path to recipe JSON file
    support_threshold: Minimum support count
    max_itemset_size: Maximum itemset size (1-5)
    use_multiset: Enable multiset support for duplicate ingredients
    '''
    apriori = FinalOptimizedApriori(
        recipes_path=recipes_path,
        support_threshold=support_threshold,
        max_itemset_size=max_itemset_size,
        use_multiset=use_multiset
    )

    return apriori.run_optimized()


if __name__ == "__main__":
    print("Running recipe-driven A-priori with multiset support...")
    results = run_recipe_driven_apriori(
        recipes_path='recipes_database_only_results.json',
        support_threshold=30,
        max_itemset_size=5,
        use_multiset=True
    )

    print(
        f"\nCompleted! Found {sum(len(results.get(f'frequent_{k}_itemsets', {})) for k in range(1, 6))} total frequent itemsets.")
