import numpy as np
from numba import jit, prange
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from collections import Counter, defaultdict
import time
import json
import os
import psutil


@jit(nopython=True, parallel=True)
def fast_cosine_silhouette_sample(feature_matrix, labels, sample_size=1000):
    '''Fast cosine-based silhouette computation for large datasets'''
    n_samples = feature_matrix.shape[0]

    if n_samples > sample_size:
        indices = np.random.choice(n_samples, sample_size, replace=False)
        sample_features = feature_matrix[indices]
        sample_labels = labels[indices]
    else:
        sample_features = feature_matrix
        sample_labels = labels
        indices = np.arange(n_samples)

    n_samples = len(sample_features)
    silhouette_scores = np.zeros(n_samples)

    for i in prange(n_samples):
        # Current point and its cluster
        point = sample_features[i]
        cluster_i = sample_labels[i]

        # Calculate intra-cluster distance (a)
        same_cluster_distances = []
        for j in range(n_samples):
            if i != j and sample_labels[j] == cluster_i:
                # Cosine distance
                dot_product = 0.0
                norm_i = 0.0
                norm_j = 0.0

                for k in range(len(point)):
                    dot_product += point[k] * sample_features[j, k]
                    norm_i += point[k] * point[k]
                    norm_j += sample_features[j, k] * sample_features[j, k]

                if norm_i > 0 and norm_j > 0:
                    cosine_sim = dot_product / (np.sqrt(norm_i) * np.sqrt(norm_j))
                    cosine_dist = 1.0 - max(-1.0, min(1.0, cosine_sim))
                    same_cluster_distances.append(cosine_dist)

        a = np.mean(np.array(same_cluster_distances)) if same_cluster_distances else 0.0

        # Calculate nearest-cluster distance (b)
        unique_clusters = np.unique(sample_labels)
        min_other_cluster_dist = 1.0

        for other_cluster in unique_clusters:
            if other_cluster == cluster_i:
                continue

            other_cluster_distances = []
            for j in range(n_samples):
                if sample_labels[j] == other_cluster:
                    # Cosine distance
                    dot_product = 0.0
                    norm_i = 0.0
                    norm_j = 0.0

                    for k in range(len(point)):
                        dot_product += point[k] * sample_features[j, k]
                        norm_i += point[k] * point[k]
                        norm_j += sample_features[j, k] * sample_features[j, k]

                    if norm_i > 0 and norm_j > 0:
                        cosine_sim = dot_product / (np.sqrt(norm_i) * np.sqrt(norm_j))
                        cosine_dist = 1.0 - max(-1.0, min(1.0, cosine_sim))
                        other_cluster_distances.append(cosine_dist)

            if other_cluster_distances:
                avg_dist = np.mean(np.array(other_cluster_distances))
                min_other_cluster_dist = min(min_other_cluster_dist, avg_dist)

        b = min_other_cluster_dist

        if max(a, b) > 0:
            silhouette_scores[i] = (b - a) / max(a, b)
        else:
            silhouette_scores[i] = 0.0

    return np.mean(silhouette_scores)


class RecursiveBatchKMeansClustering:
    '''
    Recursive batch K-means clustering that adaptively discovers natural groupings
    in recipe data by iteratively splitting heterogeneous clusters
    '''

    def __init__(self, data_path, cuisine_data_path, output_dir="recursive_kmeans_results", target_clusters=None):
        self.data_path = data_path
        self.cuisine_data_path = cuisine_data_path
        self.output_dir = output_dir
        self.transactions = None
        self.cuisine_labels = None
        self.feature_matrix = None

        self.min_cluster_size = 100
        self.max_heterogeneity = 0.7
        self.min_silhouette = 0.1
        self.target_clusters = target_clusters

        self.cluster_hierarchy = {}
        self.final_clusters = {}
        self.performance_stats = {}

        self._create_output_directory()

    def _create_output_directory(self):
        """Create output directory structure"""
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        for subdir in ['results', 'diagnostics', 'iterations']:
            path = os.path.join(self.output_dir, subdir)
            if not os.path.exists(path):
                os.makedirs(path)

    def load_data_and_features(self):
        """Load data and create feature matrix using enhanced itemset selection"""
        print("Loading data and creating feature matrix...")
        start_time = time.time()

        with open(self.data_path, 'r', encoding='utf-8') as f:
            transaction_data = json.load(f)

        self.transactions = [recipe['matched_ingredients'] for recipe in transaction_data]

        with open(self.cuisine_data_path, 'r', encoding='utf-8') as f:
            cuisine_data = json.load(f)

        recipe_id_to_cuisine = {recipe['recipe_id']: recipe['cuisine'] for recipe in cuisine_data}
        self.cuisine_labels = []

        for recipe in transaction_data:
            if 'recipe_id' in recipe and recipe['recipe_id'] in recipe_id_to_cuisine:
                self.cuisine_labels.append(recipe_id_to_cuisine[recipe['recipe_id']])
            else:
                self.cuisine_labels.append('unknown')

        self.feature_matrix = self._create_optimized_feature_matrix()

        load_time = time.time() - start_time
        print(f"Data loaded and feature matrix created in {load_time:.2f}s")
        print(f"Feature matrix shape: {self.feature_matrix.shape}")
        print(f"Sparsity: {(1 - np.sum(self.feature_matrix) / self.feature_matrix.size) * 100:.2f}%")

    def _create_optimized_feature_matrix(self):
        """Create feature matrix using ultra-fast optimized approach from hierarchical version"""
        print("Creating optimized feature matrix...")
        start_time = time.time()

        all_itemsets = []

        for size in [2, 3, 5]:
            try:
                filepath = f"frequent_db_itemsets/frequent_{size}_itemsets_multiset.json"
                with open(filepath, 'r') as f:
                    frequent_itemsets = json.load(f)

                selected = self._select_discriminative_itemsets(frequent_itemsets, top_k=166)
                all_itemsets.extend(selected)
                print(f"  Size {size}: {len(selected)} itemsets selected")

            except FileNotFoundError:
                print(f"  Size {size}: File not found, skipping")
                continue

        unique_itemsets = []
        seen_itemsets = set()
        for itemset in all_itemsets:
            itemset_tuple = tuple(sorted(itemset))
            if itemset_tuple not in seen_itemsets:
                unique_itemsets.append(itemset)
                seen_itemsets.add(itemset_tuple)

        print(f"Total unique itemsets: {len(unique_itemsets)}")

        n_transactions = len(self.transactions)
        n_itemsets = len(unique_itemsets)

        transaction_sets = [frozenset(transaction) for transaction in self.transactions]
        itemset_sets = [frozenset(itemset) for itemset in unique_itemsets]

        ingredient_to_transactions = defaultdict(set)
        for i, transaction_set in enumerate(transaction_sets):
            for ingredient in transaction_set:
                ingredient_to_transactions[ingredient].add(i)

        feature_matrix = np.zeros((n_transactions, n_itemsets), dtype=np.float32)

        for j, itemset_set in enumerate(itemset_sets):
            if not itemset_set:
                continue

            candidate_transactions = None
            for ingredient in itemset_set:
                if ingredient in ingredient_to_transactions:
                    ingredient_transactions = ingredient_to_transactions[ingredient]
                    if candidate_transactions is None:
                        candidate_transactions = ingredient_transactions.copy()
                    else:
                        candidate_transactions &= ingredient_transactions
                else:
                    candidate_transactions = set()
                    break

            if candidate_transactions:
                for i in candidate_transactions:
                    feature_matrix[i, j] = 1.0

        creation_time = time.time() - start_time
        print(f"Feature matrix created in {creation_time:.2f}s")

        return feature_matrix

    def _select_discriminative_itemsets(self, frequent_itemsets, top_k=200):
        '''Simplified discriminative itemset selection'''
        itemset_scores = []

        for itemset_key, itemset_data in frequent_itemsets.items():
            itemset = set(itemset_data['items'])
            support = itemset_data['support']

            if support > 0.6 or support < 0.02:
                continue

            cuisine_counts = defaultdict(int)
            total_containing = 0

            for i, transaction in enumerate(self.transactions):
                if itemset.issubset(set(transaction)):
                    cuisine = self.cuisine_labels[i]
                    if cuisine != 'unknown':
                        cuisine_counts[cuisine] += 1
                        total_containing += 1

            if total_containing < 5:
                continue

            probabilities = [count / total_containing for count in cuisine_counts.values()]
            entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)

            unique_cuisines = len(set(self.cuisine_labels)) - (1 if 'unknown' in self.cuisine_labels else 0)
            max_entropy = np.log2(unique_cuisines) if unique_cuisines > 1 else 1
            discrimination_score = 1 - (entropy / max_entropy) if max_entropy > 0 else 0

            combined_score = discrimination_score * 0.7 + support * 0.3

            itemset_scores.append({
                'items': itemset_data['items'],
                'score': combined_score
            })

        itemset_scores.sort(key=lambda x: x['score'], reverse=True)
        return [item['items'] for item in itemset_scores[:top_k]]

    def calculate_cluster_metrics(self, cluster_indices, recipe_indices):
        """Calculate various metrics for cluster quality assessment with optimized performance"""
        if len(cluster_indices) < 2:
            return {'heterogeneity': 0, 'silhouette': 0, 'cuisine_purity': 0, 'size': len(recipe_indices)}

        cluster_cuisines = [self.cuisine_labels[i] for i in recipe_indices]
        cuisine_counts = Counter(cluster_cuisines)
        most_common_count = cuisine_counts.most_common(1)[0][1] if cuisine_counts else 0
        cuisine_purity = most_common_count / len(cluster_cuisines) if cluster_cuisines else 0
        heterogeneity = 1 - cuisine_purity

        try:
            if len(set(cluster_indices)) > 1:
                silhouette = fast_cosine_silhouette_sample(
                    self.feature_matrix, cluster_indices, sample_size=min(1000, len(recipe_indices))
                )
            else:
                silhouette = 0
        except:
            silhouette = 0

        return {
            'heterogeneity': heterogeneity,
            'silhouette': silhouette,
            'cuisine_purity': cuisine_purity,
            'size': len(recipe_indices),
            'dominant_cuisine': cuisine_counts.most_common(1)[0][0] if cuisine_counts else 'unknown',
            'cuisine_distribution': dict(cuisine_counts)
        }

    def optimized_kmeans_split(self, recipe_indices, k_values=[2, 3, 4]):
       '''Optimized K-means splitting with performance monitoring'''
        subset_features = self.feature_matrix[recipe_indices]

        best_k = None
        best_score = -1
        best_labels = None

        for k in k_values:
            try:
                memory_before = psutil.Process().memory_info().rss / 1024 / 1024
                start_time = time.time()

                if len(recipe_indices) > 10000:
                    kmeans = MiniBatchKMeans(
                        n_clusters=k,
                        random_state=42,
                        batch_size=min(2000, len(recipe_indices) // 5),
                        max_iter=100,
                        n_init=3
                    )
                elif len(recipe_indices) > 5000:
                    kmeans = MiniBatchKMeans(
                        n_clusters=k,
                        random_state=42,
                        batch_size=1000,
                        max_iter=100,
                        n_init=5
                    )
                else:
                    kmeans = KMeans(
                        n_clusters=k,
                        random_state=42,
                        n_init=5,
                        max_iter=100
                    )

                labels = kmeans.fit_predict(subset_features)

                if len(set(labels)) > 1:
                    score = fast_cosine_silhouette_sample(subset_features, labels, sample_size=500)

                    if score > best_score:
                        best_score = score
                        best_k = k
                        best_labels = labels

                elapsed = time.time() - start_time
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024
                print(f"    K={k}: score={score:.3f}, time={elapsed:.2f}s, "
                      f"memory_delta={memory_after - memory_before:.1f}MB")

            except Exception as e:
                print(f"    K={k} failed: {e}")
                continue

        return best_k, best_score, best_labels

    def should_continue_splitting(self, cluster_size, heterogeneity, silhouette, current_cluster_count):
        '''Determine if we should continue splitting based on target clusters and quality'''

        if cluster_size < self.min_cluster_size * 2:
            return False

        if self.target_clusters is not None:
            if current_cluster_count < self.target_clusters:
                min_silhouette = max(0.05, self.min_silhouette * 0.7)
                max_heterogeneity = min(0.9, self.max_heterogeneity * 1.3)

                return (heterogeneity > max_heterogeneity or
                        silhouette > min_silhouette)

            elif current_cluster_count >= self.target_clusters:
                min_silhouette = self.min_silhouette * 1.5
                max_heterogeneity = self.max_heterogeneity * 0.7

                return (heterogeneity > max_heterogeneity and
                        silhouette > min_silhouette)

        return (heterogeneity > self.max_heterogeneity and
                silhouette > self.min_silhouette)

    def recursive_kmeans_split(self, recipe_indices, cluster_id, depth=0, max_depth=5,
                               current_cluster_count=1):
        '''
        Recursively split clusters using optimized K-means with target cluster control
        '''
        if depth > max_depth:
            print(f"  Max depth {max_depth} reached for cluster {cluster_id}")
            return {cluster_id: recipe_indices}

        if len(recipe_indices) < self.min_cluster_size * 2:
            print(f"  Cluster {cluster_id} too small to split ({len(recipe_indices)} recipes)")
            return {cluster_id: recipe_indices}

        print(f"{'  ' * depth}Analyzing cluster {cluster_id} ({len(recipe_indices)} recipes, depth {depth})")

        best_k, best_score, best_labels = self.optimized_kmeans_split(recipe_indices)

        temp_labels = np.full(len(self.feature_matrix), -1)
        temp_labels[recipe_indices] = 0
        metrics = self.calculate_cluster_metrics(temp_labels, recipe_indices)

        should_split = self.should_continue_splitting(
            len(recipe_indices),
            metrics['heterogeneity'],
            best_score,
            current_cluster_count
        )

        if not should_split:
            reason = "target reached" if (self.target_clusters and
                                          current_cluster_count >= self.target_clusters) else "quality threshold"
            print(f"  Cluster {cluster_id} stopping split ({reason}): "
                  f"silhouette={best_score:.3f}, heterogeneity={metrics['heterogeneity']:.3f}")
            return {cluster_id: recipe_indices}

        print(f"  Splitting cluster {cluster_id} into {best_k} subclusters (silhouette: {best_score:.3f})")

        final_clusters = {}

        subcluster_indices_map = defaultdict(list)
        for i, label in enumerate(best_labels):
            subcluster_indices_map[label].append(recipe_indices[i])

        for sub_k, subcluster_indices in subcluster_indices_map.items():
            if len(subcluster_indices) == 0:
                continue

            sub_cluster_id = f"{cluster_id}_{sub_k}"

            temp_labels = np.full(len(self.feature_matrix), -1)
            temp_labels[subcluster_indices] = sub_k

            metrics = self.calculate_cluster_metrics(temp_labels, subcluster_indices)

            print(f"    Subcluster {sub_cluster_id}: {metrics['size']} recipes, "
                  f"purity: {metrics['cuisine_purity']:.3f}, "
                  f"dominant: {metrics['dominant_cuisine']}")

            new_cluster_count = current_cluster_count + best_k - 1

            sub_results = self.recursive_kmeans_split(
                subcluster_indices, sub_cluster_id, depth + 1, max_depth, new_cluster_count
            )
            final_clusters.update(sub_results)

        return final_clusters

    def run_recursive_clustering(self, initial_k=5, target_clusters=None):
        '''Run the complete recursive K-means clustering with optional target'''
        if target_clusters:
            print(f"TARGET: {target_clusters} final clusters")
        print("=" * 60)

        start_time = time.time()

        if target_clusters:
            self.target_clusters = target_clusters

        self.load_data_and_features()

        print(f"\nPerforming initial K-means with k={initial_k}")

        if self.feature_matrix.shape[0] > 10000:
            initial_kmeans = MiniBatchKMeans(n_clusters=initial_k, random_state=42, batch_size=2000)
        else:
            initial_kmeans = KMeans(n_clusters=initial_k, random_state=42, n_init=10)

        initial_labels = initial_kmeans.fit_predict(self.feature_matrix)
        initial_silhouette = fast_cosine_silhouette_sample(self.feature_matrix, initial_labels, sample_size=1000)

        print(f"Initial clustering silhouette score: {initial_silhouette:.3f}")

        initial_clusters = defaultdict(list)
        for i, label in enumerate(initial_labels):
            initial_clusters[label].append(i)

        if target_clusters:
            print(f"Targeting approximately {target_clusters} final clusters...")

        all_final_clusters = {}

        for cluster_id, recipe_indices in initial_clusters.items():
            print(f"\nProcessing initial cluster {cluster_id} ({len(recipe_indices)} recipes)")

            cluster_results = self.recursive_kmeans_split(
                recipe_indices, f"cluster_{cluster_id}", depth=0,
                current_cluster_count=len(all_final_clusters) + 1
            )
            all_final_clusters.update(cluster_results)

            if target_clusters and len(all_final_clusters) >= target_clusters:
                print(f"\nReached target of {target_clusters} clusters, stopping early")
                break

        self.final_clusters = all_final_clusters

        total_time = time.time() - start_time
        print(f"\nRecursive clustering completed in {total_time:.2f}s")
        print(f"Final number of clusters: {len(self.final_clusters)}")

        if target_clusters:
            actual_clusters = len(self.final_clusters)
            if actual_clusters < target_clusters * 0.8:
                print(f"Generated {actual_clusters} clusters, below target {target_clusters}")
            elif actual_clusters > target_clusters * 1.2:
                print(f"Generated {actual_clusters} clusters, above target {target_clusters}")

        # Evaluate results
        self.evaluate_final_clustering()

        # Save results
        self.save_results()

        return self.final_clusters

    def evaluate_final_clustering(self):
        '''Evaluate the quality of final clustering'''
        print("\nEVALUATING FINAL CLUSTERING")
        print("=" * 40)

        final_labels = np.full(len(self.cuisine_labels), -1)

        cluster_metrics = {}
        total_recipes = 0

        for cluster_id, recipe_indices in self.final_clusters.items():
            cluster_num = len(cluster_metrics)
            final_labels[recipe_indices] = cluster_num

            temp_labels = np.full(len(self.feature_matrix), -1)
            temp_labels[recipe_indices] = cluster_num

            metrics = self.calculate_cluster_metrics(temp_labels, recipe_indices)
            cluster_metrics[cluster_id] = metrics
            total_recipes += len(recipe_indices)

        valid_indices = final_labels != -1
        if np.sum(valid_indices) > 0:
            valid_final_labels = final_labels[valid_indices]
            valid_cuisine_labels = [self.cuisine_labels[i] for i in range(len(self.cuisine_labels))
                                    if valid_indices[i] and self.cuisine_labels[i] != 'unknown']
            valid_cluster_labels = [valid_final_labels[i] for i in range(len(valid_final_labels))
                                    if i < len(valid_cuisine_labels)]

            if len(valid_cuisine_labels) > 0 and len(set(valid_cuisine_labels)) > 1:
                unique_cuisines = list(set(valid_cuisine_labels))
                cuisine_to_num = {cuisine: i for i, cuisine in enumerate(unique_cuisines)}
                numeric_cuisines = [cuisine_to_num[cuisine] for cuisine in valid_cuisine_labels]

                ari = adjusted_rand_score(numeric_cuisines, valid_cluster_labels)
                nmi = normalized_mutual_info_score(numeric_cuisines, valid_cluster_labels)

                overall_purity = np.mean([metrics['cuisine_purity'] for metrics in cluster_metrics.values()])

                print(f"Overall ARI: {ari:.3f}")
                print(f"Overall NMI: {nmi:.3f}")
                print(f"Overall Purity: {overall_purity:.3f}")
                print(f"Number of clusters: {len(self.final_clusters)}")
                print(f"Total recipes clustered: {total_recipes}")

                self.performance_stats = {
                    'ari': ari,
                    'nmi': nmi,
                    'overall_purity': overall_purity,
                    'n_clusters': len(self.final_clusters),
                    'total_recipes': total_recipes
                }

        print(f"\nCLUSTER BREAKDOWN:")
        sorted_clusters = sorted(cluster_metrics.items(),
                                 key=lambda x: x[1]['size'], reverse=True)

        for cluster_id, metrics in sorted_clusters[:10]:  # Show top 10
            print(f"{cluster_id}: {metrics['size']} recipes, "
                  f"purity: {metrics['cuisine_purity']:.3f}, "
                  f"dominant: {metrics['dominant_cuisine']}")

    def save_results(self):
        '''Save clustering results in multiple formats'''
        print("\nSaving results...")

        cluster_assignments = {}
        for cluster_id, recipe_indices in self.final_clusters.items():
            cluster_assignments[cluster_id] = {
                'recipe_indices': recipe_indices,
                'size': len(recipe_indices),
                'cuisines': [self.cuisine_labels[i] for i in recipe_indices],
                'sample_ingredients': [self.transactions[i][:5] for i in recipe_indices[:3]]
            }

        with open(os.path.join(self.output_dir, "results", "cluster_assignments.json"), 'w') as f:
            json.dump(cluster_assignments, f, indent=2)

        with open(os.path.join(self.output_dir, "results", "performance_stats.json"), 'w') as f:
            json.dump(self.performance_stats, f, indent=2)

        import pandas as pd

        csv_data = []
        for cluster_id, recipe_indices in self.final_clusters.items():
            for recipe_idx in recipe_indices:
                csv_data.append({
                    'recipe_index': recipe_idx,
                    'cluster_id': cluster_id,
                    'cuisine': self.cuisine_labels[recipe_idx],
                    'num_ingredients': len(self.transactions[recipe_idx]),
                    'ingredients_sample': ', '.join(self.transactions[recipe_idx][:5])
                })

        df = pd.DataFrame(csv_data)
        df.to_csv(os.path.join(self.output_dir, "results", "recipe_clusters.csv"), index=False)

        print(f"Results saved to {self.output_dir}/results/")


def main_recursive_kmeans(recipes_path,
                          target_clusters=None,
                          min_cluster_size=100,
                          max_heterogeneity=0.7,
                          min_silhouette=0.1,
                          initial_k=8,
                          max_depth=5,
                          top_k_itemsets=166,
                          support_min=0.02,
                          support_max=0.6,
                          discrimination_weight=0.7,
                          k_values=None,
                          output_dir="recursive_kmeans_results"):
    '''
    Enhanced main function for recursive K-means clustering with configurable hyperparameters

    recipes_path: Path to the recipe data JSON file
    target_clusters: Target number of final clusters (None for adaptive)
    min_cluster_size: Minimum recipes per cluster before stopping splits
    max_heterogeneity: Maximum allowed heterogeneity (1-purity) before stopping splits
    min_silhouette: Minimum silhouette score to continue splitting
    initial_k : Number of initial clusters
    max_depth : Maximum recursion depth
    top_k_itemsets: Number of itemsets to select per size (2,3,5)
    support_min: Minimum support threshold for itemset selection
    support_max: Maximum support threshold for itemset selection
    discrimination_weight: Weight for discrimination score vs support in itemset selection
    k_values: K values to try for splits (default=[2,3,4])
    output_dir: Output directory for results
    '''

    if k_values is None:
        k_values = [2, 3, 4]

    # Initialize analyzer
    analyzer = RecursiveBatchKMeansClustering(
        data_path=recipes_path,
        cuisine_data_path=recipes_path,
        output_dir=output_dir,
        target_clusters=target_clusters
    )

    # Set clustering hyperparameters
    analyzer.min_cluster_size = min_cluster_size
    analyzer.max_heterogeneity = max_heterogeneity
    analyzer.min_silhouette = min_silhouette

    # Store additional parameters for use in methods
    analyzer._top_k_itemsets = top_k_itemsets
    analyzer._support_min = support_min
    analyzer._support_max = support_max
    analyzer._discrimination_weight = discrimination_weight
    analyzer._k_values = k_values
    analyzer._max_depth = max_depth

    print(f"HYPERPARAMETERS:")
    print(f"  Target clusters: {target_clusters}")
    print(f"  Min cluster size: {min_cluster_size}")
    print(f"  Max heterogeneity: {max_heterogeneity}")
    print(f"  Min silhouette: {min_silhouette}")
    print(f"  Initial K: {initial_k}")
    print(f"  Max depth: {max_depth}")
    print(f"  Top-K itemsets: {top_k_itemsets}")
    print(f"  Support range: {support_min} - {support_max}")
    print(f"  K values: {k_values}")
    print()

    # Update the _select_discriminative_itemsets method to use these parameters
    def enhanced_select_discriminative_itemsets(self, frequent_itemsets, top_k=200):
        """Enhanced discriminative itemset selection with configurable parameters"""
        support_min = getattr(self, '_support_min', 0.02)
        support_max = getattr(self, '_support_max', 0.6)
        discrimination_weight = getattr(self, '_discrimination_weight', 0.7)

        itemset_scores = []

        for itemset_key, itemset_data in frequent_itemsets.items():
            itemset = set(itemset_data['items'])
            support = itemset_data['support']


            if support > support_max or support < support_min:
                continue


            cuisine_counts = defaultdict(int)
            total_containing = 0

            for i, transaction in enumerate(self.transactions):
                if itemset.issubset(set(transaction)):
                    cuisine = self.cuisine_labels[i]
                    if cuisine != 'unknown':
                        cuisine_counts[cuisine] += 1
                        total_containing += 1

            if total_containing < 5:
                continue

            probabilities = [count / total_containing for count in cuisine_counts.values()]
            entropy = -sum(p * np.log2(p) for p in probabilities if p > 0)

            unique_cuisines = len(set(self.cuisine_labels)) - (1 if 'unknown' in self.cuisine_labels else 0)
            max_entropy = np.log2(unique_cuisines) if unique_cuisines > 1 else 1
            discrimination_score = 1 - (entropy / max_entropy) if max_entropy > 0 else 0


            support_weight = 1.0 - discrimination_weight
            combined_score = discrimination_score * discrimination_weight + support * support_weight

            itemset_scores.append({
                'items': itemset_data['items'],
                'score': combined_score
            })

        # Sort and return top k
        itemset_scores.sort(key=lambda x: x['score'], reverse=True)
        return [item['items'] for item in itemset_scores[:top_k]]


    def enhanced_create_optimized_feature_matrix(self):
        """Enhanced feature matrix creation with configurable parameters"""

        start_time = time.time()

        top_k_itemsets = getattr(self, '_top_k_itemsets', 166)
        all_itemsets = []

        for size in [2, 3, 5]:
            try:
                filepath = f"frequent_itemsets/frequent_{size}_itemsets.json"
                with open(filepath, 'r') as f:
                    frequent_itemsets = json.load(f)

                selected = self._select_discriminative_itemsets(frequent_itemsets, top_k=top_k_itemsets)
                all_itemsets.extend(selected)
                print(f"  Size {size}: {len(selected)} itemsets selected")

            except FileNotFoundError:
                print(f"  Size {size}: File not found, skipping")
                continue


        unique_itemsets = []
        seen_itemsets = set()
        for itemset in all_itemsets:
            itemset_tuple = tuple(sorted(itemset))
            if itemset_tuple not in seen_itemsets:
                unique_itemsets.append(itemset)
                seen_itemsets.add(itemset_tuple)

        print(f"Total unique itemsets: {len(unique_itemsets)}")

        n_transactions = len(self.transactions)
        n_itemsets = len(unique_itemsets)

        transaction_sets = [frozenset(transaction) for transaction in self.transactions]
        itemset_sets = [frozenset(itemset) for itemset in unique_itemsets]

        ingredient_to_transactions = defaultdict(set)
        for i, transaction_set in enumerate(transaction_sets):
            for ingredient in transaction_set:
                ingredient_to_transactions[ingredient].add(i)

        feature_matrix = np.zeros((n_transactions, n_itemsets), dtype=np.float32)

        for j, itemset_set in enumerate(itemset_sets):
            if not itemset_set:
                continue

            candidate_transactions = None
            for ingredient in itemset_set:
                if ingredient in ingredient_to_transactions:
                    ingredient_transactions = ingredient_to_transactions[ingredient]
                    if candidate_transactions is None:
                        candidate_transactions = ingredient_transactions.copy()
                    else:
                        candidate_transactions &= ingredient_transactions
                else:
                    candidate_transactions = set()
                    break

            if candidate_transactions:
                for i in candidate_transactions:
                    feature_matrix[i, j] = 1.0

        creation_time = time.time() - start_time
        print(f"Feature matrix created in {creation_time:.2f}s")
        return feature_matrix

    def enhanced_optimized_kmeans_split(self, recipe_indices, k_values=None):
        """Enhanced K-means splitting with configurable k_values and error handling"""
        if k_values is None:
            k_values = getattr(self, '_k_values', [2, 3, 4])

        subset_features = self.feature_matrix[recipe_indices]
        best_k = None
        best_score = -1
        best_labels = None

        for k in k_values:
            try:
                memory_before = psutil.Process().memory_info().rss / 1024 / 1024
                start_time = time.time()

                if len(recipe_indices) > 10000:
                    kmeans = MiniBatchKMeans(
                        n_clusters=k, random_state=42,
                        batch_size=min(2000, len(recipe_indices) // 5),
                        max_iter=100, n_init=3
                    )
                elif len(recipe_indices) > 5000:
                    kmeans = MiniBatchKMeans(
                        n_clusters=k, random_state=42,
                        batch_size=1000, max_iter=100, n_init=5
                    )
                else:
                    kmeans = KMeans(
                        n_clusters=k, random_state=42,
                        n_init=5, max_iter=100
                    )

                labels = kmeans.fit_predict(subset_features)

                unique_labels = len(set(labels))
                if unique_labels < 2:
                    print(f"    K={k}: only {unique_labels} distinct cluster(s) found, skipping")
                    continue

                if len(set(labels)) > 1:
                    score = fast_cosine_silhouette_sample(subset_features, labels, sample_size=500)
                    if score > best_score:
                        best_score = score
                        best_k = k
                        best_labels = labels

                elapsed = time.time() - start_time
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024
                print(f"    K={k}: score={score:.3f}, time={elapsed:.2f}s, "
                      f"memory_delta={memory_after - memory_before:.1f}MB")

            except Exception as e:
                print(f"K={k} failed: {e}")
                continue

        if best_labels is None:
            print(f"No valid clustering found for any k in {k_values}")
            best_k = None
            best_score = -1.0

        return best_k, best_score, best_labels

    def enhanced_recursive_kmeans_split(self, recipe_indices, cluster_id, depth=0, max_depth=None,
                                        current_cluster_count=1):
        """Enhanced recursive splitting with configurable max_depth and error handling"""
        if max_depth is None:
            max_depth = getattr(self, '_max_depth', 5)

        if depth > max_depth:
            print(f"  Max depth {max_depth} reached for cluster {cluster_id}")
            return {cluster_id: recipe_indices}

        if len(recipe_indices) < self.min_cluster_size * 2:
            print(f"  Cluster {cluster_id} too small to split ({len(recipe_indices)} recipes)")
            return {cluster_id: recipe_indices}

        print(f"{'  ' * depth}Analyzing cluster {cluster_id} ({len(recipe_indices)} recipes, depth {depth})")

        best_k, best_score, best_labels = self.optimized_kmeans_split(recipe_indices)


        if best_labels is None or best_k is None:
            print(f"  Cluster {cluster_id} cannot be split (clustering failed)")
            return {cluster_id: recipe_indices}

        temp_labels = np.full(len(self.feature_matrix), -1)
        temp_labels[recipe_indices] = 0
        metrics = self.calculate_cluster_metrics(temp_labels, recipe_indices)

        should_split = self.should_continue_splitting(
            len(recipe_indices), metrics['heterogeneity'], best_score, current_cluster_count
        )

        if not should_split:
            reason = "target reached" if (
                    self.target_clusters and current_cluster_count >= self.target_clusters) else "quality threshold"
            print(
                f"  Cluster {cluster_id} stopping split ({reason}): silhouette={best_score:.3f}, heterogeneity={metrics['heterogeneity']:.3f}")
            return {cluster_id: recipe_indices}

        print(f"  Splitting cluster {cluster_id} into {best_k} subclusters (silhouette: {best_score:.3f})")

        final_clusters = {}
        subcluster_indices_map = defaultdict(list)
        for i, label in enumerate(best_labels):
            subcluster_indices_map[label].append(recipe_indices[i])

        for sub_k, subcluster_indices in subcluster_indices_map.items():
            if len(subcluster_indices) == 0:
                continue

            sub_cluster_id = f"{cluster_id}_{sub_k}"
            temp_labels = np.full(len(self.feature_matrix), -1)
            temp_labels[subcluster_indices] = sub_k
            metrics = self.calculate_cluster_metrics(temp_labels, subcluster_indices)

            print(f"    Subcluster {sub_cluster_id}: {metrics['size']} recipes, "
                  f"purity: {metrics['cuisine_purity']:.3f}, dominant: {metrics['dominant_cuisine']}")

            new_cluster_count = current_cluster_count + best_k - 1
            sub_results = self.recursive_kmeans_split(
                subcluster_indices, sub_cluster_id, depth + 1, max_depth, new_cluster_count
            )
            final_clusters.update(sub_results)

        return final_clusters


    import types
    analyzer._select_discriminative_itemsets = types.MethodType(enhanced_select_discriminative_itemsets, analyzer)
    analyzer._create_optimized_feature_matrix = types.MethodType(enhanced_create_optimized_feature_matrix, analyzer)
    analyzer.optimized_kmeans_split = types.MethodType(enhanced_optimized_kmeans_split, analyzer)
    analyzer.recursive_kmeans_split = types.MethodType(enhanced_recursive_kmeans_split, analyzer)


    final_clusters = analyzer.run_recursive_clustering(initial_k=initial_k, target_clusters=target_clusters)

    print(f"\nRecursive K-means clustering complete!")
    print(f"Generated {len(final_clusters)} final clusters")
    if target_clusters:
        deviation = abs(len(final_clusters) - target_clusters) / target_clusters * 100
        print(f"Target was {target_clusters} clusters (deviation: {deviation:.1f}%)")

    return analyzer, final_clusters

if __name__ == "__main__":
    recipes_path = "recipes_database_only_results_str.json"

    analyzer, clusters = main_recursive_kmeans(
        recipes_path,
        target_clusters=20,
        min_cluster_size=200,
        max_heterogeneity=0.5,
        min_silhouette=0.2,
        output_dir="pure_clusters_20"
    )
