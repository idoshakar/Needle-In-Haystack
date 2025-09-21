import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.metrics import jaccard_score
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter


def enhanced_analyze_clusters(clustering_obj, clusters, min_size=3, top_k=10):
    '''
    Enhanced cluster analysis with better interpretability

    Parameters:
    clustering_obj: FrequentItemsetClustering instance
    clusters: array of cluster labels
    min_size: minimum cluster size
    top_k: number of top itemsets to show
    '''
    filtered_clusters, filter_info = clustering_obj.filter_small_clusters(
        clusters, min_size=min_size, strategy='merge_nearest'
    )

    unique_clusters = np.unique(filtered_clusters)
    cluster_characteristics = {}

    print(f"\n🔍 Enhanced Cluster Analysis ({len(unique_clusters)} clusters):")
    print("=" * 60)

    for cluster_id in unique_clusters:
        cluster_indices = np.where(filtered_clusters == cluster_id)[0]
        cluster_size = len(cluster_indices)

        print(f"\nCluster {cluster_id} (Size: {cluster_size}, {cluster_size / len(clusters) * 100:.1f}%):")
        print("-" * 40)

        cluster_feature_means = clustering_obj.feature_matrix[cluster_indices].mean(axis=0)

        top_itemsets_idx = np.argsort(cluster_feature_means)[-top_k:][::-1]

        print("Top frequent itemsets in this cluster:")
        cluster_itemsets = []
        found_significant = False

        for idx in top_itemsets_idx:
            if cluster_feature_means[idx] > 0.1:  # Only show significant ones
                itemset_items = list(clustering_obj.frequent_itemsets[idx])
                freq = cluster_feature_means[idx]
                print(f"   • {itemset_items}: {freq:.1%}")
                cluster_itemsets.append((itemset_items, freq))
                found_significant = True

        # If no significant itemsets, show top 5 regardless
        if not found_significant:
            print("   (Showing top 5 itemsets regardless of frequency:)")
            for idx in top_itemsets_idx[:5]:
                if cluster_feature_means[idx] > 0:
                    itemset_items = list(clustering_obj.frequent_itemsets[idx])
                    freq = cluster_feature_means[idx]
                    print(f"   • {itemset_items}: {freq:.1%}")
                    cluster_itemsets.append((itemset_items, freq))

        # 2. Analyze individual ingredients frequency
        print("\nIndividual ingredient frequencies:")
        ingredient_counter = Counter()
        for idx in cluster_indices:
            transaction = clustering_obj.transactions[idx]
            ingredient_counter.update(transaction)

        total_transactions = len(cluster_indices)
        for ingredient, count in ingredient_counter.most_common(10):
            freq = count / total_transactions
            print(f"   • {ingredient}: {freq:.1%} ({count}/{total_transactions})")

        print(f"\nSample transactions (showing 5/{cluster_size}):")
        sample_indices = cluster_indices[:5] if len(cluster_indices) >= 5 else cluster_indices
        for i, idx in enumerate(sample_indices):
            transaction_items = list(clustering_obj.transactions[idx])[:8]  # Limit display
            more_items = "..." if len(clustering_obj.transactions[idx]) > 8 else ""
            print(f"   {i + 1}. T{idx}: {transaction_items}{more_items}")

        cluster_characteristics[cluster_id] = {
            'size': cluster_size,
            'top_itemsets': cluster_itemsets[:5],
            'top_ingredients': ingredient_counter.most_common(10),
            'avg_transaction_size': np.mean([len(clustering_obj.transactions[idx]) for idx in cluster_indices])
        }

    print(f"\nCluster Comparison:")
    print("=" * 40)
    for cluster_id in unique_clusters:
        char = cluster_characteristics[cluster_id]
        print(f"Cluster {cluster_id}: Avg transaction size = {char['avg_transaction_size']:.1f} ingredients")

    return filtered_clusters, cluster_characteristics


def optimize_clustering_parameters(clustering_obj, max_clusters=10):
    '''
    Find optimal number of clusters using multiple metrics

    Parameters:
    clustering_obj: FrequentItemsetClustering instance
    max_clusters: maximum number of clusters to test
    '''
    metrics_results = {
        'n_clusters': [],
        'silhouette': [],
        'calinski_harabasz': [],
        'separation_ratio': []
    }

    print("Optimizing clustering parameters...")

    for n in range(2, max_clusters + 1):
        clusters_temp = clustering_obj.get_clusters(n_clusters=n)
        metrics = clustering_obj.evaluate_clustering_quality(clusters_temp)

        if 'error' not in metrics:
            metrics_results['n_clusters'].append(n)
            metrics_results['silhouette'].append(metrics['silhouette_score'])
            metrics_results['calinski_harabasz'].append(metrics['calinski_harabasz_score'])
            metrics_results['separation_ratio'].append(metrics['separation_ratio'])

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(metrics_results['n_clusters'], metrics_results['silhouette'], 'bo-')
    axes[0].set_title('Silhouette Score')
    axes[0].set_xlabel('Number of Clusters')
    axes[0].set_ylabel('Score')

    axes[1].plot(metrics_results['n_clusters'], metrics_results['calinski_harabasz'], 'ro-')
    axes[1].set_title('Calinski-Harabasz Score')
    axes[1].set_xlabel('Number of Clusters')
    axes[1].set_ylabel('Score')

    axes[2].plot(metrics_results['n_clusters'], metrics_results['separation_ratio'], 'go-')
    axes[2].set_title('Separation Ratio')
    axes[2].set_xlabel('Number of Clusters')
    axes[2].set_ylabel('Ratio')

    plt.tight_layout()
    plt.show()

    if metrics_results['silhouette']:
        best_n_silhouette = metrics_results['n_clusters'][
            np.argmax(metrics_results['silhouette'])
        ]
        best_n_ch = metrics_results['n_clusters'][
            np.argmax(metrics_results['calinski_harabasz'])
        ]

        print(f"🎯 Optimal clusters by Silhouette Score: {best_n_silhouette}")
        print(f"🎯 Optimal clusters by Calinski-Harabasz: {best_n_ch}")

    return metrics_results


def compute_cluster_purity(clustering_obj, clusters, cuisine_labels=None):
    '''
    Compute cluster purity if you have cuisine labels

    Parameters:
    clustering_obj: FrequentItemsetClustering instance
    clusters: array of cluster labels
    cuisine_labels: list of cuisine labels for each transaction
    '''
    if cuisine_labels is None:
        print("No cuisine labels provided for purity calculation")
        return None

    unique_clusters = np.unique(clusters)
    cluster_purities = {}

    print(f"\n🎯 Cluster Purity Analysis:")
    print("=" * 40)

    for cluster_id in unique_clusters:
        cluster_indices = np.where(clusters == cluster_id)[0]
        cluster_cuisines = [cuisine_labels[i] for i in cluster_indices]
        cuisine_counts = Counter(cluster_cuisines)

        max_cuisine_count = max(cuisine_counts.values())
        purity = max_cuisine_count / len(cluster_indices)

        print(f"Cluster {cluster_id} (purity: {purity:.1%}):")
        for cuisine, count in cuisine_counts.most_common(5):
            percentage = count / len(cluster_indices)
            print(f"   • {cuisine}: {percentage:.1%} ({count}/{len(cluster_indices)})")

        cluster_purities[cluster_id] = purity

    overall_purity = np.mean(list(cluster_purities.values()))
    print(f"\nOverall clustering purity: {overall_purity:.1%}")

    return cluster_purities


class FrequentItemsetClustering:
    def __init__(self, frequent_itemsets, transactions):
        '''
        Initialize the clustering with frequent itemsets and transactions

        Parameters:
        frequent_itemsets: list of frozensets or lists containing the frequent itemsets
        transactions: list of lists/sets containing the original transactions
        '''
        # FIXED
        self.frequent_itemsets = [frozenset([itemset]) if isinstance(itemset, str)
                                  else frozenset(itemset) if not isinstance(itemset, frozenset)
                            else itemset for itemset in frequent_itemsets]
        self.transactions = [set(transaction) if not isinstance(transaction, set)
                             else transaction for transaction in transactions]
        self.feature_matrix = None
        self.linkage_matrix = None
        self.transaction_labels = None

    def optimize_clustering_parameters(self, max_clusters=10):
        '''
        Find optimal number of clusters using multiple metrics
        '''
        import matplotlib.pyplot as plt

        metrics_results = {
            'n_clusters': [],
            'silhouette': [],
            'calinski_harabasz': [],
            'separation_ratio': []
        }

        print("Optimizing clustering parameters...")

        for n in range(2, max_clusters + 1):
            clusters = self.get_clusters(n_clusters=n)
            metrics = self.evaluate_clustering_quality(clusters)

            if 'error' not in metrics:
                metrics_results['n_clusters'].append(n)
                metrics_results['silhouette'].append(metrics['silhouette_score'])
                metrics_results['calinski_harabasz'].append(metrics['calinski_harabasz_score'])
                metrics_results['separation_ratio'].append(metrics['separation_ratio'])

        fig, axes = plt.subplots(1, 3, figsize=(15, 4))

        axes[0].plot(metrics_results['n_clusters'], metrics_results['silhouette'], 'bo-')
        axes[0].set_title('Silhouette Score')
        axes[0].set_xlabel('Number of Clusters')
        axes[0].set_ylabel('Score')

        axes[1].plot(metrics_results['n_clusters'], metrics_results['calinski_harabasz'], 'ro-')
        axes[1].set_title('Calinski-Harabasz Score')
        axes[1].set_xlabel('Number of Clusters')
        axes[1].set_ylabel('Score')

        axes[2].plot(metrics_results['n_clusters'], metrics_results['separation_ratio'], 'go-')
        axes[2].set_title('Separation Ratio')
        axes[2].set_xlabel('Number of Clusters')
        axes[2].set_ylabel('Ratio')

        plt.tight_layout()
        plt.show()

        best_n_silhouette = metrics_results['n_clusters'][
            np.argmax(metrics_results['silhouette'])
        ]
        best_n_ch = metrics_results['n_clusters'][
            np.argmax(metrics_results['calinski_harabasz'])
        ]

        print(f"🎯 Optimal clusters by Silhouette Score: {best_n_silhouette}")
        print(f"🎯 Optimal clusters by Calinski-Harabasz: {best_n_ch}")

        return metrics_results

    def create_feature_matrix(self):
        '''
        Create binary feature matrix where rows=transactions, cols=frequent itemsets
        '''
        n_transactions = len(self.transactions)
        n_features = len(self.frequent_itemsets)

        self.feature_matrix = np.zeros((n_transactions, n_features), dtype=int)

        for i, transaction in enumerate(self.transactions):
            for j, itemset in enumerate(self.frequent_itemsets):
                if itemset.issubset(transaction):
                    self.feature_matrix[i, j] = 1

        print(f"Feature matrix shape: {self.feature_matrix.shape}")
        print(f"Sparsity: {(self.feature_matrix == 0).sum() / self.feature_matrix.size:.2%}")

        return self.feature_matrix

    def compute_distance_matrix(self, metric='jaccard'):
        '''
            Optimized distance matrix computation with better handling of edge cases
            '''
        if self.feature_matrix is None:
            self.create_feature_matrix()

        if metric == 'jaccard':
            def jaccard_distance(u, v):
                intersection = np.sum(np.logical_and(u, v))
                union = np.sum(np.logical_or(u, v))

                if union == 0:
                    return 0.0

                return 1.0 - (intersection / union)

            return pdist(self.feature_matrix, metric=jaccard_distance)

        elif metric == 'cosine':
            return pdist(self.feature_matrix, metric='cosine')

        elif metric == 'hamming':
            return pdist(self.feature_matrix, metric='hamming')

        elif metric == 'euclidean':
            return pdist(self.feature_matrix, metric='euclidean')

        else:
            raise ValueError("Metric must be 'jaccard', 'cosine', 'hamming', or 'euclidean'")

    def perform_clustering(self, linkage_method='average', distance_metric='jaccard'):
        '''
        Perform hierarchical clustering

        Parameters:
        linkage_method: 'single', 'complete', 'average', 'ward'
        distance_metric: 'jaccard', 'cosine', 'hamming', 'euclidean'
        '''
        # Compute distance matrix
        distances = self.compute_distance_matrix(distance_metric)

        if linkage_method == 'ward' and distance_metric != 'euclidean':
            print("Warning: Ward linkage requires Euclidean distance. Switching to Euclidean.")
            distances = self.compute_distance_matrix('euclidean')

        self.linkage_matrix = linkage(distances, method=linkage_method)

        print(f"Clustering completed using {linkage_method} linkage with {distance_metric} distance")
        return self.linkage_matrix

    def evaluate_clustering_quality(self, clusters):
        '''
        Evaluate clustering quality using multiple metrics
        '''
        from sklearn.metrics import silhouette_score, calinski_harabasz_score

        if len(np.unique(clusters)) < 2:
            return {"error": "Need at least 2 clusters for evaluation"}

        distances = self.compute_distance_matrix()
        distance_matrix = squareform(distances)

        silhouette = silhouette_score(distance_matrix, clusters, metric='precomputed')

        ch_score = calinski_harabasz_score(self.feature_matrix, clusters)

        intra_distances = []
        inter_distances = []

        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                dist = distance_matrix[i, j]
                if clusters[i] == clusters[j]:
                    intra_distances.append(dist)
                else:
                    inter_distances.append(dist)

        avg_intra = np.mean(intra_distances) if intra_distances else 0
        avg_inter = np.mean(inter_distances) if inter_distances else 0

        return {
            "silhouette_score": silhouette,
            "calinski_harabasz_score": ch_score,
            "avg_intra_cluster_distance": avg_intra,
            "avg_inter_cluster_distance": avg_inter,
            "separation_ratio": avg_inter / avg_intra if avg_intra > 0 else float('inf')
        }


    def plot_dendrogram(self, max_d=None, figsize=(12, 8), show_labels=True):
        '''
        Plot the dendrogram

        Parameters:
        max_d: maximum distance to draw horizontal line
        figsize: figure size
        show_labels: whether to show transaction labels
        '''
        plt.figure(figsize=figsize)

        if show_labels:
            labels = [f"T{i}" for i in range(len(self.transactions))]
        else:
            labels = None

        dendrogram_plot = dendrogram(
            self.linkage_matrix,
            labels=labels,
            leaf_rotation=90,
            leaf_font_size=8
        )

        if max_d is not None:
            plt.axhline(y=max_d, color='r', linestyle='--', alpha=0.7,
                        label=f'Cut at distance {max_d}')
            plt.legend()

        plt.title('Hierarchical Clustering Dendrogram\n(Based on Frequent Itemsets)')
        plt.xlabel('Transaction Index')
        plt.ylabel('Distance')
        plt.tight_layout()
        plt.show()

        return dendrogram_plot

    def get_clusters(self, n_clusters=None, distance_threshold=None):
        '''
        Extract flat clusters from the hierarchical clustering

        Parameters:
        n_clusters: number of clusters to extract
        distance_threshold: distance threshold to cut the dendrogram
        '''
        if self.linkage_matrix is None:
            raise ValueError("Must perform clustering first")

        if n_clusters is not None:
            clusters = fcluster(self.linkage_matrix, n_clusters, criterion='maxclust')
        elif distance_threshold is not None:
            clusters = fcluster(self.linkage_matrix, distance_threshold, criterion='distance')
        else:
            raise ValueError("Must specify either n_clusters or distance_threshold")

        return clusters

    def analyze_clusters(self, clusters, min_size=3):
        '''
        Analyze the characteristics of each cluster
        '''
        filtered_clusters, filter_info = self.filter_small_clusters(
            clusters, min_size=min_size, strategy='merge_nearest'
        )

        validation = self.validate_cluster_sizes(filtered_clusters, min_size)

        print(f"Filtering Summary: {filter_info}")
        print(f"Validation Results: {validation}")

        unique_clusters = np.unique(filtered_clusters)

        print(f"\nCluster Analysis ({len(unique_clusters)} clusters):")
        print("=" * 50)

        for cluster_id in unique_clusters:
            cluster_indices = np.where(filtered_clusters == cluster_id)[0]
            cluster_size = len(cluster_indices)

            print(f"\nCluster {cluster_id} (Size: {cluster_size}):")
            print("-" * 30)

            print("Transactions:")
            for idx in cluster_indices[:5]:
                transaction_items = list(self.transactions[idx])
                print(f"  T{idx}: {transaction_items}")
            if len(cluster_indices) > 5:
                print(f"  ... and {len(cluster_indices) - 5} more")

            cluster_feature_means = self.feature_matrix[cluster_indices].mean(axis=0)
            top_itemsets_idx = np.argsort(cluster_feature_means)[-5:][::-1]

            print("Top frequent itemsets in this cluster:")
            for idx in top_itemsets_idx:
                if cluster_feature_means[idx] > 0:
                    itemset_items = list(self.frequent_itemsets[idx])
                    print(f"  {itemset_items}: {cluster_feature_means[idx]:.2f}")

        return filtered_clusters
    def plot_feature_heatmap(self, clusters=None, figsize=(12, 8)):
        '''
        Plot heatmap of feature matrix, optionally grouped by clusters
        '''
        if self.feature_matrix is None:
            self.create_feature_matrix()

        plt.figure(figsize=figsize)

        feature_df = pd.DataFrame(
            self.feature_matrix,
            columns=[f"F{i}: {list(itemset)}" for i, itemset in enumerate(self.frequent_itemsets)],
            index=[f"T{i}" for i in range(len(self.transactions))]
        )

        if clusters is not None:
            cluster_order = np.argsort(clusters)
            feature_df = feature_df.iloc[cluster_order]

        sns.heatmap(feature_df, cmap='Blues', cbar=True,
                    xticklabels=True, yticklabels=True)
        plt.title('Transaction-Itemset Feature Matrix')
        plt.xlabel('Frequent Itemsets')
        plt.ylabel('Transactions')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

    def filter_small_clusters(self, clusters, min_size=3, strategy='merge_nearest'):
        '''
        Handle small clusters using various strategies

        Parameters:
        clusters: array of cluster labels
        min_size: minimum acceptable cluster size
        strategy: 'remove', 'merge_nearest', 'outlier_class', 'reassign'
        '''
        import numpy as np
        from collections import Counter

        cluster_counts = Counter(clusters)
        small_clusters = [c for c, count in cluster_counts.items() if count < min_size]

        if not small_clusters:
            return clusters, {}

        filtered_clusters = clusters.copy()
        cluster_info = {'small_clusters_found': small_clusters, 'strategy_used': strategy}

        if strategy == 'remove':
            for small_cluster in small_clusters:
                filtered_clusters[clusters == small_cluster] = -1
            cluster_info['outliers_marked'] = np.sum(filtered_clusters == -1)

        elif strategy == 'merge_nearest':
            large_clusters = [c for c, count in cluster_counts.items() if count >= min_size]

            for small_cluster in small_clusters:
                small_indices = np.where(clusters == small_cluster)[0]

                min_avg_distance = float('inf')
                best_target = None

                for large_cluster in large_clusters:
                    large_indices = np.where(clusters == large_cluster)[0]

                    distances = []
                    for i in small_indices:
                        for j in large_indices:
                            if hasattr(self, 'distance_matrix_full'):
                                distances.append(self.distance_matrix_full[i, j])
                            else:
                                intersection = np.sum(np.logical_and(
                                    self.feature_matrix[i], self.feature_matrix[j]))
                                union = np.sum(np.logical_or(
                                    self.feature_matrix[i], self.feature_matrix[j]))
                                dist = 1 - (intersection / union) if union > 0 else 0
                                distances.append(dist)

                    avg_distance = np.mean(distances)
                    if avg_distance < min_avg_distance:
                        min_avg_distance = avg_distance
                        best_target = large_cluster

                if best_target is not None:
                    filtered_clusters[clusters == small_cluster] = best_target
                    cluster_info[f'merged_{small_cluster}_into_{best_target}'] = len(small_indices)

        elif strategy == 'outlier_class':
            misc_cluster_id = max(clusters) + 1
            for small_cluster in small_clusters:
                filtered_clusters[clusters == small_cluster] = misc_cluster_id
            cluster_info['misc_cluster_id'] = misc_cluster_id
            cluster_info['misc_cluster_size'] = sum(cluster_counts[c] for c in small_clusters)

        elif strategy == 'reassign':
            large_clusters = [c for c, count in cluster_counts.items() if count >= min_size]

            for small_cluster in small_clusters:
                small_indices = np.where(clusters == small_cluster)[0]

                for idx in small_indices:
                    min_distance = float('inf')
                    best_cluster = None

                    for large_cluster in large_clusters:
                        large_indices = np.where(clusters == large_cluster)[0]

                        distances_to_cluster = []
                        for large_idx in large_indices:
                            intersection = np.sum(np.logical_and(
                                self.feature_matrix[idx], self.feature_matrix[large_idx]))
                            union = np.sum(np.logical_or(
                                self.feature_matrix[idx], self.feature_matrix[large_idx]))
                            dist = 1 - (intersection / union) if union > 0 else 0
                            distances_to_cluster.append(dist)

                        min_dist_to_cluster = min(distances_to_cluster)
                        if min_dist_to_cluster < min_distance:
                            min_distance = min_dist_to_cluster
                            best_cluster = large_cluster

                    if best_cluster is not None:
                        filtered_clusters[idx] = best_cluster

        return filtered_clusters, cluster_info

    def validate_cluster_sizes(self, clusters, min_size=3):
        '''
        Validate that clusters meet minimum size requirements
        '''
        from collections import Counter

        cluster_counts = Counter(clusters)
        validation_results = {
            'total_clusters': len(cluster_counts),
            'valid_clusters': sum(1 for count in cluster_counts.values() if count >= min_size),
            'small_clusters': sum(1 for count in cluster_counts.values() if count < min_size),
            'cluster_sizes': dict(cluster_counts),
            'meets_min_size': all(count >= min_size for count in cluster_counts.values())
        }

        return validation_results

def extract_matched_ingredients(path):
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)

        cuisines = []
        for recipe in data:
            cuisines.append(recipe['matched_ingredients'])
        return cuisines

def example_usage():
    '''
    Example with sample transaction data using enhanced analysis functions
    '''
    import json

    path = "ingredient_matching_best_of_both_results.json"
    all_transactions = extract_matched_ingredients(path)
    sample_transactions = all_transactions[:5000]
    with open("frequent_itemsets_whats-cooking/frequent_2_itemsets.json", 'r') as f:
        one_frequent = json.load(f)

    sorted_itemsets = sorted(one_frequent.items(),
                             key=lambda x: x[1]['support'],
                             reverse=True)
    cumulative_coverage = 0
    selected_itemsets = []
    target_coverage = 0.8

    for item, data in sorted_itemsets:
        selected_itemsets.append(data['items'])
        cumulative_coverage += data['support']
        if cumulative_coverage >= target_coverage:
            break

    sample_frequent_itemsets = selected_itemsets

    print(f"Dataset Summary:")
    print(f"   • Total transactions: {len(sample_transactions)}")
    print(f"   • Selected frequent itemsets: {len(sample_frequent_itemsets)}")
    print(f"   • Target coverage achieved: {cumulative_coverage:.1%}")

    clustering = FrequentItemsetClustering(sample_frequent_itemsets, sample_transactions)

    feature_matrix = clustering.create_feature_matrix()
    print(f"\nFeature Matrix:")
    print(f"   • Shape: {feature_matrix.shape}")
    print(f"   • Sparsity: {(feature_matrix == 0).sum() / feature_matrix.size:.2%}")

    print(f"\nPerforming hierarchical clustering...")
    linkage_matrix = clustering.perform_clustering(
        linkage_method='average',
        distance_metric='jaccard'
    )

    # Plot dendrogram
    print(f"\nPlotting dendrogram...")
    clustering.plot_dendrogram(max_d=0.5)

    # Extract clusters using distance threshold
    clusters = clustering.get_clusters(distance_threshold=0.9)
    print(f"\nExtracted {len(set(clusters))} clusters using distance threshold 0.9")

    print(f"\n" + "=" * 70)
    print(f"🔍 ENHANCED CLUSTER ANALYSIS")
    print(f"=" * 70)

    filtered_clusters, characteristics = enhanced_analyze_clusters(
        clustering, clusters, min_size=3, top_k=10
    )

    print(f"\n" + "=" * 70)
    print(f"⚙️ CLUSTERING OPTIMIZATION")
    print(f"=" * 70)

    metrics_results = optimize_clustering_parameters(clustering, max_clusters=8)

    print(f"\n" + "=" * 70)
    print(f"🔄 TESTING DIFFERENT CLUSTER NUMBERS")
    print(f"=" * 70)

    for n_clusters in [3, 4, 5]:
        print(f"\n--- Testing {n_clusters} clusters ---")
        test_clusters = clustering.get_clusters(n_clusters=n_clusters)

        unique_clusters = set(test_clusters)
        cluster_sizes = [sum(1 for c in test_clusters if c == cluster_id)
                         for cluster_id in unique_clusters]

        print(f"Cluster sizes: {cluster_sizes}")

        quality_metrics = clustering.evaluate_clustering_quality(test_clusters)
        if 'error' not in quality_metrics:
            print(f"Silhouette score: {quality_metrics['silhouette_score']:.3f}")
            print(f"Calinski-Harabasz score: {quality_metrics['calinski_harabasz_score']:.1f}")

    print(f"\nPlotting feature heatmap...")
    clustering.plot_feature_heatmap(filtered_clusters)

    results = {
        'clustering_obj': clustering,
        'original_clusters': clusters,
        'filtered_clusters': filtered_clusters,
        'characteristics': characteristics,
        'metrics_results': metrics_results,
        'feature_matrix': feature_matrix
    }

    return results


def analyze_cluster_differences(clustering, characteristics):
    '''
    Additional analysis function to compare clusters more deeply
    '''
    print(f"\nDETAILED CLUSTER COMPARISON:")
    print("=" * 50)

    cluster_ids = list(characteristics.keys())

    # Compare average transaction sizes
    print(f"\n📏 Transaction Size Analysis:")
    for cluster_id in cluster_ids:
        avg_size = characteristics[cluster_id]['avg_transaction_size']
        size = characteristics[cluster_id]['size']
        print(f"   Cluster {cluster_id}: {avg_size:.1f} avg ingredients ({size} recipes)")

    print(f"\nIngredient Overlap Analysis:")
    all_ingredients = {}
    for cluster_id in cluster_ids:
        top_ingredients = [ing for ing, count in characteristics[cluster_id]['top_ingredients'][:5]]
        all_ingredients[cluster_id] = set(top_ingredients)
        print(f"   Cluster {cluster_id} top 5: {top_ingredients}")

    if len(cluster_ids) == 2:
        common = all_ingredients[cluster_ids[0]] & all_ingredients[cluster_ids[1]]
        unique_1 = all_ingredients[cluster_ids[0]] - all_ingredients[cluster_ids[1]]
        unique_2 = all_ingredients[cluster_ids[1]] - all_ingredients[cluster_ids[0]]

        print(f"\nIngredient Overlap:")
        print(f"   Common ingredients: {list(common)}")
        print(f"   Unique to Cluster {cluster_ids[0]}: {list(unique_1)}")
        print(f"   Unique to Cluster {cluster_ids[1]}: {list(unique_2)}")

def example_usage_2():
    '''
    Example with sample transaction data
    '''

    sample_transactions = [
        ['bread', 'milk', 'eggs'],
        ['bread', 'butter'],
        ['milk', 'eggs', 'cheese'],
        ['bread', 'milk', 'butter'],
        ['eggs', 'cheese', 'yogurt'],
        ['bread', 'milk', 'eggs', 'butter'],
        ['cheese', 'yogurt', 'milk'],
        ['bread', 'butter', 'jam'],
        ['milk', 'eggs'],
        ['bread', 'milk']
    ]

    sample_frequent_itemsets = [
        ['bread'],
        ['milk'],
        ['eggs'],
        ['butter'],
        ['cheese'],
        ['bread', 'milk'],
        ['bread', 'butter'],
        ['milk', 'eggs'],
        ['eggs', 'cheese'],
        ['bread', 'milk', 'eggs']
    ]

    clustering = FrequentItemsetClustering(sample_frequent_itemsets, sample_transactions)
    print(clustering.frequent_itemsets)
    print(clustering.transactions)

    feature_matrix = clustering.create_feature_matrix()
    print("Feature matrix:")
    print(feature_matrix)

    linkage_matrix = clustering.perform_clustering(
        linkage_method='average',
        distance_metric='jaccard'
    )

    clustering.plot_dendrogram(max_d=0.5)

    clusters = clustering.get_clusters(distance_threshold=0.9)

    clustering.analyze_clusters(clusters)

    clustering.plot_feature_heatmap(clusters)

    return clustering, clusters

def cluster_by_cuisines():
    path = "../whats-cooking/train.json/train.json"
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    cuisine_clusters = {}
    for recipe in data:
        cuisine_clusters[recipe['cuisine']] = recipe['id']
    return cuisine_clusters


def fix_one():
    path = "frequent_itemsets_whats-cooking/frequent_1_itemsets.json"
    with open(path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    new_data = {}
    for item in data:
        new_item = data[item]
        new_item['items'] = [item]
        new_data[item] = new_item

    with open(path, 'w', encoding='utf-8') as file:
        json.dump(new_data, file, indent=4)


if __name__ == "__main__":
    pass
