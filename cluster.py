import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
import plotly.express as px
from collections import Counter, defaultdict
import warnings

warnings.filterwarnings('ignore')

# Import your existing classes
from Hierarchial_clustering import FrequentItemsetClustering


class ComprehensiveClusteringAnalysis:
    def __init__(self, data_path, cuisine_data_path):
        """
        Initialize comprehensive analysis

        Parameters:
        data_path: path to ingredient_matching_best_of_both_results.json
        cuisine_data_path: path to original train.json with cuisine labels
        """
        self.data_path = data_path
        self.cuisine_data_path = cuisine_data_path
        self.transactions = None
        self.cuisine_labels = None
        self.cuisine_to_recipes = {}
        self.results = {}

    def load_data(self):
        """Load and prepare all data"""
        print("Loading transaction data...")
        with open(self.data_path, 'r', encoding='utf-8') as f:
            transaction_data = json.load(f)
        self.transactions = [recipe['matched_ingredients'] for recipe in transaction_data]

        print("Loading cuisine data...")
        with open(self.cuisine_data_path, 'r', encoding='utf-8') as f:
            cuisine_data = json.load(f)

        # Create cuisine labels array matching transaction indices
        self.cuisine_labels = []
        recipe_id_to_cuisine = {recipe['id']: recipe['cuisine'] for recipe in cuisine_data}

        for recipe in transaction_data:
            if 'recipe_id' in recipe and recipe['recipe_id'] in recipe_id_to_cuisine:
                self.cuisine_labels.append(recipe_id_to_cuisine[recipe['recipe_id']])
            else:
                self.cuisine_labels.append('unknown')

        # Create cuisine clusters dictionary
        self.cuisine_to_recipes = defaultdict(list)
        for idx, cuisine in enumerate(self.cuisine_labels):
            self.cuisine_to_recipes[cuisine].append(idx)

        print(f"Loaded {len(self.transactions)} recipes across {len(self.cuisine_to_recipes)} cuisines")

        # Print cuisine distribution
        cuisine_counts = Counter(self.cuisine_labels)
        print("\nCuisine distribution:")
        for cuisine, count in cuisine_counts.most_common(10):
            print(f"  {cuisine}: {count} recipes")

    def load_frequent_itemsets(self, itemset_size):
        """Load frequent itemsets for given size"""
        filepath = f"frequent_itemsets_whats-cooking/frequent_{itemset_size}_itemsets.json"
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: {filepath} not found")
            return None

    def stratified_sample(self, sample_size):
        """
        Perform stratified sampling to maintain cuisine proportions

        Parameters:
        sample_size: total number of samples to draw

        Returns:
        list of indices for sampled recipes
        """
        from collections import Counter

        cuisine_counts = Counter(self.cuisine_labels)
        total_recipes = len(self.cuisine_labels)

        # Calculate samples per cuisine (proportional)
        samples_per_cuisine = {}
        total_assigned = 0

        for cuisine, count in cuisine_counts.items():
            if cuisine == 'unknown':
                continue
            proportion = count / total_recipes
            n_samples = max(1, int(sample_size * proportion))  # At least 1 sample per cuisine
            samples_per_cuisine[cuisine] = n_samples
            total_assigned += n_samples

        # Adjust if we're over/under the target sample_size
        if total_assigned != sample_size:
            # Distribute remaining samples to largest cuisines
            remaining = sample_size - total_assigned
            largest_cuisines = sorted(samples_per_cuisine.keys(),
                                      key=lambda x: cuisine_counts[x], reverse=True)

            for cuisine in largest_cuisines:
                if remaining == 0:
                    break
                if remaining > 0:
                    samples_per_cuisine[cuisine] += 1
                    remaining -= 1
                else:
                    if samples_per_cuisine[cuisine] > 1:  # Don't go below 1
                        samples_per_cuisine[cuisine] -= 1
                        remaining += 1

        # Sample from each cuisine
        selected_indices = []

        for cuisine, n_samples in samples_per_cuisine.items():
            # Get all indices for this cuisine
            cuisine_indices = [i for i, c in enumerate(self.cuisine_labels) if c == cuisine]

            if len(cuisine_indices) <= n_samples:
                # Take all if we need more than available
                selected_indices.extend(cuisine_indices)
            else:
                # Random sample from this cuisine
                sampled = np.random.choice(cuisine_indices, n_samples, replace=False)
                selected_indices.extend(sampled)

        print(f"Stratified sampling: {len(selected_indices)} recipes from {len(samples_per_cuisine)} cuisines")
        print("Sample distribution:")
        sample_cuisine_counts = Counter([self.cuisine_labels[i] for i in selected_indices])
        for cuisine, count in sample_cuisine_counts.most_common(5):
            original_count = cuisine_counts[cuisine]
            percentage = (count / original_count) * 100
            print(f"  {cuisine}: {count}/{original_count} ({percentage:.1f}%)")

        return selected_indices

    def select_itemsets_by_strategy(self, frequent_itemsets, strategy='coverage',
                                    coverage_threshold=0.8, top_k=500):
        """
        Select itemsets using different strategies

        Parameters:
        strategy: 'coverage', 'top_support', 'balanced'
        """
        if frequent_itemsets is None:
            return []

        if strategy == 'coverage':
            # Your existing cumulative coverage approach
            sorted_itemsets = sorted(frequent_itemsets.items(),
                                     key=lambda x: x[1]['support'], reverse=True)
            cumulative_coverage = 0
            selected = []

            for item, data in sorted_itemsets:
                selected.append(data['items'])
                cumulative_coverage += data['support']
                if cumulative_coverage >= coverage_threshold:
                    break
            return selected

        elif strategy == 'top_support':
            # Simply take top K by support
            sorted_itemsets = sorted(frequent_itemsets.items(),
                                     key=lambda x: x[1]['support'], reverse=True)
            return [data['items'] for item, data in sorted_itemsets[:top_k]]

        elif strategy == 'balanced':
            # Mix of high and medium support itemsets
            sorted_itemsets = sorted(frequent_itemsets.items(),
                                     key=lambda x: x[1]['support'], reverse=True)

            # Take top 50% of top_k from high support
            high_support_count = int(top_k * 0.5)
            selected = [data['items'] for item, data in sorted_itemsets[:high_support_count]]

            # Fill remaining with medium support items
            remaining_count = top_k - high_support_count
            medium_start = high_support_count
            medium_end = min(medium_start + remaining_count, len(sorted_itemsets))

            selected.extend([data['items'] for item, data in
                             sorted_itemsets[medium_start:medium_end]])
            return selected

    def run_single_experiment(self, itemset_size, strategy='coverage', sample_size=5000):
        """Run clustering experiment for single itemset size"""
        print(f"\n{'=' * 60}")
        print(f"Running experiment: {itemset_size}-itemsets, strategy: {strategy}")
        print(f"{'=' * 60}")

        # Load frequent itemsets
        frequent_itemsets = self.load_frequent_itemsets(itemset_size)
        if frequent_itemsets is None:
            return None

        # Select itemsets
        selected_itemsets = self.select_itemsets_by_strategy(
            frequent_itemsets, strategy=strategy, top_k=200  # Increased from 500
        )

        if not selected_itemsets:
            print(f"No itemsets selected for size {itemset_size}")
            return None

        print(f"Selected {len(selected_itemsets)} frequent {itemset_size}-itemsets")

        # # Sample transactions if needed
        # if sample_size and len(self.transactions) > sample_size:
        #     sample_indices = np.random.choice(len(self.transactions), sample_size, replace=False)
        #     sample_transactions = [self.transactions[i] for i in sample_indices]
        #     sample_cuisine_labels = [self.cuisine_labels[i] for i in sample_indices]
        # else:
        #     sample_transactions = self.transactions
        #     sample_cuisine_labels = self.cuisine_labels
        #     sample_indices = np.arange(len(self.transactions))
        # Sample transactions if needed
        if sample_size and len(self.transactions) > sample_size:
            sample_indices = self.stratified_sample(sample_size)
            sample_transactions = [self.transactions[i] for i in sample_indices]
            sample_cuisine_labels = [self.cuisine_labels[i] for i in sample_indices]
        else:
            sample_transactions = self.transactions
            sample_cuisine_labels = self.cuisine_labels
            sample_indices = np.arange(len(self.transactions))

        # Initialize and run clustering
        clustering = FrequentItemsetClustering(selected_itemsets, sample_transactions)
        feature_matrix = clustering.create_feature_matrix()
        clustering.perform_clustering(linkage_method='average', distance_metric='jaccard')

        # Try different numbers of clusters
        cluster_results = {}
        # for n_clusters in [5, 10, 15, 20, 25]:
        # for n_clusters in [20, 25]:
        for n_clusters in [10]:
            try:
                clusters = clustering.get_clusters(n_clusters=n_clusters)

                # Evaluate against cuisine labels
                metrics = self.evaluate_clustering_vs_cuisine(
                    clusters, sample_cuisine_labels, n_clusters
                )

                cluster_results[n_clusters] = {
                    'clusters': clusters,
                    'metrics': metrics,
                    'clustering_obj': clustering
                }

                print(f"  {n_clusters} clusters: ARI={metrics['ari']:.3f}, "
                      f"NMI={metrics['nmi']:.3f}, Purity={metrics['purity']:.3f}")

            except Exception as e:
                print(f"  Error with {n_clusters} clusters: {e}")

        return {
            'itemset_size': itemset_size,
            'strategy': strategy,
            'sample_size': len(sample_transactions),
            'sample_transactions': sample_transactions,
            'sample_indices': sample_indices,
            'selected_itemsets': selected_itemsets,
            'feature_matrix': feature_matrix,
            'cluster_results': cluster_results,
            'cuisine_labels': sample_cuisine_labels
        }

    def evaluate_clustering_vs_cuisine(self, clusters, cuisine_labels, n_clusters):
        """Evaluate how well clusters match cuisine labels"""

        # Remove any 'unknown' cuisine labels for evaluation
        valid_indices = [i for i, cuisine in enumerate(cuisine_labels) if cuisine != 'unknown']
        if not valid_indices:
            return {'error': 'No valid cuisine labels'}

        valid_clusters = [clusters[i] for i in valid_indices]
        valid_cuisines = [cuisine_labels[i] for i in valid_indices]

        # Convert cuisine labels to numeric
        unique_cuisines = list(set(valid_cuisines))
        cuisine_to_num = {cuisine: i for i, cuisine in enumerate(unique_cuisines)}
        numeric_cuisines = [cuisine_to_num[cuisine] for cuisine in valid_cuisines]

        # Calculate metrics
        ari = adjusted_rand_score(numeric_cuisines, valid_clusters)
        nmi = normalized_mutual_info_score(numeric_cuisines, valid_clusters)

        # Calculate cluster purity
        cluster_purities = []
        cluster_cuisine_dist = {}

        for cluster_id in set(valid_clusters):
            cluster_indices = [i for i, c in enumerate(valid_clusters) if c == cluster_id]
            cluster_cuisines = [valid_cuisines[i] for i in cluster_indices]
            cuisine_counts = Counter(cluster_cuisines)

            # Purity = most common cuisine / cluster size
            most_common_count = cuisine_counts.most_common(1)[0][1]
            purity = most_common_count / len(cluster_cuisines)
            cluster_purities.append(purity)

            cluster_cuisine_dist[cluster_id] = cuisine_counts

        avg_purity = np.mean(cluster_purities)

        return {
            'ari': ari,
            'nmi': nmi,
            'purity': avg_purity,
            'n_clusters': n_clusters,
            'cluster_cuisine_distribution': cluster_cuisine_dist,
            'unique_cuisines': unique_cuisines
        }

    def run_comprehensive_analysis(self, sample_size=None):
        """Run analysis across all itemset sizes and strategies"""
        print("Starting comprehensive clustering analysis...")
        self.load_data()

        # Set random seed for reproducibility
        np.random.seed(42)

        strategies = ['top_support']
        # strategies = ['coverage', 'top_support']
        # itemset_sizes = [2, 3, 4, 5, 6]
        itemset_sizes = [3]

        for strategy in strategies:
            self.results[strategy] = {}
            for itemset_size in itemset_sizes:
                result = self.run_single_experiment(
                    itemset_size, strategy=strategy, sample_size=sample_size
                )
                if result:
                    self.results[strategy][itemset_size] = result

    def create_cluster_scatter_plot(self, experiment_result, n_clusters=20, method='tsne'):
        """Create 2D scatter plot of clusters using dimensionality reduction"""

        if n_clusters not in experiment_result['cluster_results']:
            print(f"No results for {n_clusters} clusters")
            return None

        feature_matrix = experiment_result['feature_matrix']
        clusters = experiment_result['cluster_results'][n_clusters]['clusters']
        cuisine_labels = experiment_result['cuisine_labels']

        # Dimensionality reduction
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(feature_matrix) // 4))
        else:  # PCA
            reducer = PCA(n_components=2, random_state=42)

        # Standardize features for better reduction
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(feature_matrix.astype(float))
        coords_2d = reducer.fit_transform(scaled_features)

        # Create DataFrame for plotting
        plot_df = pd.DataFrame({
            'x': coords_2d[:, 0],
            'y': coords_2d[:, 1],
            'cluster': clusters,
            'cuisine': cuisine_labels,
            'index': range(len(clusters))
        })

        # Create plotly scatter plot
        fig = px.scatter(
            plot_df, x='x', y='y',
            color='cluster',
            hover_data=['cuisine', 'index'],
            title=f"Cluster Visualization ({method.upper()}) - {experiment_result['itemset_size']}-itemsets",
            color_continuous_scale='viridis'
        )

        fig.update_layout(
            width=800, height=600,
            xaxis_title=f'{method.upper()} Component 1',
            yaxis_title=f'{method.upper()} Component 2'
        )

        return fig

    def create_cuisine_cluster_comparison(self, experiment_result, n_clusters=20):
        """Create confusion matrix comparing algorithmic clusters to cuisines"""

        if n_clusters not in experiment_result['cluster_results']:
            return None

        clusters = experiment_result['cluster_results'][n_clusters]['clusters']
        cuisine_labels = experiment_result['cuisine_labels']

        # Create confusion matrix
        unique_cuisines = sorted(list(set(cuisine_labels)))
        unique_clusters = sorted(list(set(clusters)))

        confusion_matrix = np.zeros((len(unique_clusters), len(unique_cuisines)))

        for i, cluster_id in enumerate(unique_clusters):
            cluster_indices = [j for j, c in enumerate(clusters) if c == cluster_id]
            cluster_cuisines = [cuisine_labels[j] for j in cluster_indices]
            cuisine_counts = Counter(cluster_cuisines)

            for j, cuisine in enumerate(unique_cuisines):
                confusion_matrix[i, j] = cuisine_counts.get(cuisine, 0)

        # Normalize by cluster size
        normalized_matrix = confusion_matrix / confusion_matrix.sum(axis=1, keepdims=True)

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(
            normalized_matrix,
            xticklabels=unique_cuisines,
            yticklabels=[f'Cluster {c}' for c in unique_clusters],
            annot=False,
            cmap='YlOrRd',
            ax=ax
        )

        plt.title(
            f'Cluster-Cuisine Confusion Matrix\n{experiment_result["itemset_size"]}-itemsets, {n_clusters} clusters')
        plt.xlabel('True Cuisine')
        plt.ylabel('Algorithmic Cluster')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()

        return fig

    def create_metrics_comparison_plot(self):
        """Create comparison plot of metrics across all experiments"""

        # Collect all metrics
        metrics_data = []

        for strategy in self.results:
            for itemset_size in self.results[strategy]:
                experiment = self.results[strategy][itemset_size]
                for n_clusters in experiment['cluster_results']:
                    metrics = experiment['cluster_results'][n_clusters]['metrics']
                    if 'error' not in metrics:
                        metrics_data.append({
                            'strategy': strategy,
                            'itemset_size': itemset_size,
                            'n_clusters': n_clusters,
                            'ari': metrics['ari'],
                            'nmi': metrics['nmi'],
                            'purity': metrics['purity']
                        })

        df = pd.DataFrame(metrics_data)

        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        metrics = ['ari', 'nmi', 'purity']
        titles = ['Adjusted Rand Index', 'Normalized Mutual Information', 'Cluster Purity']

        for i, (metric, title) in enumerate(zip(metrics, titles)):
            pivot_data = df.pivot_table(
                values=metric,
                index=['itemset_size'],
                columns=['strategy', 'n_clusters'],
                aggfunc='mean'
            )

            sns.heatmap(
                pivot_data,
                annot=True,
                fmt='.3f',
                cmap='RdYlBu_r',
                ax=axes[i],
                cbar_kws={'label': metric.upper()}
            )

            axes[i].set_title(title)
            axes[i].set_xlabel('Strategy & Number of Clusters')
            axes[i].set_ylabel('Itemset Size')

        plt.tight_layout()
        return fig

    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""

        print("\n" + "=" * 80)
        print("COMPREHENSIVE CLUSTERING ANALYSIS REPORT")
        print("=" * 80)

        # Overall summary
        total_experiments = sum(len(self.results[strategy]) for strategy in self.results)
        print(f"\nTotal experiments conducted: {total_experiments}")
        print(f"Strategies tested: {list(self.results.keys())}")

        # Find best performing combinations
        best_results = []

        for strategy in self.results:
            for itemset_size in self.results[strategy]:
                experiment = self.results[strategy][itemset_size]
                for n_clusters in experiment['cluster_results']:
                    metrics = experiment['cluster_results'][n_clusters]['metrics']
                    if 'error' not in metrics:
                        best_results.append({
                            'strategy': strategy,
                            'itemset_size': itemset_size,
                            'n_clusters': n_clusters,
                            'ari': metrics['ari'],
                            'nmi': metrics['nmi'],
                            'purity': metrics['purity'],
                            'combined_score': (metrics['ari'] + metrics['nmi'] + metrics['purity']) / 3
                        })

        # Sort by combined score
        best_results.sort(key=lambda x: x['combined_score'], reverse=True)

        print(f"\nTOP 10 PERFORMING CONFIGURATIONS:")
        print("-" * 50)
        for i, result in enumerate(best_results[:10]):
            print(f"{i + 1:2d}. {result['itemset_size']}-itemsets, {result['strategy']}, "
                  f"{result['n_clusters']} clusters: "
                  f"ARI={result['ari']:.3f}, NMI={result['nmi']:.3f}, "
                  f"Purity={result['purity']:.3f} (Combined: {result['combined_score']:.3f})")

        # Analysis by itemset size
        print(f"\nPERFORMANCE BY ITEMSET SIZE:")
        print("-" * 30)

        size_performance = {}
        for result in best_results:
            size = result['itemset_size']
            if size not in size_performance:
                size_performance[size] = []
            size_performance[size].append(result['combined_score'])

        for size in sorted(size_performance.keys()):
            avg_score = np.mean(size_performance[size])
            max_score = np.max(size_performance[size])
            print(f"{size}-itemsets: Avg={avg_score:.3f}, Max={max_score:.3f}")

        return best_results

def main():
    """Main execution function"""

    # Initialize analysis
    analyzer = ComprehensiveClusteringAnalysis(
        data_path="ingredient_matching_adaptive_results.json",
        cuisine_data_path="whats-cooking/train.json"
    )

    # Run comprehensive analysis
    analyzer.run_comprehensive_analysis(sample_size=5000)

    # Generate report
    best_results = analyzer.generate_comprehensive_report()

    # Create visualizations for best performing configuration
    best_config = best_results[0]
    best_experiment = analyzer.results[best_config['strategy']][best_config['itemset_size']]

    print(f"\nCreating visualizations for best configuration:")
    print(f"Strategy: {best_config['strategy']}")
    print(f"Itemset size: {best_config['itemset_size']}")
    print(f"Number of clusters: {best_config['n_clusters']}")

    # Create scatter plot
    scatter_fig = analyzer.create_cluster_scatter_plot(
        best_experiment, n_clusters=best_config['n_clusters']
    )
    if scatter_fig:
        scatter_fig.show()

    # Create confusion matrix
    confusion_fig = analyzer.create_cuisine_cluster_comparison(
        best_experiment, n_clusters=best_config['n_clusters']
    )
    if confusion_fig:
        plt.show()

    # Create metrics comparison
    #metrics_fig = analyzer.create_metrics_comparison_plot()
    plt.show()

    print("\nAnalysis complete!")

    return analyzer


if __name__ == "__main__":
    analyzer = main()
