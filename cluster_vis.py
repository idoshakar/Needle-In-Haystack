import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter, defaultdict
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
from pathlib import Path
import os

warnings.filterwarnings('ignore')


class ClusterCuisineVisualizer:
    """
    Visualization class for comparing clustering results with cuisine labels
    """

    def __init__(self, cluster_assignments_path, original_data_path=None):
        self.cluster_assignments_path = cluster_assignments_path
        self.original_data_path = original_data_path
        self.cluster_data = None
        self.df = None

        self.output_dir = Path(cluster_assignments_path).parent / "visualizations"
        self.output_dir.mkdir(exist_ok=True)
        print(f"Plots will be saved to: {self.output_dir}")

        self.load_data()

    def load_data(self):
        """Load and prepare data for visualization"""
        with open(self.cluster_assignments_path, 'r') as f:
            self.cluster_data = json.load(f)

        records = []
        for cluster_id, cluster_info in self.cluster_data.items():
            for i, recipe_idx in enumerate(cluster_info['recipe_indices']):
                records.append({
                    'recipe_index': recipe_idx,
                    'cluster_id': cluster_id,
                    'cluster_size': cluster_info['size'],
                    'cuisine': cluster_info['cuisines'][i]
                })

        self.df = pd.DataFrame(records)
        print(f"Loaded {len(self.df)} recipes across {len(self.cluster_data)} clusters")

    def create_confusion_matrix_heatmap(self, top_n_clusters=20, top_n_cuisines=15):
        """Create a confusion matrix heatmap showing cluster vs cuisine alignment"""

        top_clusters = self.df['cluster_id'].value_counts().head(top_n_clusters).index
        top_cuisines = self.df['cuisine'].value_counts().head(top_n_cuisines).index

        filtered_df = self.df[
            (self.df['cluster_id'].isin(top_clusters)) &
            (self.df['cuisine'].isin(top_cuisines))
            ]

        confusion_matrix = pd.crosstab(
            filtered_df['cluster_id'],
            filtered_df['cuisine'],
            normalize='index'
        )


        plt.figure(figsize=(16, 12))
        sns.heatmap(
            confusion_matrix,
            annot=True,
            fmt='.2f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Proportion of Cluster'}
        )
        plt.title(
            f'Cluster vs Cuisine Confusion Matrix\n(Top {top_n_clusters} clusters, Top {top_n_cuisines} cuisines)')
        plt.xlabel('Cuisine')
        plt.ylabel('Cluster ID')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        output_path = self.output_dir / "confusion_matrix_heatmap.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved confusion matrix heatmap to: {output_path}")

        plt.show()

        return confusion_matrix

    def create_cluster_purity_analysis(self):
        """Analyze and visualize cluster purity scores"""

        cluster_stats = []
        for cluster_id, cluster_info in self.cluster_data.items():
            cuisines = cluster_info['cuisines']
            cuisine_counts = Counter(cuisines)

            most_common_count = cuisine_counts.most_common(1)[0][1]
            purity = most_common_count / len(cuisines)

            total = len(cuisines)
            entropy = -sum((count / total) * np.log2(count / total) for count in cuisine_counts.values())

            cluster_stats.append({
                'cluster_id': cluster_id,
                'size': cluster_info['size'],
                'purity': purity,
                'entropy': entropy,
                'dominant_cuisine': cuisine_counts.most_common(1)[0][0],
                'n_cuisines': len(cuisine_counts)
            })

        stats_df = pd.DataFrame(cluster_stats)

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        axes[0, 0].hist(stats_df['purity'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0, 0].axvline(stats_df['purity'].mean(), color='red', linestyle='--',
                           label=f'Mean: {stats_df["purity"].mean():.3f}')
        axes[0, 0].set_xlabel('Cluster Purity')
        axes[0, 0].set_ylabel('Number of Clusters')
        axes[0, 0].set_title('Distribution of Cluster Purity Scores')
        axes[0, 0].legend()

        scatter = axes[0, 1].scatter(stats_df['size'], stats_df['purity'],
                                     alpha=0.6, c=stats_df['entropy'], cmap='viridis')
        axes[0, 1].set_xlabel('Cluster Size (# recipes)')
        axes[0, 1].set_ylabel('Cluster Purity')
        axes[0, 1].set_title('Cluster Size vs Purity (colored by entropy)')
        plt.colorbar(scatter, ax=axes[0, 1], label='Entropy')

        axes[1, 0].hist(stats_df['n_cuisines'], bins=range(1, max(stats_df['n_cuisines']) + 2),
                        alpha=0.7, color='lightcoral', edgecolor='black')
        axes[1, 0].set_xlabel('Number of Different Cuisines in Cluster')
        axes[1, 0].set_ylabel('Number of Clusters')
        axes[1, 0].set_title('Cuisine Diversity per Cluster')

        dominant_cuisine_counts = Counter(stats_df['dominant_cuisine'])
        top_cuisines = dominant_cuisine_counts.most_common(10)

        cuisines, counts = zip(*top_cuisines)
        axes[1, 1].bar(range(len(cuisines)), counts, color='lightgreen', alpha=0.7)
        axes[1, 1].set_xticks(range(len(cuisines)))
        axes[1, 1].set_xticklabels(cuisines, rotation=45, ha='right')
        axes[1, 1].set_ylabel('Number of Clusters Dominated')
        axes[1, 1].set_title('Most Frequent Dominant Cuisines')

        plt.tight_layout()

        output_path = self.output_dir / "cluster_purity_analysis.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved cluster purity analysis to: {output_path}")

        plt.show()

        return stats_df

    def create_interactive_cluster_cuisine_sunburst(self):
        """Create an interactive sunburst chart showing cluster-cuisine hierarchy"""

        sunburst_data = []

        for cluster_id, cluster_info in self.cluster_data.items():
            cluster_size = cluster_info['size']
            cuisine_counts = Counter(cluster_info['cuisines'])

            sunburst_data.append({
                'ids': cluster_id,
                'labels': f"{cluster_id}\n({cluster_size} recipes)",
                'parents': '',
                'values': cluster_size
            })

            for cuisine, count in cuisine_counts.items():
                if count >= 10:
                    sunburst_data.append({
                        'ids': f"{cluster_id}_{cuisine}",
                        'labels': f"{cuisine}\n({count})",
                        'parents': cluster_id,
                        'values': count
                    })

        df_sunburst = pd.DataFrame(sunburst_data)

        fig = go.Figure(go.Sunburst(
            ids=df_sunburst['ids'],
            labels=df_sunburst['labels'],
            parents=df_sunburst['parents'],
            values=df_sunburst['values'],
            branchvalues="total",
            maxdepth=2,
        ))

        fig.update_layout(
            title="Cluster-Cuisine Hierarchy (Interactive Sunburst)",
            height=800,
            width=800
        )

        output_path = self.output_dir / "sunburst_cluster_cuisine.html"
        fig.write_html(output_path)
        print(f"Saved interactive sunburst chart to: {output_path}")

        fig.show()

        return fig

    def create_cluster_size_distribution(self):
        """Visualize cluster size distribution and compare with cuisine distribution"""

        cluster_sizes = [info['size'] for info in self.cluster_data.values()]

        all_cuisines = []
        for cluster_info in self.cluster_data.values():
            all_cuisines.extend(cluster_info['cuisines'])
        cuisine_counts = Counter(all_cuisines)

        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        axes[0].hist(cluster_sizes, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0].axvline(np.mean(cluster_sizes), color='red', linestyle='--',
                        label=f'Mean: {np.mean(cluster_sizes):.0f}')
        axes[0].axvline(np.median(cluster_sizes), color='orange', linestyle='--',
                        label=f'Median: {np.median(cluster_sizes):.0f}')
        axes[0].set_xlabel('Cluster Size (# recipes)')
        axes[0].set_ylabel('Number of Clusters')
        axes[0].set_title('Distribution of Cluster Sizes')
        axes[0].legend()

        top_cuisines = cuisine_counts.most_common(15)
        cuisines, counts = zip(*top_cuisines)

        bars = axes[1].bar(range(len(cuisines)), counts, color='lightcoral', alpha=0.7)
        axes[1].set_xticks(range(len(cuisines)))
        axes[1].set_xticklabels(cuisines, rotation=45, ha='right')
        axes[1].set_ylabel('Number of Recipes')
        axes[1].set_title('Recipe Distribution by Cuisine (Top 15)')

        for bar, count in zip(bars, counts):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                         f'{count}', ha='center', va='bottom')

        plt.tight_layout()

        output_path = self.output_dir / "size_distribution_comparison.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved size distribution comparison to: {output_path}")

        plt.show()

        return cluster_sizes, cuisine_counts

    def create_cluster_cuisine_flow_diagram(self, min_flow=50):
        """Create a Sankey-style flow diagram showing recipe flow from cuisines to clusters"""

        cuisine_to_cluster = defaultdict(lambda: defaultdict(int))

        for cluster_id, cluster_info in self.cluster_data.items():
            for cuisine in cluster_info['cuisines']:
                cuisine_to_cluster[cuisine][cluster_id] += 1

        filtered_flows = []
        for cuisine, clusters in cuisine_to_cluster.items():
            for cluster_id, count in clusters.items():
                if count >= min_flow:
                    filtered_flows.append((cuisine, cluster_id, count))

        if not filtered_flows:
            print(f"No flows >= {min_flow} recipes found. Try lowering min_flow parameter.")
            return

        cuisines = sorted(set(flow[0] for flow in filtered_flows))
        clusters = sorted(set(flow[1] for flow in filtered_flows))

        node_labels = cuisines + clusters
        cuisine_indices = {cuisine: i for i, cuisine in enumerate(cuisines)}
        cluster_indices = {cluster: i + len(cuisines) for i, cluster in enumerate(clusters)}

        source_indices = []
        target_indices = []
        values = []

        for cuisine, cluster_id, count in filtered_flows:
            source_indices.append(cuisine_indices[cuisine])
            target_indices.append(cluster_indices[cluster_id])
            values.append(count)

        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=node_labels,
                color=["lightblue"] * len(cuisines) + ["lightcoral"] * len(clusters)
            ),
            link=dict(
                source=source_indices,
                target=target_indices,
                value=values
            )
        )])

        fig.update_layout(
            title_text=f"Recipe Flow: Cuisines → Clusters (min {min_flow} recipes per flow)",
            font_size=10,
            height=600,
            width=1000
        )

        output_path = self.output_dir / f"sankey_flow_diagram_min{min_flow}.html"
        fig.write_html(output_path)
        print(f"Saved Sankey flow diagram to: {output_path}")

        fig.show()

        return fig

    def save_summary_statistics(self, stats_df, cluster_sizes, cuisine_counts):
        """Save summary statistics to a text file"""

        output_path = self.output_dir / "analysis_summary.txt"

        with open(output_path, 'w') as f:
            f.write("Cluster-Cuisine Analysis Summary\n")
            f.write("=" * 40 + "\n\n")

            f.write(f"Total Recipes: {len(self.df)}\n")
            f.write(f"Total Clusters: {len(self.cluster_data)}\n")
            f.write(f"Total Cuisines: {self.df['cuisine'].nunique()}\n")
            f.write(f"Average Cluster Size: {self.df['cluster_size'].mean():.1f}\n\n")

            f.write("Cluster Purity Statistics:\n")
            f.write(f"  Mean Purity: {stats_df['purity'].mean():.3f}\n")
            f.write(f"  Median Purity: {stats_df['purity'].median():.3f}\n")
            f.write(f"  High Purity Clusters (>0.5): {(stats_df['purity'] > 0.5).sum()}\n")
            f.write(f"  Low Purity Clusters (<0.3): {(stats_df['purity'] < 0.3).sum()}\n\n")

            f.write("Top 10 Cuisines by Recipe Count:\n")
            for cuisine, count in cuisine_counts.most_common(10):
                f.write(f"  {cuisine}: {count} recipes\n")

            f.write(f"\nCluster Size Statistics:\n")
            f.write(f"  Mean: {np.mean(cluster_sizes):.1f}\n")
            f.write(f"  Median: {np.median(cluster_sizes):.1f}\n")
            f.write(f"  Largest cluster: {max(cluster_sizes)} recipes\n")
            f.write(f"  Smallest cluster: {min(cluster_sizes)} recipes\n")

        print(f"Saved summary statistics to: {output_path}")

    def create_comprehensive_dashboard(self):
        """Create a comprehensive analysis dashboard"""
        print("Creating Comprehensive Cluster-Cuisine Analysis Dashboard")
        print("=" * 60)

        print(f"Total Recipes: {len(self.df)}")
        print(f"Total Clusters: {len(self.cluster_data)}")
        print(f"Total Cuisines: {self.df['cuisine'].nunique()}")
        print(f"Average Cluster Size: {self.df['cluster_size'].mean():.1f}")
        print()

        print("1. Confusion Matrix Heatmap:")
        confusion_matrix = self.create_confusion_matrix_heatmap()

        print("\n2. Cluster Purity Analysis:")
        stats_df = self.create_cluster_purity_analysis()

        print("\n3. Size Distribution Analysis:")
        cluster_sizes, cuisine_counts = self.create_cluster_size_distribution()

        print("\n4. Interactive Cluster-Cuisine Hierarchy:")
        self.create_interactive_cluster_cuisine_sunburst()

        print("\n5. Cuisine-to-Cluster Flow Diagram:")
        self.create_cluster_cuisine_flow_diagram(min_flow=30)

        print("\n6. Saving Summary Statistics:")
        self.save_summary_statistics(stats_df, cluster_sizes, cuisine_counts)

        print("\n7. scatter plot:")
        self.create_cluster_scatter_plot()


        print("\n8. Summary Statistics:")
        print(f"   Mean Cluster Purity: {stats_df['purity'].mean():.3f}")
        print(f"   Median Cluster Purity: {stats_df['purity'].median():.3f}")
        print(f"   High Purity Clusters (>0.5): {(stats_df['purity'] > 0.5).sum()}")
        print(f"   Low Purity Clusters (<0.3): {(stats_df['purity'] < 0.3).sum()}")

        print(f"\nAll visualizations saved to: {self.output_dir}")

        return {
            'confusion_matrix': confusion_matrix,
            'cluster_stats': stats_df,
            'cluster_sizes': cluster_sizes,
            'cuisine_counts': cuisine_counts
        }

    def create_cluster_scatter_plot(self):
        """Create a simple scatter plot showing cluster size vs purity"""

        cluster_stats = []
        for cluster_id, cluster_info in self.cluster_data.items():
            cuisines = cluster_info['cuisines']
            cuisine_counts = Counter(cuisines)

            most_common_count = cuisine_counts.most_common(1)[0][1]
            purity = most_common_count / len(cuisines)

            cluster_stats.append({
                'cluster_id': cluster_id,
                'size': cluster_info['size'],
                'purity': purity,
                'dominant_cuisine': cuisine_counts.most_common(1)[0][0]
            })

        stats_df = pd.DataFrame(cluster_stats)

        plt.figure(figsize=(10, 6))
        scatter = plt.scatter(stats_df['size'], stats_df['purity'],
                              alpha=0.7, s=60, c='steelblue')

        plt.xlabel('Cluster Size (# recipes)')
        plt.ylabel('Cluster Purity')
        plt.title('Cluster Size vs Purity')
        plt.grid(True, alpha=0.3)

        output_path = self.output_dir / "cluster_scatter_plot.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved scatter plot to: {output_path}")

        plt.show()

        return stats_df

    def create_cluster_interaction_scatter(self, feature_matrix):
        """Create a scatter plot showing spatial relationships between cluster centroids"""

        cluster_centroids = {}
        for cluster_id, cluster_info in self.cluster_data.items():
            recipe_indices = cluster_info['recipe_indices']
            cluster_features = feature_matrix[recipe_indices]
            centroid = np.mean(cluster_features, axis=0)
            cluster_centroids[cluster_id] = centroid

        cluster_ids = list(cluster_centroids.keys())
        centroid_matrix = np.array([cluster_centroids[cid] for cid in cluster_ids])

        from sklearn.decomposition import PCA
        pca = PCA(n_components=2, random_state=42)
        centroids_2d = pca.fit_transform(centroid_matrix)

        cluster_sizes = [self.cluster_data[cid]['size'] for cid in cluster_ids]
        point_sizes = (np.array(cluster_sizes) / max(cluster_sizes) * 300 + 50)

        # Get dominant cuisine for coloring
        cluster_cuisines = []
        for cid in cluster_ids:
            cuisines = self.cluster_data[cid]['cuisines']
            dominant_cuisine = Counter(cuisines).most_common(1)[0][0]
            cluster_cuisines.append(dominant_cuisine)

        markers = ['o', 's', '^', 'D', 'v', '<', '>', 'p', '*', 'h', 'H', '+', 'x', '8']
        cluster_to_marker = {}
        for idx, cluster_id in enumerate(cluster_ids):
            cluster_to_marker[cluster_id] = markers[idx % len(markers)]

        plt.figure(figsize=(14, 8))

        unique_cuisines = list(set(cluster_cuisines))
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_cuisines)))
        cuisine_to_color = dict(zip(unique_cuisines, colors))

        for i, (x, y) in enumerate(centroids_2d):
            cuisine = cluster_cuisines[i]
            cluster_id = cluster_ids[i]
            marker = cluster_to_marker[cluster_id]

            plt.scatter(x, y, s=point_sizes[i], c=[cuisine_to_color[cuisine]],
                        marker=marker, alpha=0.7, edgecolors='black', linewidth=0.5)

        plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title(
            'Cluster Interactions in Feature Space\n(Point size = cluster size, Color = dominant cuisine, Shape = cluster)')
        plt.grid(True, alpha=0.3)

        top_cuisines = Counter(cluster_cuisines).most_common(10)
        cuisine_legend = [plt.scatter([], [], c=[cuisine_to_color[cuisine]],
                                      label=f'{cuisine} ({count})', s=60, marker='o')
                          for cuisine, count in top_cuisines]

        plt.legend(handles=cuisine_legend, title="Cuisine",
                   bbox_to_anchor=(1.05, 1), loc='upper left')

        plt.tight_layout()

        output_path = self.output_dir / "cluster_interaction_scatter.png"
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved cluster interaction scatter plot to: {output_path}")

        plt.show()

        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(centroids_2d))

        print("Closest cluster pairs:")
        for i in range(len(cluster_ids)):
            for j in range(i + 1, len(cluster_ids)):
                if distances[i, j] < np.percentile(distances, 10):  # Bottom 10% distances
                    print(f"  {cluster_ids[i]} ↔ {cluster_ids[j]}: distance = {distances[i, j]:.3f}")

        return centroids_2d, cluster_ids
def analyze_clustering_results(cluster_assignments_path):
    """Main function to run complete clustering analysis"""

    visualizer = ClusterCuisineVisualizer(cluster_assignments_path)
    results = visualizer.create_comprehensive_dashboard()

    return visualizer, results

# if __name__ == "__main__":
#     cluster_file = "pure_clusters_25/final clusters/cluster_assignments.json"
#
#     # Run analysis
#     visualizer, results = analyze_clustering_results(cluster_file)
#
#     print("\nAnalysis complete! Check the generated visualizations.")
#     print("Key insights:")
#     print(f"- {len(results['cluster_stats'])} clusters analyzed")
#     print(f"- Mean purity: {results['cluster_stats']['purity'].mean():.3f}")
#     print(f"- Clusters with >50% purity: {(results['cluster_stats']['purity'] > 0.5).sum()}")
