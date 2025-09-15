import cluster
import numpy as np
import mapping
import shelve

def recommend_recipe(ingredients, experiment_result, n_clusters = 10):
    """
    Recommend a recipe given input ingredients.

    ingredients: set of ingredients (e.g. {"tomato", "garlic"})
    experiment_result: output dict from run_single_experiment
    n_clusters: number of clusters used in that experiment
    """
    search = mapping.Search('recipe_cuisine.db')
    ingredients = set(mapping.resolve_words(list(ingredients)))
    feature_matrix = experiment_result['feature_matrix']
    clusters = experiment_result['cluster_results'][n_clusters]['clusters']
    recipes = experiment_result['sample_transactions']
    itemsets = experiment_result['selected_itemsets']

    # Step 1: Build query vector
    query_vector = np.array([
        1 if set(itemset).issubset(ingredients) else 0
        for itemset in itemsets
    ])

    # Step 2: Score clusters by average similarity
    cluster_scores = {}
    for cluster_id in set(clusters):
        cluster_indices = [i for i, c in enumerate(clusters) if c == cluster_id]
        cluster_vectors = feature_matrix[cluster_indices]
        sims = []
        for vec in cluster_vectors:
            inter = np.sum(query_vector * vec)
            union = np.sum((query_vector + vec) > 0)
            sims.append(inter / union if union > 0 else 0)
        cluster_scores[cluster_id] = np.mean(sims) if sims else 0

    # Step 3: Pick best cluster
    best_cluster = max(cluster_scores, key=cluster_scores.get)

    # Step 4: From that cluster, select recipe with max ingredient overlap
    candidates = [i for i, c in enumerate(clusters) if c == best_cluster]
    best_recipe_idx = max(candidates, key=lambda i: len(ingredients & set(recipes[i])))

    return {
        "recipe_index": best_recipe_idx,
        "recipe": recipes[best_recipe_idx],
        "cluster": best_cluster
    }

def main(itemset_size: int):
    with shelve.open('run_single_experiment') as db:
        experiment_result = db.get(str(itemset_size), None)

    if experiment_result is None:
        analyzer = cluster.ComprehensiveClusteringAnalysis(
            data_path="train_lines.json",
            cuisine_data_path="whats-cooking/train.json"
        )

        analyzer.load_data()

        experiment_result = analyzer.run_single_experiment(itemset_size)
        with shelve.open('run_single_experiment') as db:
            db[str(itemset_size)] = experiment_result

    rec = recommend_recipe(
        {"onion", "mayonaise"},  # input ingredients
        experiment_result          # pick the same n_clusters you used in clustering
    )

    print(rec)

if __name__ == '__main__':
    main(2)
