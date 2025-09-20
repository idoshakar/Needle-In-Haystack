import cluster
import numpy as np
from compare import *
import shelve

def recommend_recipe(ingredients, experiment_result, n_clusters = 10):
    """
    Recommend a recipe given input ingredients.

    ingredients: set of ingredients
    experiment_result: output dict from run_single_experiment
    n_clusters: number of clusters used in that experiment
    """

    matcher = HybridIngredientMatcher(
        foods_db_path=FINAL_FOOD_DATASET,
        foodb_path=FOOD_JSON
    )
    results = matcher.match_ingredients_batch(list(ingredients))
    ingredients = set([r.matched for r in results])
    feature_matrix = experiment_result['feature_matrix']
    clusters = experiment_result['cluster_results'][n_clusters]['clusters']
    recipes = experiment_result['sample_transactions']
    itemsets = experiment_result['selected_itemsets']

    query_vector = np.array([
        1 if set(itemset).issubset(ingredients) else 0
        for itemset in itemsets
    ])

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

    best_cluster = max(cluster_scores, key=cluster_scores.get)


    candidates = [i for i, c in enumerate(clusters) if c == best_cluster]
    best_recipe_idx = max(candidates, key=lambda i: len(ingredients & set(recipes[i])))
    indices = { i for i in candidates
        if len(ingredients & set(recipes[i])) == len(ingredients & set(recipes[best_recipe_idx])) }
    return {
        "recipe_indices": indices,
        "recipes": [recipes[i] for i in candidates if i in indices],
        "cluster": best_cluster
    }

def main(ingredients: set[str]) -> dict:
    itemset_size = len(ingredients)

    with shelve.open('run_single_experiment') as db:
        experiment_result = db.get(str(itemset_size), None)

    if experiment_result is None:
        analyzer = cluster.ComprehensiveClusteringAnalysis(
            data_path="ingredient_matching_adaptive_results.json",
            cuisine_data_path="whats-cooking/train.json"
        )

        analyzer.load_data()

        experiment_result = analyzer.run_single_experiment(itemset_size)
        with shelve.open('run_single_experiment') as db:
            db[str(itemset_size)] = experiment_result

    return recommend_recipe(
        ingredients,
        experiment_result
    )


if __name__ == '__main__':
    print(main({"onion", "mayonaise"}))
