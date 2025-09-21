
import os
import zipfile
import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from collections import Counter , defaultdict
from matplotlib.colors import PowerNorm
from wordcloud import WordCloud
from itertools import combinations
from pywaffle import Waffle
from matplotlib.cm import get_cmap
import matplotlib.patches as mpatches
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import colorsys
import itertools



def load_json_from_zip(zip_path, json_file_name):
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(json_file_name) as f:
            return json.load(f)

df = load_json_from_zip("needle.zip", "ingredient_matching_best_of_both_results.json") 
df = pd.DataFrame(df)
print(df.head())
print(df.columns)
print(df['cuisine'].unique())


df_foods = pd.read_json("Food.json", lines=True, encoding="utf-8")
print(df_foods.head())
print(df_foods.columns)


#-----------------------------1. Heat map of most common ingredients -------------------------------------------------------------
def compute_ingredient_percentages(df):
    """
    Returns a DataFrame:
    - index = cuisines
    - columns = ingredients
    - values = percentage of recipes in that cuisine containing that ingredient
    """
    cuisines = df['cuisine'].unique()
    all_ingredients = set()
    
    # First, collect all unique ingredients
    for ing_list in df['matched_ingredients']:
        all_ingredients.update(ing_list)
    
    data = []
    
    for cuisine in cuisines:
        subset = df[df['cuisine'] == cuisine]
        n_recipes = len(subset)
        
        # Count how many recipes contain each ingredient
        ingredient_counts = Counter()
        for ing_list in subset['matched_ingredients']:
            ingredient_counts.update(set(ing_list))  # count only once per recipe
        
        # Calculate percentage
        row = {ingredient: (ingredient_counts.get(ingredient, 0) / n_recipes * 100)
               for ingredient in all_ingredients}
        row['cuisine'] = cuisine
        data.append(row)
    
    df_percent = pd.DataFrame(data).set_index('cuisine')
    return df_percent

def plot_top_ingredients_heatmap(df_percent, top_n=20):
    # Sum across cuisines to find the most prevalent ingredients overall
    ingredient_totals = df_percent.sum().sort_values(ascending=False)
    top_ingredients = ingredient_totals.head(top_n).index
    
    # Select top ingredients for the heatmap
    heatmap_data = df_percent[top_ingredients]
    
    fig = plt.figure(figsize=(12,8))
    sns.heatmap(heatmap_data, annot=True, fmt=".1f", cmap="YlOrRd")
    plt.title(f"Top {top_n} Ingredients Across All Cuisines (%)", fontsize=16)
    plt.ylabel("Cuisine")
    plt.xlabel("Ingredient")
    plt.show()

    return fig

# Usage
#df_percent = compute_ingredient_percentages(df)
#plot_top_ingredients_heatmap(df_percent, top_n=20)

#--------------------------- 2. Avrage Number Of Ingredients Box Plot ----------------------------------------------------------


def plot_ingredient_count_boxplot_with_means_axis(df):
    """
    Boxplot with cuisine means in x-axis labels and overall mean as horizontal line,
    with fixed font sizes
    """
    # Count ingredients per recipe
    df['num_ingredients'] = df['matched_ingredients'].apply(len)
    
    # Compute mean per cuisine
    cuisine_means = df.groupby('cuisine')['num_ingredients'].mean()
    
    fig = plt.figure(figsize=(12,8))
    
    # Boxplot
    ax = sns.boxplot(x='cuisine', y='num_ingredients', data=df, palette="pastel")
    
    # Overall mean
    overall_mean = df['num_ingredients'].mean()
    plt.axhline(overall_mean, color='blue', linestyle='--', label=f'Overall Mean ({overall_mean:.2f})')
    
    # Update x-axis labels with mean
    new_labels = [f"{cuisine}\n(mean={cuisine_means[cuisine]:.1f})" for cuisine in cuisine_means.index]
    ax.set_xticklabels(new_labels, rotation=45, fontsize=6)  # fixed font size
    
    # Axis labels and title with fixed font size
    ax.set_xlabel("Cuisine", fontsize=14)
    ax.set_ylabel("Number of Ingredients", fontsize=14)
    ax.set_title("Number of Ingredients per Recipe by Cuisine", fontsize=16)
    
    plt.legend()
    plt.show()

    return fig

# Usage
#plot_ingredient_count_boxplot_with_means_axis(df)

#------------------ 3. Number of Unique Ingredients Per Cuisine Normilized by num of recepices ---------------------

def plot_normalized_unique_ingredients_per_cuisine(df):
    """
    Bar chart of normalized ingredient diversity per cuisine
    """
    cuisines = df['cuisine'].unique()
    normalized_diversity = {}
    
    for cuisine in cuisines:
        subset = df[df['cuisine'] == cuisine]
        n_recipes = len(subset)
        
        # Count in how many recipes each ingredient appears
        ingredient_counts = Counter()
        for ing_list in subset['matched_ingredients']:
            ingredient_counts.update(set(ing_list))  # unique per recipe
        
        # Normalize by number of recipes
        percentages = [count / n_recipes for count in ingredient_counts.values()]
        
        # Sum of normalized percentages = average proportion of unique ingredients per recipe
        normalized_diversity[cuisine] = sum(percentages)
    
    # Convert to Series for plotting
    normalized_diversity_series = pd.Series(normalized_diversity).sort_values(ascending=False)
    
    # Plot
    fig = plt.figure(figsize=(12,8))
    normalized_diversity_series.plot(kind='bar', color='skyblue')
    plt.title("Normalized Ingredient Diversity per Cuisine", fontsize=16)
    plt.xlabel("Cuisine", fontsize=14)
    plt.ylabel("Normalized Ingredient Diversity", fontsize=14)
    plt.xticks(rotation=45)
    plt.show()

    return fig

# Usage
#plot_normalized_unique_ingredients_per_cuisine(df)

#---------------------- 4. Geographical Locaiton Of Cuisines Cosine Similarity ------------------------------------------

# ------------------------------
# 1️. Compute ingredient vectors per cuisine
# ------------------------------
def get_cuisine_ingredient_matrix(df):
    cuisines = df['cuisine'].unique()
    all_ingredients = set(ing for lst in df['matched_ingredients'] for ing in lst)
    
    data = []
    for cuisine in cuisines:
        subset = df[df['cuisine'] == cuisine]
        n_recipes = len(subset)
        ingredient_counts = Counter()
        for ing_list in subset['matched_ingredients']:
            ingredient_counts.update(set(ing_list))  # count once per recipe
        row = {ing: ingredient_counts.get(ing,0)/n_recipes for ing in all_ingredients}
        row['cuisine'] = cuisine
        data.append(row)
    
    df_matrix = pd.DataFrame(data).set_index('cuisine')
    return df_matrix

# ------------------------------
# 2️. Compute cosine similarity between cuisines
# ------------------------------
def compute_cuisine_cosine_similarity(df_matrix):
    similarity_matrix = cosine_similarity(df_matrix.values)
    similarity_df = pd.DataFrame(similarity_matrix, index=df_matrix.index, columns=df_matrix.index)
    return similarity_df

# ------------------------------
# 3️. Plot heatmap with spacing, colormap, and thresholding
# ------------------------------
def plot_cuisine_similarity_heatmap(df):
    # Geographical order
    geo_order = [
        'irish', 'british', 'french', 'italian', 'spanish', 'greek', 'moroccan', 'russian',
        'southern_us', 'cajun_creole', 'jamaican', 'brazilian', 'mexican',
        'indian', 'thai', 'vietnamese', 'chinese', 'japanese', 'korean', 'filipino'
    ]
    
    # Compute ingredient matrix and similarity
    df_matrix = get_cuisine_ingredient_matrix(df)
    similarity_df = compute_cuisine_cosine_similarity(df_matrix)
    
    # Filter geo_order to cuisines present in data
    geo_order_filtered = [c for c in geo_order if c in similarity_df.index]
    sim_ordered = similarity_df.loc[geo_order_filtered, geo_order_filtered]
    
    # Mask low similarity values (<0.3) to appear lighter
    mask = sim_ordered < 0.4
    
    fig = plt.figure(figsize=(14,12))
    sns.heatmap(sim_ordered,
                annot=False,
                fmt=".2f",
                cmap="YlOrRd",
                linewidths=1,        # space between cells
                linecolor='white',
                mask=mask,
                cbar_kws={'label': 'Cosine Similarity'})
    
    plt.title("Cosine Similarity Between Cuisines (geographically ordered)", fontsize=16)
    plt.xlabel("Cuisine", fontsize=14)
    plt.ylabel("Cuisine", fontsize=14)
    plt.show()

    return fig

# ------------------------------
# 4️. Usage
# ------------------------------
#plot_cuisine_similarity_heatmap(df)


#--------------------------- 4.5 Most Similar Cuisines Pair / Triplets ------------------------------------------------------

def get_cuisine_ingredient_matrix(df):
    """
    Create a binary matrix [cuisines x ingredients].
    1 if ingredient appears in cuisine, else 0.
    """
    all_ings = sorted({ing for row in df["matched_ingredients"] for ing in row})
    cuisines = df["cuisine"].unique()

    data = []
    for cuisine in cuisines:
        cuisine_df = df[df["cuisine"] == cuisine]
        ing_set = {ing for row in cuisine_df["matched_ingredients"] for ing in row}
        row = [1 if ing in ing_set else 0 for ing in all_ings]
        data.append(row)

    return pd.DataFrame(data, index=cuisines, columns=all_ings)




# -----------------------------
# Step 3: Most similar cuisines (pairs + triplets)
# -----------------------------
def most_similar_cuisines(similarity_df, top_n_pairs=5, top_n_triplets=5):
    cuisines = similarity_df.index.tolist()
    
    # Pairs
    pairs = list(itertools.combinations(cuisines, 2))
    pair_sims = [(a, b, similarity_df.loc[a, b]) for a, b in pairs]
    pair_sims_sorted = sorted(pair_sims, key=lambda x: x[2], reverse=True)

    text = []
    text.append(f"Top {top_n_pairs} most similar cuisine pairs:\n")
    for a, b, sim in pair_sims_sorted[:top_n_pairs]:
        text.append(f"{a} - {b}: {sim:.3f}")
    
    # Triplets
    triplets = list(itertools.combinations(cuisines, 3))
    triplet_sims = []
    for a, b, c in triplets:
        avg_sim = (similarity_df.loc[a, b] + similarity_df.loc[a, c] + similarity_df.loc[b, c]) / 3
        triplet_sims.append((a, b, c, avg_sim))
    
    triplet_sims_sorted = sorted(triplet_sims, key=lambda x: x[3], reverse=True)
    
    text.append(f"\nTop {top_n_triplets} most similar cuisine triplets:\n")
    for a, b, c, sim in triplet_sims_sorted[:top_n_triplets]:
        text.append(f"{a} - {b} - {c}: {sim:.3f}")

    return "\n".join(text)


# -----------------------------
# Step 4: Top similar cuisines per cuisine
# -----------------------------
def top_similar_cuisines_per_cuisine(similarity_df, top_n=3):
    cuisines = similarity_df.index.tolist()
    text = ["\nTop similar cuisines per cuisine:\n"]

    for cuisine in cuisines:
        sims = similarity_df.loc[cuisine].drop(cuisine)
        top_cuisines = sims.sort_values(ascending=False).head(top_n)
        text.append(f"{cuisine}:")
        for other_cuisine, sim in top_cuisines.items():
            text.append(f"  {other_cuisine}: {sim:.3f}")
        text.append("")

    return "\n".join(text)

#------------------------5. Most Common Food Groups Among All Cuisines ----------------------------------------------------------


def cuisine_food_group_stacked_bar(df, df_foods, min_percent=3):
    cuisines = [
        'irish', 'british', 'french', 'italian', 'spanish', 'greek', 'moroccan', 'russian',
        'southern_us', 'cajun_creole', 'jamaican', 'brazilian', 'mexican',
        'indian', 'thai', 'vietnamese', 'chinese', 'japanese', 'korean', 'filipino']
    
    # Prepare food lookup dictionary
    food_lookup = df_foods[["name", "food_group"]].dropna().copy()
    food_lookup["name"] = food_lookup["name"].str.strip().str.lower()
    lookup_dict = dict(zip(food_lookup["name"], food_lookup["food_group"]))

    # Count food groups per cuisine
    cuisine_group_counts = {}
    for cuisine in cuisines:
        cuisine_df = df[df["cuisine"] == cuisine]
        if cuisine_df.empty:
            continue

        group_counts = Counter()
        for _, row in cuisine_df.iterrows():
            for ing in row["matched_ingredients"]:
                group = lookup_dict.get(ing.strip().lower())
                if group:
                    group_counts[group] += 1

        total = sum(group_counts.values())
        if total == 0:
            continue

        # Split into main groups vs small groups
        main_counts = {g: v for g, v in group_counts.items() if 100*v/total >= min_percent}
        other_count = sum(v for g, v in group_counts.items() if 100*v/total < min_percent)

        if other_count > 0:
            main_counts["Other"] = other_count

        # Normalize to percentages
        filtered_percent = {g: 100*v/sum(main_counts.values()) for g, v in main_counts.items()}
        cuisine_group_counts[cuisine] = filtered_percent

    if not cuisine_group_counts:
        print("No food group matches found for any cuisine")
        return

    # Get all unique food groups
    all_groups = sorted({g for counts in cuisine_group_counts.values() for g in counts})

    # Force "Other" to be last
    if "Other" in all_groups:
        all_groups = [g for g in all_groups if g != "Other"] + ["Other"]

    # Prepare normalized data
    data = {cuisine: [cuisine_group_counts[cuisine].get(g, 0) for g in all_groups] 
            for cuisine in cuisine_group_counts}

    # Sort cuisines by largest segment
    cuisines_sorted = sorted(
        data.keys(),
        key=lambda c: max(data[c]),
        reverse=True
    )

    # Colors
    cmap = plt.get_cmap("Paired")
    colors = {group: cmap(i / len(all_groups)) for i, group in enumerate(all_groups)}

    # Plot
    fig, ax = plt.subplots(figsize=(12,6))
    bottom = [0] * len(cuisines_sorted)
    for i, group in enumerate(all_groups):
        values = [data[cuisine][i] for cuisine in cuisines_sorted]
        ax.bar(cuisines_sorted, values, bottom=bottom, label=group, color=colors[group])
        bottom = [bottom[j] + values[j] for j in range(len(values))]

    # Annotate percentages
    for j, cuisine in enumerate(cuisines_sorted):
        cumulative = 0
        for i, group in enumerate(all_groups):
            val = data[cuisine][i]
            if val > 0:
                ax.text(j, cumulative + val/2, f"{val:.1f}%", ha='center', va='center', fontsize=6)
                cumulative += val

    ax.set_ylabel("Percentage of Ingredients")
    ax.set_title("Food Group Distribution Across Cuisines (Normalized)")
    ax.legend(title="Food Group", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(fontsize=7, rotation=45)
    plt.tight_layout()
    plt.show()
    return fig

#Usage
#cuisine_food_group_stacked_bar(df, df_foods, min_percent=3)


