
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






def load_json_from_zip(zip_path, json_file_name):
    with zipfile.ZipFile(zip_path, 'r') as z:
        with z.open(json_file_name) as f:
            return json.load(f)

df = load_json_from_zip("needle.zip", "ingredient_matching_best_of_both_results.json") 
df = pd.DataFrame(df)
print(df.head())
print(df.columns)


df_foods = pd.read_json("Food.json", lines=True, encoding="utf-8")
print(df_foods.head())
print(df_foods.columns)

#--------------------------------------------------------------------------------------------------------------------------------------

def ingredient_count_distribution(df, cuisine_name):
    """
    Plot % frequency distribution of number of matched ingredients per recipe
    for a given cuisine.
    """

    # Filter recipes by cuisine
    df_cuisine = df[df["cuisine"] == cuisine_name].copy()

    # Count number of matched ingredients per recipe
    df_cuisine["num_ingredients"] = df_cuisine["matched_ingredients"].apply(len)

    # Frequency distribution in %
    freq = df_cuisine["num_ingredients"].value_counts(normalize=True).sort_index() * 100

    # Plot
    fig = plt.figure(figsize=(8, 5))
    freq.plot(kind="bar", color="teal", edgecolor="black")
    plt.title(f"Distribution of Number of Ingredients in {cuisine_name} Cuisine")
    plt.xlabel("Number of Ingredients per Recipe")
    plt.ylabel("Frequency (%)")
    plt.xticks(rotation=0)

    return fig  # optional: return the frequency table



#-------------------------------------------------------------------------------------------------------------------------------------


def get_top_group_color_map(df, df_foods, top_n=20, cmap_name="tab20"):
    # normalize food names
    food_lookup = df_foods[["name","food_group"]].dropna().copy()
    food_lookup["name"] = food_lookup["name"].str.strip().str.lower()
    lookup_dict = dict(zip(food_lookup["name"], food_lookup["food_group"]))

    # count food groups across all recipes
    from collections import Counter
    group_counts = Counter()
    for _, row in df.iterrows():
        for ing in row["matched_ingredients"]:
            group = lookup_dict.get(ing.strip().lower())
            if group and group.lower() != "unknown":
                group_counts[group] += 1

    # take the top_n food groups
    top_groups = [g for g, _ in group_counts.most_common(top_n)]

    # assign colors using tab20
    cmap = get_cmap(cmap_name, len(top_groups))
    colors = {group: cmap(i) for i, group in enumerate(top_groups)}

    return colors

# build once
GROUP_COLORS = get_top_group_color_map(df, df_foods, top_n=20)



def cuisine_food_group_pie(df, df_foods, cuisine, min_percent=3):
    cuisine_df = df[df["cuisine"] == cuisine]
    if cuisine_df.empty:
        print(f"No recipes found for cuisine: {cuisine}")
        return

    food_lookup = df_foods[["name","food_group"]].dropna().copy()
    food_lookup["name"] = food_lookup["name"].str.strip().str.lower()
    lookup_dict = dict(zip(food_lookup["name"], food_lookup["food_group"]))

    group_counts = Counter()
    for _, row in cuisine_df.iterrows():
        for ing in row["matched_ingredients"]:
            group = lookup_dict.get(ing.strip().lower())
            if group:
                group_counts[group] += 1

    if not group_counts:
        print(f"No food group matches for cuisine '{cuisine}'")
        return

    total = sum(group_counts.values())
    labels, sizes, colors = [], [], []
    for g, v in group_counts.items():
        if 100*v/total >= min_percent:
            labels.append(g)
            sizes.append(v)
            colors.append(GROUP_COLORS.get(g, "grey"))

    fig = plt.figure(figsize=(8,8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=colors)
    plt.title(f"{cuisine.capitalize()} Cuisine - Food Group Distribution")

    return fig


def cuisine_food_group_pie_top_ingredients_plot_combined(df, df_foods, cuisine, min_percent=3, top_n=3):
    cuisine_df = df[df["cuisine"] == cuisine]
    if cuisine_df.empty:
        print(f"No recipes found for cuisine: {cuisine}")
        return

    food_lookup = df_foods[["name","food_group"]].dropna().copy()
    food_lookup["name"] = food_lookup["name"].str.strip().str.lower()
    lookup_dict = dict(zip(food_lookup["name"], food_lookup["food_group"]))

    group_counts = Counter()
    group_ingredients = defaultdict(Counter)

    for _, row in cuisine_df.iterrows():
        for ing in row["matched_ingredients"]:
            ing_norm = ing.strip().lower()
            group = lookup_dict.get(ing_norm)
            if group and group.lower() != "unknown":
                group_counts[group] += 1
                group_ingredients[group][ing_norm] += 1

    total = sum(group_counts.values())
    filtered_groups = [g for g, v in group_counts.items() if 100*v/total >= min_percent]
    if not filtered_groups:
        print(f"All food groups below {min_percent}% for cuisine '{cuisine}'")
        return

    x_labels, values, colors = [], [], []
    for group in filtered_groups:
        top_ings = group_ingredients[group].most_common(top_n)
        for ing, count in top_ings:
            x_labels.append(ing)
            values.append(count)
            colors.append(GROUP_COLORS.get(group, "grey"))

    fig = plt.figure(figsize=(max(12, len(x_labels)*0.5),6))
    plt.bar(x_labels, values, color=colors)
    plt.title(f"{cuisine.capitalize()} Cuisine - Top {top_n} Ingredients per Major Food Group")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")

    legend_handles = [mpatches.Patch(color=GROUP_COLORS[g], label=g) for g in filtered_groups]
    plt.legend(handles=legend_handles, title="Food Groups", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()

    return fig

#----------------------------------------------------------------------------------------------------------------------------


# ------------- TOP N INGREDIENT COMBOS--------------------------------------------------------------------------------------
def plot_top_n_ingredient_combos(df, cuisine, n=2, top_k=10):
    """
    Computes and plots the top `top_k` n-ingredient combinations for a given cuisine
    using a waffle-style matrix where each row = combo, each column = ingredient.
    """
    # Filter recipes by cuisine
    cuisine_df = df[df['cuisine'] == cuisine]
    
    # Count combinations
    combo_counter = Counter()
    for ingredients_list in cuisine_df['matched_ingredients']:
        combo_counter.update(combinations(sorted(ingredients_list), n))
    
    # Get top combinations
    top_combos = combo_counter.most_common(top_k)
    combos, counts = zip(*top_combos)
    
    # Collect unique ingredients in top combos
    unique_ingredients = sorted(set(ing for combo in combos for ing in combo))
    
    # Build binary matrix and color matrix
    matrix = []
    for combo in combos:
        row = [1 if ing in combo else 0 for ing in unique_ingredients]
        matrix.append(row)
    
    df_matrix = pd.DataFrame(matrix, columns=unique_ingredients)

    # Plot heatmap
    fig = plt.figure(figsize=(8, 4))
    ax = sns.heatmap(df_matrix, annot=False, cbar=False, cmap=['white', 'skyblue'], fmt='d', linewidths=3, linecolor='white')
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=8, rotation=45, ha='right')

    
    plt.xlabel('matched_ingredients',fontsize=10)
    plt.ylabel(f"Top {top_k} {n}-ingredient Combos",fontsize=10)
    plt.title(f"Top {top_k} {n}-ingredient Combos in {cuisine} Cuisine",fontsize=10)
    plt.tight_layout()

    return fig


#---------------------------------------------------------------------------------------------------------------------------------




#---------------------------------------------------------------------------------------------------------------------------------------

def plot_least_ingredient_wordcloud(df, cuisine, ignore_outliers=True, max_words=50):
    # Filter recipes by cuisine
    cuisine_df = df[df["cuisine"] == cuisine]
    if cuisine_df.empty:
        print(f"No recipes found for cuisine: {cuisine}")
        return
    
    # Count ingredients
    ingredient_counts = Counter()
    for _, row in cuisine_df.iterrows():
        for ing in row["matched_ingredients"]:
            ingredient_counts[ing.lower()] += 1
    
    if not ingredient_counts:
        print(f"No ingredients found for cuisine '{cuisine}'")
        return
    
    df_counts = pd.DataFrame.from_dict(ingredient_counts, orient="index", columns=["count"])
    
    if ignore_outliers:
        # Remove extreme outliers using IQR
        Q1 = df_counts['count'].quantile(0.25)
        Q3 = df_counts['count'].quantile(0.75)
        IQR = Q3 - Q1
        df_counts = df_counts[(df_counts['count'] >= Q1 - 1.5*IQR) & (df_counts['count'] <= Q3 + 1.5*IQR)]
    
    # Convert back to dict
    freqs = df_counts['count'].to_dict()
    
    # Generate word cloud
    wc = WordCloud(width=800, height=400, background_color="white", colormap="winter",
                   max_words=max_words).generate_from_frequencies(freqs)
    
    # Plot
    fig  = plt.figure(figsize=(10,6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Least common ingredient Word Cloud for {cuisine.capitalize()} Cuisine", fontsize=16)
    
    return fig


#------------------------------------------------------------------------------------------------------------------------------------




#--------------------------------------------------------------------------------------------------------------------------------


# ---------------------1. Word Cloud of Ingredients------------------------------------------------------------------------------

def plot_most_common_ingredients_wordcloud(df,cuisine_name):
    recipes = df[df["cuisine"] == cuisine_name]['matched_ingredients']
    all_ingredients = [ingredient for recipe in recipes for ingredient in recipe]
    text = " ".join(all_ingredients)
    
    wc = WordCloud(width=800, height=400, background_color="white",colormap="hot").generate(text)
    
    fig = plt.figure(figsize=(12,6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(f"Most common ingredients Word Cloud for - {cuisine_name.capitalize()} Cuisine")

    return fig

#------------------------------------------------------------------------------------------------------------------------------

# ----------- 2. Ingredient Co-occurrence Heatmap ------------------------------------------
def cooccurrence_heatmap(cuisine_name, top_n=20):
    recipes = df[df["cuisine"] == cuisine_name]['matched_ingredients']
    all_ingredients = [ingredient for recipe in recipes for ingredient in recipe]
    
    # Count top N ingredients
    top_ingredients = [item[0] for item in Counter(all_ingredients).most_common(top_n)]
    
    # Initialize co-occurrence matrix
    co_matrix = pd.DataFrame(0, index=top_ingredients, columns=top_ingredients)
    
    for recipe in recipes:
        # Consider only top ingredients
        present = [i for i in recipe if i in top_ingredients]
        for a, b in combinations(present, 2):
            co_matrix.loc[a, b] += 1
            co_matrix.loc[b, a] += 1
    
    # Plot heatmap
    plt.figure(figsize=(10,8))
    sns.heatmap(co_matrix, annot=False, fmt="d", cmap="rocket_r")
    plt.title(f"Ingredient Co-occurrence Heatmap - {cuisine_name.capitalize()}")
    plt.show()

# Example usage:
#cooccurrence_heatmap("indian", top_n=15)


# ---------------3. Unique Ingredients per Recipe --------------------------------
def unique_ingredients_stats(cuisine_name):
    recipes = df[df["cuisine"] == cuisine_name]['matched_ingredients']
    unique_counts = [len(set(recipe)) for recipe in recipes]
    
    plt.figure(figsize=(10,5))
    plt.hist(unique_counts, bins=15, color="skyblue", edgecolor="black")
    plt.title(f"Unique Ingredients per Recipe - {cuisine_name.capitalize()}")
    plt.xlabel("Number of Unique Ingredients")
    plt.ylabel("Number of Recipes")
    plt.show()
    
    print(f"{cuisine_name.capitalize()} - Avg unique ingredients per recipe: {sum(unique_counts)/len(unique_counts):.2f}")
    print(f"Median unique ingredients: {pd.Series(unique_counts).median()}")
    
# Example usage:
#unique_ingredients_stats("mexican")

#---------------------------------------------------------------------------------------------------------------------------


# ------------ 4. Jaccard Similarity Between Recipes------------------------------------------------------------------------

def recipe_similarity_jaccard(df, cuisine):
    # Filter recipes for the cuisine
    cuisine_df = df[df["cuisine"] == cuisine]
    recipes = cuisine_df["matched_ingredients"].tolist()
    
    similarities = []
    
    # Compare all pairs
    for r1, r2 in combinations(recipes, 2):
        set1, set2 = set([i.lower() for i in r1]), set([i.lower() for i in r2])
        jaccard = len(set1 & set2) / len(set1 | set2)
        similarities.append(jaccard)
    
    if similarities:
        mean_similarity = sum(similarities) / len(similarities)
        print(f"Average similarity in {cuisine} cuisine: {mean_similarity:.2f}")
        
        # Plot histogram
        fig = plt.figure(figsize=(8,5))
        plt.hist(similarities, bins=10, color='skyblue', edgecolor='black')
        plt.title(f"Distribution of Recipe Similarities ({cuisine.capitalize()} Cuisine)")
        plt.xlabel("Jaccard Similarity")
        plt.ylabel("Number of Recipe Pairs")
        plt.grid(axis='y', alpha=0.75)
    else:
        print(f"Not enough recipes in {cuisine} to measure similarity.")
    
    return fig

#----------------------------------------------------------------------------------------------------------------------------------


#--------------------------- 5. Recipe Cosine Similarity -------------------------------------------------------------------------------
def recipe_similarity_cosine(df, cuisine):
    # Filter recipes for the cuisine
    cuisine_df = df[df["cuisine"] == cuisine]
    recipes = cuisine_df["matched_ingredients"].tolist()
    
    if len(recipes) < 2:
        print(f"Not enough recipes in {cuisine} to measure similarity.")
        return []

    # Join ingredients as a space-separated string per recipe
    recipe_texts = [" ".join([i.lower() for i in r]) for r in recipes]
    
    # Vectorize the recipes
    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform(recipe_texts)
    
    # Compute cosine similarity matrix
    sim_matrix = cosine_similarity(X)
    
    # Extract upper triangle (excluding diagonal) as pairwise similarities
    similarities = sim_matrix[np.triu_indices_from(sim_matrix, k=1)]
    
    mean_similarity = np.mean(similarities)
    print(f"Average cosine similarity in {cuisine} cuisine: {mean_similarity:.2f}")
    
    # Plot histogram
    fig = plt.figure(figsize=(8,5))
    plt.hist(similarities, bins=10, color='lightcoral', edgecolor='black')
    plt.title(f"Distribution of Recipe Cosine Similarities ({cuisine.capitalize()} Cuisine)")
    plt.xlabel("Cosine Similarity")
    plt.ylabel("Number of Recipe Pairs")
    plt.grid(axis='y', alpha=0.75)
    
    return fig


# -------------- Top X-Ingredient Combinations -----------------------------------------------------------------------------------------
# TODO change frequency to precentage
def plot_top_n_ingredient_combos_old(df, cuisine, n=2, top_k=10):
    """
    Computes and plots the top `top_k` n-ingredient combinations for a given cuisine.
    
    Parameters:
        df (pd.DataFrame): DataFrame with columns ['id', 'cuisine', 'matched_ingredients']
        cuisine (str): Cuisine to filter
        n (int): Number of ingredients in each combination
        top_k (int): Number of top combinations to show
    """
    # Filter recipes by cuisine
    cuisine_df = df[df['cuisine'] == cuisine]
    
    # Count combinations
    combo_counter = Counter()
    for ingredients_list in cuisine_df['matched_ingredients']:
        ingredients_list = sorted(ingredients_list)
        combo_counter.update(combinations(ingredients_list, n))
    
    # Get top combinations
    top_combos = combo_counter.most_common(top_k)
    combos, counts = zip(*top_combos)
    
    # Convert tuples to string for plotting
    combo_labels = [' + '.join(combo) for combo in combos]
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.barh(combo_labels[::-1], counts[::-1], color='skyblue')  # reverse for descending order
    plt.xlabel('Frequency')
    plt.ylabel(f'Top {n}-Ingredient Combinations')
    plt.title(f'Top {top_k} {n}-Ingredient Combinations for {cuisine.capitalize()} Cuisine')
    plt.tight_layout()
    plt.show()

#-------------------------------------------------------------------------------------------------------------------------------------

#--------------------- 6. Cuisine General Information --------------------------------------------------------------


def cuisine_summary_metrics_text(df, df_foods, cuisine):
    # Filter recipes for the cuisine
    cuisine_df = df[df["cuisine"] == cuisine]
    if cuisine_df.empty:
        return f"No recipes found for cuisine: {cuisine}"

    # Total number of recipes
    total_recipes = len(cuisine_df)

    # Flatten all ingredients into one list
    all_ingredients = [ing.strip().lower() for lst in cuisine_df["matched_ingredients"] for ing in lst]
    total_ingredients = len(all_ingredients)

    # Average number of ingredients per recipe
    recipe_lengths = [len(lst) for lst in cuisine_df["matched_ingredients"]]
    avg_ingredients = np.mean(recipe_lengths)
    std_ingredients = np.std(recipe_lengths)

    # Most common food group
    food_lookup = df_foods[["name","food_group"]].dropna().copy()
    food_lookup["name"] = food_lookup["name"].str.strip().str.lower()
    lookup_dict = dict(zip(food_lookup["name"], food_lookup["food_group"]))
    
    group_counts = Counter()
    for ing in all_ingredients:
        group = lookup_dict.get(ing)
        if group and group.lower() != "unknown":
            group_counts[group] += 1
    most_common_group = group_counts.most_common(1)[0][0] if group_counts else "N/A"

    # Ingredient rarity index (proportion of ingredients used only once)
    ing_counter = Counter(all_ingredients)
    rarity_index = sum(1 for count in ing_counter.values() if count == 1) / len(ing_counter) if ing_counter else 0

    # Create summary string
    summary_text = (
        f"Cuisine: {cuisine.capitalize()}\n"
        f"Total recipes: {total_recipes}\n"
        f"Total ingredients: {total_ingredients}\n"
        f"Average ingredients per recipe: {avg_ingredients:.2f}\n"
        f"Recipe length std dev: {std_ingredients:.2f}\n"
        f"Most common food group: {most_common_group}\n"
        f"Ingredient rarity index: {rarity_index:.2f}\n"
    )

    return summary_text

#-----------------------------------------------------------------------------------------------------------

#----------- Main Funciton For Per-Cuisine Analysis --------------------------------------------------------------

def run_cuisine_analysis(df, df_foods, output_dir="cuisine_plots"):
    """
    Runs all analysis functions for each cuisine and saves plots and summary text.
    
    Parameters:
        df (pd.DataFrame): Recipe DataFrame
        df_foods (pd.DataFrame): Food info DataFrame
        output_dir (str): Base folder to save plots and summaries
    """
    cuisines = df["cuisine"].dropna().unique()
    
    for cuisine in cuisines:
        print(f"Processing {cuisine} cuisine...")
        
        # Create folder for this cuisine
        cuisine_dir = os.path.join(output_dir, cuisine)
        os.makedirs(cuisine_dir, exist_ok=True)
        
        # --- Food group pie ---
        fig = cuisine_food_group_pie(df, df_foods, cuisine)
        fig.savefig(os.path.join(cuisine_dir, "cuisine_food_group_pie.png"))
        plt.close(fig)
        
        # --- Top ingredients per food group ---
        fig = cuisine_food_group_pie_top_ingredients_plot_combined(df, df_foods, cuisine)
        fig.savefig(os.path.join(cuisine_dir, "top_ingredients_per_group.png"))
        plt.close(fig)
        
        # --- Top ingredient combinations ---
        fig = plot_top_n_ingredient_combos(df, cuisine=cuisine, n=4, top_k=15)
        fig.savefig(os.path.join(cuisine_dir, "top_ingredient_combos.png"))
        plt.close(fig)
        
        # --- Word clouds Most and Least Common ingredients---
        fig = plot_least_ingredient_wordcloud(df, cuisine)
        fig.savefig(os.path.join(cuisine_dir, "least_common_ingredients_wordcloud.png"))
        plt.close(fig)
        
        fig = plot_most_common_ingredients_wordcloud(df, cuisine)
        fig.savefig(os.path.join(cuisine_dir, "most_common_ingredients_wordcloud.png"))
        plt.close(fig)
        
        # --- Recipe similarities ---
        fig = recipe_similarity_jaccard(df, cuisine)
        fig.savefig(os.path.join(cuisine_dir, "recipe_similarity_jaccard.png"))
        plt.close(fig)
        
        fig = recipe_similarity_cosine(df, cuisine)
        fig.savefig(os.path.join(cuisine_dir, "recipe_similarity_cosine.png"))
        plt.close(fig)

        # --- Ingredient count distribution ---
        fig = ingredient_count_distribution(df, cuisine)
        fig.savefig(os.path.join(cuisine_dir, "ingredient_count_distribution.png"))
        plt.close(fig)
        
        # --- Cuisine summary text ---
        summary_text = cuisine_summary_metrics_text(df, df_foods, cuisine)
        with open(os.path.join(cuisine_dir, "summary.txt"), "w", encoding="utf-8") as f:
            f.write(summary_text)
        
        print(f"Finished processing {cuisine}\n")


run_cuisine_analysis(df, df_foods, output_dir="cuisine_plots")