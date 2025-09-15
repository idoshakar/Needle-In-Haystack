import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import duckdb;

TRAIN_SET_SIZE = 20000
TEST_SET_SIZE = 200
SIZE = TRAIN_SET_SIZE + TEST_SET_SIZE
DATABASE_FILENAME = 'recipe_cuisine.db'
RECIPES_FILENAME = 'train_lines.json'

def get_ingredient_names(con):
    unique_ingredients = con.execute("""
            SELECT DISTINCT UNNEST(ingredients) AS ingredient
                FROM recipes
        ORDER BY ingredient;
    """).fetchall()

    ingredient_names = [row[0].replace("'", "''") for row in unique_ingredients]

    return ingredient_names


def get_ingredients(con):
    ingredient_names = get_ingredient_names(con)

    print("\nFound unique ingredients:")
    print(ingredient_names)

    ingredient_case_statements = [
        rf"LIST_CONTAINS(ingredients, '{ingredient}') AS '{ingredient}'"
        for ingredient in ingredient_names
    ]
    case_statements_str = ",\n    ".join(ingredient_case_statements)

    return case_statements_str

def get_cuisine_names(con):
    unique_cuisines = con.execute("""
            SELECT DISTINCT cuisine AS cuisine
                FROM recipes
        ORDER BY cuisine;
    """).fetchall()

    return [row[0].replace("'", "''") for row in unique_cuisines]

def get_cuisines(con):
    cuisine_names = get_cuisine_names(con)

    print("\nFound unique cuisines:")
    print(cuisine_names)

    case_statements = [
        rf"cuisine = '{cuisine}' AS '{cuisine}'"
        for cuisine in cuisine_names
    ]
    cuisine_case_statements_str = ",\n    ".join(case_statements)

    return cuisine_case_statements_str

def get_database():
    return duckdb.connect(DATABASE_FILENAME)

def process_recipe_cuisine():
    con = get_database()

    print("Creating sample 'recipes' table with a list of ingredients...")
    con.execute(rf"""
        SET preserve_insertion_order=false;
        SET threads=1;
        CREATE TABLE recipes AS
            SELECT *
                FROM read_json('{RECIPES_FILENAME}');
    """)

    ingredient_case_statements = get_ingredients(con);

    # Extract the cuisine names from the result
    cuisine_case_statements = get_cuisines(con)

    # Construct the full SQL query
    one_hot_sql = rf"""
        CREATE TABLE processed_recipe_cuisine AS
                SELECT id, {ingredient_case_statements}, {cuisine_case_statements}
                    FROM
                        recipes
            ORDER BY
                id;
        DESCRIBE processed_recipe_cuisine;
    """

    print("\nGenerated SQL Query:\n" + one_hot_sql)

    # --- Step 4: Execute the final one-hot encoding query ---
    result = con.execute(one_hot_sql).df()

    print("\nOne-Hot Encoded Result:")
    print(result)

    con.execute("COPY processed_recipe_cuisine TO 'output.csv' (HEADER, DELIMITER ',');")

    # Close the connection
    con.close()

if __name__ == '__main__':
    process_recipe_cuisine()
