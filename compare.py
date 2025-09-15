import json
import pandas as pd
import re
import time
import glob
import duckdb
import shelve
import spellchecker
from typing import Dict, List, Tuple, Optional
from rapidfuzz import fuzz
from dataclasses import dataclass
import warnings

from pathlib import Path

def get_json_files_in_directory(directory_path):
    """
    Finds and loads all JSON files from a specified directory.

    Args:
        directory_path (str or Path): The path to the directory.

    Returns:
        list: A list of dictionaries, where each dictionary
              is the content of a JSON file.
    """
    json_data = []
    # Create a Path object from the given directory string
    path = Path(directory_path)

    # Use glob() to find all files ending with '.json'
    for file_path in path.glob("*.json"):
        #try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                json_data.append(data)
        # except (json.JSONDecodeError, FileNotFoundError) as e:
        #     print(f"Error reading {file_path}: {e}")

    return json_data

# Example usage:
# Assuming you have a folder named 'my_data' with some JSON files
# my_data/file1.json
# my_data/file2.json
# all_json_content = get_json_files_in_directory('my_data')
# print(all_json_content)


warnings.filterwarnings('ignore')

# Initialize spell checker
spell = spellchecker.SpellChecker('en')
spell.distance = 1

FOOD_JSON = 'foodb/Food.json'

FINAL_FOOD_DATASET = 'FINAL FOOD DATASET//FOOD-DATA-GROUP*.csv'

RECIPES = 'whats-cooking/train.json'

def correct(sentence: str) -> str:
    words = sentence.split()
    corrected_words = []
    for word in words:
        correction = spell.correction(word)
        if correction == None:
            return sentence
        corrected_words.append(correction)
    return ' '.join(corrected_words)


@dataclass
class MatchResult:
    """Container for match results"""
    original: str
    cleaned: str
    matched: str
    score: float
    method: str
    confidence: str
    metadata: Dict | None = None


class DatabaseSearch:
    """Wrapper for the DuckDB search functionality"""

    def __init__(self, database_name: str = ':memory:',
                 ingredients_file: str = FOOD_JSON):
        self._database_name = database_name
        self.ingredients_file = ingredients_file
        self.con = None

    def __enter__(self):
        self.con = duckdb.connect(self._database_name)
        self.con.execute(f'''
            INSTALL fts;
            LOAD fts;
            SET preserve_insertion_order=false;
            SET threads=1;
            CREATE TABLE ingredients AS
                SELECT *
                    FROM read_json_auto('{self.ingredients_file}');
        ''')
        self.con.execute('''PRAGMA create_fts_index(
            'main.ingredients', 'id', 'name', 'description'
        );''')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.con:
            self.con.close()
            self.con = None

    def resolve_word(self, word: str) -> list[str]:
        if not self.con:
            return []
        word = word.replace("'", "''")
        search = self.con.execute(f'''
            SELECT id, name, score
                        FROM (
                            SELECT *, fts_main_ingredients.match_bm25(
                                id,
                                '{word}',
                                k:=1.2, b:=0.0
                            ) + ((LOWER(name) = LOWER('{word}'))::FLOAT) AS score
                            FROM ingredients
                        ) sq
                    WHERE score IS NOT NULL
            ORDER BY score DESC;
        ''')
        search_list = search.df()['name'].tolist()
        return search_list


class HybridIngredientMatcher:
    """
    Combines fuzzy matching with real database search for optimal ingredient normalization.
    """

    def __init__(self, foods_db_path: str,
                 foodb_path: str = FOOD_JSON,
                 fuzzy_threshold: float = 0.8,
                 db_threshold: float = 0.7):
        """
        Initialize the hybrid matcher with real databases

        Args:
            foods_db_path: Path to CSV files with food database
            foodb_path: Path to FooDB JSON file
            fuzzy_threshold: Minimum score for fuzzy matches
            db_threshold: Minimum score for database matches
        """
        self.fuzzy_threshold = fuzzy_threshold
        self.db_threshold = db_threshold
        self.foodb_path = foodb_path

        # Load fuzzy matching database
        #if foods_db_path:
        csv_files = glob.glob(foods_db_path)
        df_combined = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)
        df_combined['food'] = self._clean_ingredients_fuzzy(df_combined['food'])
        self.ingredient_database = df_combined['food'].str.replace("suger", "sugar").tolist()
        # else:
        #     # Default database if no path provided
        #     self.ingredient_database = [
        #         'salt', 'pepper', 'sugar', 'white sugar', 'brown sugar', 'flour', 'butter',
        #         'milk', 'egg', 'eggs', 'cheese', 'cheddar cheese', 'tomato', 'tomatoes',
        #         'onion', 'red onion', 'garlic', 'olive oil', 'vegetable oil', 'chicken',
        #         'beef', 'pork', 'rice', 'pasta', 'bread', 'lettuce', 'romaine lettuce',
        #         'carrot', 'potato', 'apple', 'soy sauce', 'vinegar', 'rice vinegar',
        #         'lemon', 'ginger', 'basil', 'oregano', 'vanilla extract', 'baking powder',
        #         'tomato sauce', 'chicken breast', 'ground beef', 'bell pepper', 'mushrooms',
        #         'sesame oil', 'hoisin sauce', 'wonton wrappers'
        #     ]

        # Create lookup sets for fast exact matching
        self.exact_lookup = set([ing.lower().strip() for ing in self.ingredient_database])

    def _clean_ingredients_fuzzy(self, ingredients):
        """Clean a list of ingredients for fuzzy matching"""
        cleaned_ingredients = []
        for ingredient in ingredients:
            cleaned_ingredients.append(self.clean_ingredient_fuzzy(ingredient))
        return cleaned_ingredients

    def clean_ingredient_fuzzy(self, ingredient: str) -> str:
        """Clean ingredient for fuzzy matching (your original method)"""
        if not isinstance(ingredient, str):
            return ""

        # Replace hyphens with whitespace
        cleaned = ingredient.replace('-', ' ')

        # More permissive: allow unicode letters
        cleaned = re.sub(r'[^\w\s]', '', cleaned, flags=re.UNICODE)
        cleaned = re.sub(r'[0-9]', '', cleaned)

        # Remove extra spaces and convert to lowercase
        cleaned = ' '.join(cleaned.split()).lower()

        # Special treatment: remove "beaten" from ingredients containing "egg"
        if 'egg' in cleaned:
            cleaned = re.sub(r'\bbeaten\b', '', cleaned)

        # Remove specific words
        words_to_remove = ['aged', 'frozen', 'fresh', 'organic', 'chopped', 'ground',
                           'cubed', 'sliced', 'cooked', 'roasted', 'dried', 'fried',
                           'fully', 'flavored', 'unflavored', 'grated', 'cold']

        for word in words_to_remove:
            cleaned = re.sub(rf'\b{word}\b', '', cleaned)

        # Clean up any extra spaces
        cleaned = ' '.join(cleaned.split())
        return cleaned

    def clean_ingredient_db(self, ingredient: str) -> str:
        """Clean ingredient for database search (simplified)"""
        if not isinstance(ingredient, str):
            return ""

        cleaned = re.sub(r'[^\w\s]', '', ingredient, flags=re.UNICODE)
        cleaned = re.sub(r'[0-9]', '', cleaned)
        cleaned = ' '.join(cleaned.split()).lower()

        words_to_remove = ['aged', 'frozen', 'fresh', 'organic', 'chopped', 'ground',
                           'cubed', 'sliced', 'cooked', 'roasted', 'dried', 'fried',
                           'fully', 'flavored', 'unflavored', 'powdered', 'plain']

        for word in words_to_remove:
            cleaned = cleaned.replace(word, '')

        cleaned = ' '.join(cleaned.split())
        return cleaned

    def exact_match(self, cleaned_ingredient: str) -> Optional[str]:
        """Check for exact matches in fuzzy database"""
        if cleaned_ingredient in self.exact_lookup:
            return cleaned_ingredient
        return None

    def database_search_with_cache(self, ingredient: str) -> Tuple[str, float]:
        """Search using real DuckDB with caching"""
        with shelve.open("cached_search") as db:
            result = db.get(ingredient, None)
            if result:
                return result  # Cached results assumed high confidence

        # Spell correction
        corrected_ingredient = correct(ingredient)

        # Search in database
        #try:
        with DatabaseSearch(ingredients_file=self.foodb_path) as search:
            # Try original first
            results = search.resolve_word(ingredient)
            if not results and corrected_ingredient != ingredient:
                # Try corrected version
                results = search.resolve_word(corrected_ingredient)
                if results:
                    print(f"original: {ingredient}, corrected: {corrected_ingredient}")

            if results:
                best_match = results[0].lower()

                # Calculate confidence based on position in results
                confidence = 1.0 if len(results) == 1 else max(0.7, 1.0 - (0.1 * min(3, len(results))))

                # Cache the result
                with shelve.open("cached_search") as db:
                    db[ingredient] = best_match, confidence

                return best_match, confidence
        # except Exception as e:
        #     print(f"Database search failed for '{ingredient}': {e}")
        #     return ingredient, 0.0
        with shelve.open("cached_search") as db:
            db[ingredient] = "", 0.0
        return ingredient, 0.0

    def fuzzy_match(self, cleaned_ingredient: str) -> List[Tuple[str, float]]:
        """Enhanced fuzzy matching with hierarchical scoring (your original method)"""

        with shelve.open("cached_fuzzy") as db:
            result = db.get(cleaned_ingredient, None)
            if result:
                return result  # Cached results assumed high confidence

        results = []

        for candidate in self.ingredient_database:
            # Extract words for both strings
            words1 = set(re.findall(r'\b\w+\b', cleaned_ingredient))
            words2 = set(re.findall(r'\b\w+\b', candidate))

            if not words1 or not words2:
                continue

            # Calculate enhanced score
            base_score = fuzz.ratio(cleaned_ingredient, candidate) / 100.0
            common_words = words1.intersection(words2)
            num_common_words = len(common_words)

            # Hierarchical scoring
            if num_common_words >= 3:
                word_ratio = num_common_words / max(len(words1), len(words2))
                enhanced_score = 0.6 + (word_ratio * 0.1) + (base_score * 0.2)
            elif num_common_words == 2:
                word_ratio = num_common_words / max(len(words1), len(words2))
                enhanced_score = 0.4 + (word_ratio * 0.1) + (base_score * 0.3)
            elif num_common_words == 1:
                word_ratio = 1.0 / max(len(words1), len(words2))
                enhanced_score = 0.2 + (word_ratio * 0.1) + (base_score * 0.4)
            else:
                enhanced_score = base_score * 0.6

            # Exact match bonuses
            exact_bonus = 0.0
            if words2.issubset(words1):
                if len(words2) == len(words1):
                    exact_bonus = 0.5  # Perfect match
                else:
                    exact_bonus = 0.3  # Target subset of query
            elif words1.issubset(words2):
                exact_bonus = 0.1  # Query subset of target

            final_score = min(enhanced_score + exact_bonus, 1.0)
            results.append((candidate, final_score))

        ret = sorted(results, key=lambda x: x[1], reverse=True)

        with shelve.open("cached_fuzzy") as db:
            db[cleaned_ingredient] = ret

        return ret

    def get_confidence_level(self, score: float, method: str) -> str:
        """Determine confidence level based on score and method"""
        if method == 'exact':
            return 'very_high'
        elif score >= 0.9:
            return 'high'
        elif score >= 0.7:
            return 'medium'
        elif score >= 0.5:
            return 'low'
        elif score != 0.0:
            return 'very_low'
        else:
            return 'none'

    def match_ingredient(self, ingredient: str, strategy: str = 'adaptive') -> MatchResult:
        """
        Main matching function that combines real database search with fuzzy matching
        """
        with shelve.open("cached_search_" + strategy) as db:
            result = db.get(ingredient, None)
            if result:
                return result

        if strategy == 'adaptive':
            ret = self._adaptive_match(ingredient)
        elif strategy == 'db_first':
            ret = self._db_first_match(ingredient)
        elif strategy == 'fuzzy_first':
            ret = self._fuzzy_first_match(ingredient)
        elif strategy == 'best_of_both':
            ret = self._best_of_both_match(ingredient)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        with shelve.open("cached_search_" + strategy) as db:
            db[ingredient] = ret

        return ret

    def _adaptive_match(self, ingredient: str) -> MatchResult:
        """Adaptive strategy using real databases"""
        # Clean for both methods
        cleaned_fuzzy = self.clean_ingredient_fuzzy(ingredient)
        cleaned_db = self.clean_ingredient_db(ingredient)

        # Try exact match first
        exact = self.exact_match(cleaned_fuzzy)
        if exact:
            return MatchResult(
                original=ingredient,
                cleaned=cleaned_fuzzy,
                matched=exact,
                score=1.0,
                method='exact',
                confidence='very_high'
            )

        # Initialize result variables
        db_result = None
        db_score = 0.0
        fuzzy_results = []

        # Strategy selection based on ingredient characteristics
        words = cleaned_fuzzy.split()

        # For single words or common ingredient types, prefer database search
        if len(words) <= 1 or any(word in ['oil', 'sauce', 'cheese', 'milk', 'vinegar', 'sugar'] for word in words):
            db_match, db_score = self.database_search_with_cache(cleaned_db)
            if db_score >= self.db_threshold:
                return MatchResult(
                    original=ingredient,
                    cleaned=cleaned_db,
                    matched=db_match,
                    score=db_score,
                    method='database',
                    confidence=self.get_confidence_level(db_score, 'database')
                )
            db_result = (db_match, db_score)

        # For complex ingredients or if database didn't succeed, use fuzzy matching
        fuzzy_results = self.fuzzy_match(cleaned_fuzzy)
        if fuzzy_results and fuzzy_results[0][1] >= self.fuzzy_threshold:
            return MatchResult(
                original=ingredient,
                cleaned=cleaned_fuzzy,
                matched=fuzzy_results[0][0],
                score=fuzzy_results[0][1],
                method='fuzzy',
                confidence=self.get_confidence_level(fuzzy_results[0][1], 'fuzzy')
            )

        # If we haven't tried database yet, try it now
        if db_result is None:
            db_match, db_score = self.database_search_with_cache(cleaned_db)
            db_result = (db_match, db_score)

        # Fallback to best available match
        all_results = []
        if db_result and db_result[1] > 0:
            all_results.append((db_result[0], db_result[1], 'database'))
        if fuzzy_results:
            all_results.extend([(r[0], r[1], 'fuzzy') for r in fuzzy_results[:3]])

        if all_results:
            best = max(all_results, key=lambda x: x[1])
            return MatchResult(
                original=ingredient,
                cleaned=cleaned_fuzzy if best[2] == 'fuzzy' else cleaned_db,
                matched=best[0],
                score=best[1],
                method=f'{best[2]}_fallback',
                confidence=self.get_confidence_level(best[1], best[2])
            )

        # No good match found
        return MatchResult(
            original=ingredient,
            cleaned=cleaned_fuzzy,
            matched=cleaned_fuzzy,
            score=0.0,
            method='no_match',
            confidence='none'
        )

    def _best_of_both_match(self, ingredient: str) -> MatchResult:
        """Run both real methods and return the best result"""
        cleaned_fuzzy = self.clean_ingredient_fuzzy(ingredient)
        cleaned_db = self.clean_ingredient_db(ingredient)

        # Try exact match first
        exact = self.exact_match(cleaned_fuzzy)
        if exact:
            return MatchResult(
                original=ingredient,
                cleaned=cleaned_fuzzy,
                matched=exact,
                score=1.0,
                method='exact',
                confidence='very_high'
            )

        # Run both methods
        db_match, db_score = self.database_search_with_cache(cleaned_db)
        fuzzy_results = self.fuzzy_match(cleaned_fuzzy)
        fuzzy_score = fuzzy_results[0][1] if fuzzy_results else 0.0

        # Choose the better method
        if db_score >= fuzzy_score and db_score >= self.db_threshold:
            return MatchResult(
                original=ingredient,
                cleaned=cleaned_db,
                matched=db_match,
                score=db_score,
                method='database',
                confidence=self.get_confidence_level(db_score, 'database'),
                metadata={'fuzzy_score': fuzzy_score}
            )
        elif fuzzy_score >= self.fuzzy_threshold:
            return MatchResult(
                original=ingredient,
                cleaned=cleaned_fuzzy,
                matched=fuzzy_results[0][0],
                score=fuzzy_score,
                method='fuzzy',
                confidence=self.get_confidence_level(fuzzy_score, 'fuzzy'),
                metadata={'db_score': db_score}
            )
        else:
            # Return best available
            if db_score >= fuzzy_score:
                return MatchResult(
                    original=ingredient,
                    cleaned=cleaned_db,
                    matched=db_match,
                    score=db_score,
                    method='database_low_conf',
                    confidence='low'
                )
            else:
                return MatchResult(
                    original=ingredient,
                    cleaned=cleaned_fuzzy,
                    matched=fuzzy_results[0][0] if fuzzy_results else cleaned_fuzzy,
                    score=fuzzy_score,
                    method='fuzzy_low_conf',
                    confidence='low'
                )

    def _db_first_match(self, ingredient: str) -> MatchResult:
        """Try real database method first, fallback to fuzzy"""
        cleaned_db = self.clean_ingredient_db(ingredient)
        cleaned_fuzzy = self.clean_ingredient_fuzzy(ingredient)

        exact = self.exact_match(cleaned_fuzzy)
        if exact:
            return MatchResult(ingredient, cleaned_fuzzy, exact, 1.0, 'exact', 'very_high')

        db_match, db_score = self.database_search_with_cache(cleaned_db)
        if db_score >= self.db_threshold:
            return MatchResult(
                ingredient, cleaned_db, db_match, db_score,
                'database', self.get_confidence_level(db_score, 'database')
            )

        # Fallback to fuzzy
        fuzzy_results = self.fuzzy_match(cleaned_fuzzy)
        if fuzzy_results:
            return MatchResult(
                ingredient, cleaned_fuzzy, fuzzy_results[0][0], fuzzy_results[0][1],
                'fuzzy_fallback', self.get_confidence_level(fuzzy_results[0][1], 'fuzzy')
            )

        return MatchResult(ingredient, cleaned_fuzzy, cleaned_fuzzy, 0.0, 'no_match', 'none')

    def _fuzzy_first_match(self, ingredient: str) -> MatchResult:
        """Try fuzzy method first, fallback to real database"""
        cleaned_fuzzy = self.clean_ingredient_fuzzy(ingredient)
        cleaned_db = self.clean_ingredient_db(ingredient)

        exact = self.exact_match(cleaned_fuzzy)
        if exact:
            return MatchResult(ingredient, cleaned_fuzzy, exact, 1.0, 'exact', 'very_high')

        fuzzy_results = self.fuzzy_match(cleaned_fuzzy)
        if fuzzy_results and fuzzy_results[0][1] >= self.fuzzy_threshold:
            return MatchResult(
                ingredient, cleaned_fuzzy, fuzzy_results[0][0], fuzzy_results[0][1],
                'fuzzy', self.get_confidence_level(fuzzy_results[0][1], 'fuzzy')
            )

        # Fallback to database
        db_match, db_score = self.database_search_with_cache(cleaned_db)
        if db_score > 0:
            return MatchResult(
                ingredient, cleaned_db, db_match, db_score,
                'database_fallback', self.get_confidence_level(db_score, 'database')
            )

        return MatchResult(ingredient, cleaned_fuzzy, cleaned_fuzzy, 0.0, 'no_match', 'none')

    def match_ingredients_batch(self, ingredients: List[str],
                                strategy: str = 'adaptive') -> List[MatchResult]:
        """Match a batch of ingredients"""
        results = []
        for ingredient in ingredients:
            result = self.match_ingredient(ingredient, strategy)
            if result.confidence != 'none':
                results.append(result)
        return results

    def get_statistics(self, results: List[MatchResult]) -> Dict:
        """Generate statistics from match results"""
        total = len(results)
        methods = {}
        confidences = {}

        for result in results:
            methods[result.method] = methods.get(result.method, 0) + 1
            confidences[result.confidence] = confidences.get(result.confidence, 0) + 1

        high_confidence = sum(1 for r in results if r.confidence in ['very_high', 'high'])
        avg_score = sum(r.score for r in results) / total if total > 0 else 0

        return {
            'total_ingredients': total,
            'methods_used': methods,
            'confidence_levels': confidences,
            'high_confidence_rate': high_confidence / total if total > 0 else 0,
            'average_score': avg_score
        }


# Usage example with real databases - process all recipes with all methods
def create_real_pipeline():
    """Process all recipes with all matching methods"""

    foods_db_path = FINAL_FOOD_DATASET
    foodb_path = FOOD_JSON

    # Initialize pipeline with real databases
    matcher = HybridIngredientMatcher(
        foods_db_path=foods_db_path,
        foodb_path=foodb_path
    )

    # Load recipe file (single JSON file with list of recipe dicts)
    print("\nLOADING RECIPE FILE")
    print("=" * 40)

    recipe_file_path = RECIPES
    print(f"Loading recipe file: {recipe_file_path}")

    #try:
        # import os
        # if os.path.exists(recipe_file_path):
        #     file_size = os.path.getsize(recipe_file_path) / (1024 * 1024)  # MB
        #     print(f"Recipe file found: {file_size:.2f} MB")
        # else:
        #     print(f"ERROR: Recipe file not found at {recipe_file_path}")
        #     return

        # Load the JSON file
    with open(recipe_file_path, 'r', encoding='utf-8') as file:
        recipes_data = json.load(file)

    print(f"Loaded {len(recipes_data)} recipes from file")

    # Show sample recipe structure
    if recipes_data:
        sample_recipe = recipes_data[0]
        print(f"Sample recipe structure: {list(sample_recipe.keys())}")
        if 'ingredients' in sample_recipe:
            print(f"Sample ingredients: {sample_recipe['ingredients'][:3]}")

    # except Exception as e:
    #     print(f"ERROR loading recipe file: {e}")
    #     return

    # All strategies to test
    #strategies = ['adaptive', 'best_of_both', 'db_first', 'fuzzy_first']
    strategies = ['adaptive']
    #print("\nPROCESSING ALL RECIPES WITH ALL METHODS")
    print("=" * 60)

    all_results = {}
    overall_stats = {}

    for strategy in strategies:
        print(f"\nProcessing with strategy: {strategy.upper()}")
        print("-" * 40)

        strategy_results = []
        total_ingredients = 0
        total_recipes = 0

        start_time = time.time()

        # Process all recipes in the list
        for recipe_idx, recipe in enumerate(recipes_data):
            #try:
                recipe_title = recipe.get('title', recipe.get('id', f'Recipe_{recipe_idx}'))

                if 'ingredients' not in recipe:
                    print(f"    WARNING: Recipe {recipe_idx} has no ingredients")
                    continue

                # Get ingredients list
                ingredients = recipe['ingredients']

                # Handle case where ingredients might be nested
                if ingredients and isinstance(ingredients[0], list):
                    # If ingredients are in format [["ingredient", "amount"], ...]
                    ingredients = [item[0] if isinstance(item, list) else item for item in ingredients]

                #print(f"    Recipe '{recipe_title}': {len(ingredients)} ingredients")
                if recipe_idx == 0:  # Show first recipe's ingredients
                    print(f"      Sample ingredients: {ingredients[:3]}")

                # Match ingredients using current strategy
                results = matcher.match_ingredients_batch(ingredients, strategy)

                # Store results
                recipe_result = {
                    'recipe_id': recipe.get('id', recipe_idx),
                    'recipe_title': recipe_title,
                    'cuisine': recipe.get('cuisine', 'unknown'),
                    'original_ingredients': ingredients,
                    'matched_ingredients': [r.matched for r in results],
                    'scores': [r.score for r in results],
                    'methods': [r.method for r in results],
                    'confidences': [r.confidence for r in results]
                }

                strategy_results.append(recipe_result)
                total_ingredients += len(ingredients)
                total_recipes += 1

                # Progress update every 1000 recipes
                if total_recipes % 1000 == 0:
                    print(f"  Processed {total_recipes} recipes...")

            # except Exception as e:
            #     print(f"    ERROR processing recipe {recipe_idx}: {e}")
            #     continue

        elapsed_time = time.time() - start_time

        # Calculate statistics for this strategy
        all_results_flat = []
        for recipe_result in strategy_results:
            for i, original in enumerate(recipe_result['original_ingredients']):
                all_results_flat.append(MatchResult(
                    original=original,
                    cleaned="",  # Not needed for stats
                    matched=recipe_result['matched_ingredients'][i],
                    score=recipe_result['scores'][i],
                    method=recipe_result['methods'][i],
                    confidence=recipe_result['confidences'][i]
                ))

        stats = matcher.get_statistics(all_results_flat)

        print(f"\nStrategy: {strategy.upper()} COMPLETED")
        print(f"Recipes processed: {total_recipes}")
        print(f"Total ingredients: {total_ingredients}")
        print(f"High confidence rate: {stats['high_confidence_rate']:.1%}")
        print(f"Average score: {stats['average_score']:.3f}")
        print(f"Processing time: {elapsed_time:.2f} seconds")
        print(f"Methods used: {stats['methods_used']}")

        # Store results
        all_results[strategy] = strategy_results
        overall_stats[strategy] = {
            'total_recipes': total_recipes,
            'total_ingredients': total_ingredients,
            'processing_time': elapsed_time,
            'stats': stats
        }

    # Save all results to files
        print("\nSAVING RESULTS...")

        # Save detailed results for each strategy
        output_file = f'ingredient_matching_{strategy}_results.json'
        #try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(strategy_results, f, indent=2, ensure_ascii=False)
        print(f"Successfully saved {strategy} results to {output_file}")
        # except Exception as e:
        #     print(f"ERROR saving {strategy} results: {e}")

        # Save summary statistics
        #try:
        to_write = json.dumps(overall_stats, indent=2)
        with open('trials/ingredient_matching_summary.jsonl', 'a', encoding='utf-8') as f:
            f.write(to_write + '\n')
        print("Successfully saved summary statistics")

        # except Exception as e:
        #     print(f"ERROR saving summary: {e}")

    # Create comparison report
    create_comparison_report(overall_stats)

    print("\nPROCESSING COMPLETE!")
    if overall_stats:
        max_recipes = max(s['total_recipes'] for s in overall_stats.values())
        print(f"Processed {max_recipes} total recipes")
    print("Check output files for detailed results")


def create_comparison_report(overall_stats):
    """Create a comparison report of all methods"""

    print("\n" + "=" * 80)
    print("COMPREHENSIVE METHOD COMPARISON REPORT")
    print("=" * 80)

    # Performance comparison table
    print(f"\n{'Method':<15} {'Recipes':<8} {'Ingredients':<12} {'Time (s)':<10} {'High Conf %':<12} {'Avg Score':<10}")
    print("-" * 80)

    for strategy, stats in overall_stats.items():
        print(f"{strategy:<15} "
              f"{stats['total_recipes']:<8} "
              f"{stats['total_ingredients']:<12} "
              f"{stats['processing_time']:<10.2f} "
              f"{stats['stats']['high_confidence_rate'] * 100:<12.1f} "
              f"{stats['stats']['average_score']:<10.3f}")

    # Method usage breakdown
    print("\nMETHOD USAGE BREAKDOWN:")
    print("-" * 40)
    for strategy, stats in overall_stats.items():
        print(f"\n{strategy.upper()}:")
        for method, count in stats['stats']['methods_used'].items():
            percentage = (count / stats['total_ingredients']) * 100
            print(f"  {method:<20} {count:>6} ({percentage:>5.1f}%)")

    # Find best performing method
    best_strategy = max(overall_stats.keys(),
                        key=lambda s: overall_stats[s]['stats']['high_confidence_rate'])

    print("\nRECOMMENDATION:")
    print(f"Best performing strategy: {best_strategy.upper()}")
    print(f"- Highest confidence rate: {overall_stats[best_strategy]['stats']['high_confidence_rate']:.1%}")
    print(f"- Average score: {overall_stats[best_strategy]['stats']['average_score']:.3f}")


def process_single_strategy_all_recipes(strategy='adaptive'):
    """Alternative function to process all recipes with just one strategy"""

    foods_db_path = FINAL_FOOD_DATASET
    foodb_path = FOOD_JSON

    matcher = HybridIngredientMatcher(foods_db_path=foods_db_path, foodb_path=foodb_path)
    recipe_file_path = RECIPES

    print(f"PROCESSING ALL RECIPES WITH {strategy.upper()} STRATEGY")
    print("=" * 60)

    # Load recipes from single JSON file
    #try:
    with open(recipe_file_path, 'r', encoding='utf-8') as file:
        recipes_data = json.load(file)
    print(f"Loaded {len(recipes_data)} recipes from {recipe_file_path}")
    # except Exception as e:
    #     print(f"Error loading recipes: {e}")
    #     return []

    all_recipe_results = []
    total_processed = 0

    start_time = time.time()

    for recipe in recipes_data:
        #try:
            # Get ingredients list
            ingredients = recipe['ingredients']

            # Handle nested ingredient format if needed
            if ingredients and isinstance(ingredients[0], list):
                ingredients = [item[0] if isinstance(item, list) else item for item in ingredients]

            results = matcher.match_ingredients_batch(ingredients, strategy)

            # Create normalized ingredient list
            normalized_ingredients = [r.matched for r in results]

            all_recipe_results.append({
                'id': recipe.get('id', total_processed),
                'cuisine': recipe.get('cuisine', 'unknown'),
                'title': recipe.get('title', f'Recipe_{total_processed}'),
                'original_ingredients': ingredients,
                'normalized_ingredients': normalized_ingredients
            })

            total_processed += 1

            if total_processed % 2500 == 0:
                print(f"  Processed {total_processed} recipes...")

        # except Exception as e:
        #     print(f"Error processing recipe {total_processed}: {e}")
        #     continue

    elapsed_time = time.time() - start_time

    # Save results
    output_file = f'all_recipes_normalized_{strategy}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_recipe_results, f, indent=2, ensure_ascii=False)

    print("\nCOMPLETED!")
    print(f"Total recipes processed: {total_processed}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Results saved to: {output_file}")

    return all_recipe_results


if __name__ == "__main__":
    create_real_pipeline()
