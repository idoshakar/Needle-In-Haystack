import json
import os
import pandas as pd
import re
import time
import glob
import duckdb
import spellchecker
from typing import Dict, List, Tuple, Optional
from rapidfuzz import fuzz
from dataclasses import dataclass
import warnings

warnings.filterwarnings('ignore')

# Initialize spell checker ONCE globally
spell = spellchecker.SpellChecker('en')
spell.distance = 1


def correct(sentence: str) -> str:
    """Spell correction function"""
    words = sentence.split()
    corrected_words = []
    for word in words:
        correction = spell.correction(word)
        if correction is None:
            return sentence
        corrected_words.append(correction)
    return ' '.join(corrected_words)


@dataclass
class ClusterMatchResult:
    """Container for cluster match results"""
    original: str
    cleaned: str
    matched_food: str
    matched_cluster_id: int
    score: float
    method: str
    confidence: str
    metadata: Dict = None


class EnhancedOptimizedClusteredMatcher:
    """
    Highly optimized version with full functionality:
    - Database clustering
    - Fuzzy matching fallback
    - Multiple strategies
    - Comprehensive analysis
    """

    def __init__(self, foods_db_path: str = None,
                 clustered_ingredients_path: str = 'output_table.json',
                 fuzzy_threshold: float = 0.8,
                 db_threshold: float = 0.7):
        """Initialize with full fuzzy + database support"""

        self.fuzzy_threshold = fuzzy_threshold
        self.db_threshold = db_threshold
        self.clustered_ingredients_path = clustered_ingredients_path

        # In-memory caches
        self.db_cache = {}  # Persistent cache for database searches
        self.fuzzy_cache = {}  # Cache for fuzzy matches
        self.clean_cache = {}  # Cache for cleaned ingredients

        # Compile regex patterns FIRST (before using them in other methods)
        self._compile_regex_patterns()

        # Initialize persistent database connection
        self._init_persistent_db_connection()

        # Load fuzzy database (RESTORED FUNCTIONALITY)
        self._load_fuzzy_database(foods_db_path)

    def _compile_regex_patterns(self):
        """Pre-compile all regex patterns for better performance"""
        self.unicode_pattern = re.compile(r'[^\w\s]', flags=re.UNICODE)
        self.number_pattern = re.compile(r'[0-9]')
        self.beaten_pattern = re.compile(r'\bbeaten\b')
        self.word_pattern = re.compile(r'\b\w+\b')

        # Pre-compile word removal patterns
        words_to_remove = ['aged', 'frozen', 'fresh', 'organic', 'chopped', 'ground',
                           'cubed', 'sliced', 'cooked', 'roasted', 'dried', 'fried',
                           'fully', 'flavored', 'unflavored', 'grated', 'cold', 'powdered', 'plain']
        self.removal_patterns = [(word, re.compile(rf'\b{word}\b')) for word in words_to_remove]

    def _init_persistent_db_connection(self):
        """Initialize a persistent database connection that stays open"""
        try:
            self.con = duckdb.connect(':memory:')
            self.con.execute('''
                INSTALL fts;
                LOAD fts;
                SET preserve_insertion_order=false;
                SET threads TO 4;  -- Use multiple threads
                SET memory_limit='4GB';  -- Increase memory limit
            ''')

            print(f"Loading database from {self.clustered_ingredients_path}...")
            start_time = time.time()

            # Load data with optimizations
            self.con.execute(f'''
                CREATE TABLE ingredients AS
                    SELECT cluster_id, fdc_id, food_name,
                           LOWER(food_name) as food_name_lower  -- Pre-compute lowercase
                        FROM read_json_auto('{self.clustered_ingredients_path}');
            ''')

            # Create optimized indexes
            self.con.execute('''CREATE INDEX idx_cluster_id ON ingredients(cluster_id);''')
            self.con.execute('''CREATE INDEX idx_food_name_lower ON ingredients(food_name_lower);''')

            # Create FTS index
            self.con.execute('''PRAGMA create_fts_index(
                'main.ingredients', 'fdc_id', 'food_name'
            );''')

            load_time = time.time() - start_time
            print(f"Database loaded in {load_time:.2f} seconds")

            # Get ingredient count
            count_result = self.con.execute("SELECT COUNT(*) FROM ingredients").fetchone()
            print(f"Loaded {count_result[0]:,} ingredients")

        except Exception as e:
            print(f"Database initialization failed: {e}")
            self.con = None

    def _load_fuzzy_database(self, foods_db_path):
        """Load fuzzy database once and optimize (RESTORED FROM compare.py)"""
        if foods_db_path:
            csv_files = glob.glob(foods_db_path)
            if csv_files:
                print("Loading fuzzy matching database...")
                df_list = []
                for file in csv_files:
                    # Read only the 'food' column if it exists
                    df = pd.read_csv(file)
                    if 'food' in df.columns:
                        df = df[['food']]
                    df_list.append(df)

                df_combined = pd.concat(df_list, ignore_index=True)

                # Clean and deduplicate
                df_combined['food'] = df_combined['food'].astype(str).str.lower().str.strip()
                df_combined = df_combined.drop_duplicates()

                # Apply fuzzy cleaning to ingredients
                cleaned_ingredients = [self.clean_ingredient_fuzzy(ing) for ing in df_combined['food']]
                self.ingredient_database = [ing.replace("suger", "sugar") for ing in cleaned_ingredients]
                self.exact_lookup = set(self.ingredient_database)
                print(f"Loaded {len(self.ingredient_database)} fuzzy ingredients")
            else:
                print("No fuzzy CSV files found")
                self.ingredient_database = []
                self.exact_lookup = set()
        else:
            print("Warning: No foods_db_path provided, fuzzy matching disabled")
            self.ingredient_database = []
            self.exact_lookup = set()

    def clean_ingredient_optimized(self, ingredient: str, for_db: bool = False) -> str:
        """Highly optimized ingredient cleaning with caching"""
        # Check cache first
        cache_key = f"{ingredient}_{for_db}"
        if cache_key in self.clean_cache:
            return self.clean_cache[cache_key]

        if not isinstance(ingredient, str) or not ingredient.strip():
            self.clean_cache[cache_key] = ""
            return ""

        # Replace hyphens with spaces
        cleaned = ingredient.replace('-', ' ')

        # Remove punctuation and numbers in one pass
        cleaned = self.unicode_pattern.sub('', cleaned)
        cleaned = self.number_pattern.sub('', cleaned)

        # Convert to lowercase and normalize whitespace
        cleaned = ' '.join(cleaned.split()).lower()

        # Special egg handling
        if 'egg' in cleaned:
            cleaned = self.beaten_pattern.sub('', cleaned)

        # Remove descriptor words using pre-compiled patterns
        for word, pattern in self.removal_patterns:
            cleaned = pattern.sub('', cleaned)

        # Final whitespace cleanup
        cleaned = ' '.join(cleaned.split())

        # Cache result
        self.clean_cache[cache_key] = cleaned
        return cleaned

    def clean_ingredient_fuzzy(self, ingredient: str) -> str:
        """Clean ingredient for fuzzy matching (from compare.py)"""
        return self.clean_ingredient_optimized(ingredient, for_db=False)

    def clean_ingredient_db(self, ingredient: str) -> str:
        """Clean ingredient for database search (from compare.py)"""
        return self.clean_ingredient_optimized(ingredient, for_db=True)

    def exact_match(self, cleaned_ingredient: str) -> Optional[str]:
        """Check for exact matches in fuzzy database (RESTORED)"""
        if cleaned_ingredient in self.exact_lookup:
            return cleaned_ingredient
        return None

    def database_search_optimized(self, ingredient: str) -> Tuple[int, str, float]:
        """Optimized database search with persistent connection"""
        # Check cache first
        if ingredient in self.db_cache:
            return self.db_cache[ingredient]

        if not self.con:
            return -1, ingredient, 0.0

        try:
            cleaned = self.clean_ingredient_optimized(ingredient, for_db=True)

            # Spell correction
            corrected_ingredient = correct(cleaned)

            # Try exact match first (fastest)
            exact_result = self.con.execute(f'''
                SELECT cluster_id, food_name, 1.0 as score
                FROM ingredients
                WHERE food_name_lower = '{cleaned.replace("'", "''")}'
                ORDER BY cluster_id ASC
                LIMIT 1
            ''').fetchone()

            if exact_result:
                result = (int(exact_result[0]), exact_result[1], 1.0)
                self.db_cache[ingredient] = result
                return result

            # Try fuzzy search with BM25
            search_term = cleaned.replace("'", "''")
            fuzzy_results = self.con.execute(f'''
                SELECT cluster_id, food_name, score
                FROM (
                    SELECT cluster_id, food_name,
                           fts_main_ingredients.match_bm25(
                               CAST(fdc_id AS VARCHAR),
                               '{search_term}',
                               k:=1.2, b:=0.0
                           ) as score
                    FROM ingredients
                ) sq
                WHERE score IS NOT NULL AND score > 0
                ORDER BY score DESC, cluster_id ASC
                LIMIT 5
            ''').fetchall()

            # If original didn't work and we have a correction, try corrected version
            if not fuzzy_results and corrected_ingredient != cleaned:
                corrected_term = corrected_ingredient.replace("'", "''")
                fuzzy_results = self.con.execute(f'''
                    SELECT cluster_id, food_name, score
                    FROM (
                        SELECT cluster_id, food_name,
                               fts_main_ingredients.match_bm25(
                                   CAST(fdc_id AS VARCHAR),
                                   '{corrected_term}',
                                   k:=1.2, b:=0.0
                               ) as score
                        FROM ingredients
                    ) sq
                    WHERE score IS NOT NULL AND score > 0
                    ORDER BY score DESC, cluster_id ASC
                    LIMIT 5
                ''').fetchall()
                if fuzzy_results:
                    print(f"original: {ingredient}, corrected: {corrected_ingredient}")

            if fuzzy_results:
                best = fuzzy_results[0]
                # Calculate confidence based on position in results and score
                confidence = 1.0 if len(fuzzy_results) == 1 else max(0.7, 1.0 - (0.1 * min(3, len(fuzzy_results))))
                confidence = min(confidence, best[2])  # Cap by actual BM25 score
                result = (int(best[0]), best[1], confidence)
                self.db_cache[ingredient] = result
                return result

        except Exception as e:
            print(f"Database search failed for '{ingredient}': {e}")

        # No match found
        result = (-1, ingredient, 0.0)
        self.db_cache[ingredient] = result
        return result

    def fuzzy_match_optimized(self, cleaned_ingredient: str) -> List[Tuple[str, float]]:
        """Enhanced fuzzy matching with hierarchical scoring (RESTORED FROM compare.py)"""
        # Check cache
        if cleaned_ingredient in self.fuzzy_cache:
            return self.fuzzy_cache[cleaned_ingredient]

        if not self.ingredient_database or not cleaned_ingredient:
            return []

        # Check exact match first
        if cleaned_ingredient in self.exact_lookup:
            result = [(cleaned_ingredient, 1.0)]
            self.fuzzy_cache[cleaned_ingredient] = result
            return result

        # Pre-compute words for target
        target_words = set(self.word_pattern.findall(cleaned_ingredient))
        if not target_words:
            return []

        results = []

        # Enhanced fuzzy matching with hierarchical scoring (from compare.py)
        for candidate in self.ingredient_database[:5000]:  # Limit for performance
            candidate_words = set(self.word_pattern.findall(candidate))
            if not candidate_words:
                continue

            # Quick word overlap check
            common_words = target_words.intersection(candidate_words)
            if not common_words:
                continue  # Skip if no common words

            # Calculate enhanced score
            base_score = fuzz.ratio(cleaned_ingredient, candidate) / 100.0
            if base_score < 0.3:  # Skip very low scores
                continue

            num_common_words = len(common_words)

            # Hierarchical scoring (from compare.py)
            if num_common_words >= 3:
                word_ratio = num_common_words / max(len(target_words), len(candidate_words))
                enhanced_score = 0.6 + (word_ratio * 0.1) + (base_score * 0.2)
            elif num_common_words == 2:
                word_ratio = num_common_words / max(len(target_words), len(candidate_words))
                enhanced_score = 0.4 + (word_ratio * 0.1) + (base_score * 0.3)
            elif num_common_words == 1:
                word_ratio = 1.0 / max(len(target_words), len(candidate_words))
                enhanced_score = 0.2 + (word_ratio * 0.1) + (base_score * 0.4)
            else:
                enhanced_score = base_score * 0.6

            # Exact match bonuses (from compare.py)
            exact_bonus = 0.0
            if candidate_words.issubset(target_words):
                if len(candidate_words) == len(target_words):
                    exact_bonus = 0.5  # Perfect match
                else:
                    exact_bonus = 0.3  # Target subset of query
            elif target_words.issubset(candidate_words):
                exact_bonus = 0.1  # Query subset of target

            final_score = min(enhanced_score + exact_bonus, 1.0)
            results.append((candidate, final_score))

        # Sort and cache top results only
        results = sorted(results, key=lambda x: x[1], reverse=True)[:10]
        self.fuzzy_cache[cleaned_ingredient] = results
        return results

    def get_confidence_level(self, score: float, method: str) -> str:
        """Determine confidence level based on score and method (from compare.py)"""
        if method == 'exact':
            return 'very_high'
        elif score >= 0.9:
            return 'high'
        elif score >= 0.7:
            return 'medium'
        elif score >= 0.5:
            return 'low'
        else:
            return 'very_low'

    def match_ingredient_with_strategy(self, ingredient: str, strategy: str = 'adaptive') -> ClusterMatchResult:
        """
        Main matching function with multiple strategies (RESTORED FROM compare.py)
        """
        if strategy == 'adaptive':
            return self._adaptive_match(ingredient)
        elif strategy == 'db_first':
            return self._db_first_match(ingredient)
        elif strategy == 'fuzzy_first':
            return self._fuzzy_first_match(ingredient)
        elif strategy == 'best_of_both':
            return self._best_of_both_match(ingredient)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    def _adaptive_match(self, ingredient: str) -> ClusterMatchResult:
        """Adaptive strategy using clustered database + fuzzy fallback"""
        cleaned_fuzzy = self.clean_ingredient_fuzzy(ingredient)
        cleaned_db = self.clean_ingredient_db(ingredient)

        # Try exact match first (if fuzzy database available)
        if self.ingredient_database:
            exact = self.exact_match(cleaned_fuzzy)
            if exact:
                return ClusterMatchResult(
                    original=ingredient,
                    cleaned=cleaned_fuzzy,
                    matched_food=exact,
                    matched_cluster_id=-1,  # No cluster for exact fuzzy matches
                    score=1.0,
                    method='exact',
                    confidence='very_high'
                )

        # Strategy selection based on ingredient characteristics
        words = cleaned_fuzzy.split()

        # For single words or common ingredient types, prefer database search
        if len(words) <= 1 or any(word in ['oil', 'sauce', 'cheese', 'milk', 'vinegar', 'sugar'] for word in words):
            cluster_id, matched_food, db_score = self.database_search_optimized(ingredient)
            if cluster_id != -1 and db_score >= self.db_threshold:
                return ClusterMatchResult(
                    original=ingredient,
                    cleaned=cleaned_db,
                    matched_food=matched_food,
                    matched_cluster_id=cluster_id,
                    score=db_score,
                    method='cluster_database',
                    confidence=self.get_confidence_level(db_score, 'database')
                )

        # For complex ingredients or if database didn't succeed, use fuzzy matching
        if self.ingredient_database:
            fuzzy_results = self.fuzzy_match_optimized(cleaned_fuzzy)
            if fuzzy_results and fuzzy_results[0][1] >= self.fuzzy_threshold:
                return ClusterMatchResult(
                    original=ingredient,
                    cleaned=cleaned_fuzzy,
                    matched_food=fuzzy_results[0][0],
                    matched_cluster_id=-1,  # No cluster for fuzzy matches
                    score=fuzzy_results[0][1],
                    method='fuzzy',
                    confidence=self.get_confidence_level(fuzzy_results[0][1], 'fuzzy')
                )

        # If we haven't tried database yet, try it now
        cluster_id, matched_food, db_score = self.database_search_optimized(ingredient)

        # Fallback to best available match
        all_results = []
        if cluster_id != -1 and db_score > 0:
            all_results.append((matched_food, db_score, 'cluster_database', cluster_id))

        if self.ingredient_database:
            fuzzy_results = self.fuzzy_match_optimized(cleaned_fuzzy)
            if fuzzy_results:
                for fuzzy_match, fuzzy_score in fuzzy_results[:3]:
                    all_results.append((fuzzy_match, fuzzy_score, 'fuzzy', -1))

        if all_results:
            best = max(all_results, key=lambda x: x[1])
            return ClusterMatchResult(
                original=ingredient,
                cleaned=cleaned_fuzzy if best[2] == 'fuzzy' else cleaned_db,
                matched_food=best[0],
                matched_cluster_id=best[3],
                score=best[1],
                method=f'{best[2]}_fallback',
                confidence=self.get_confidence_level(best[1], best[2])
            )

        # No good match found
        return ClusterMatchResult(
            original=ingredient,
            cleaned=cleaned_fuzzy,
            matched_food=cleaned_fuzzy,
            matched_cluster_id=-1,
            score=0.0,
            method='no_match',
            confidence='very_low'
        )

    def _best_of_both_match(self, ingredient: str) -> ClusterMatchResult:
        """Run both methods and return the best result"""
        cleaned_fuzzy = self.clean_ingredient_fuzzy(ingredient)
        cleaned_db = self.clean_ingredient_db(ingredient)

        # Try exact match first
        if self.ingredient_database:
            exact = self.exact_match(cleaned_fuzzy)
            if exact:
                return ClusterMatchResult(
                    original=ingredient,
                    cleaned=cleaned_fuzzy,
                    matched_food=exact,
                    matched_cluster_id=-1,
                    score=1.0,
                    method='exact',
                    confidence='very_high'
                )

        # Run both methods
        cluster_id, matched_food, db_score = self.database_search_optimized(ingredient)
        fuzzy_results = self.fuzzy_match_optimized(cleaned_fuzzy)
        fuzzy_score = fuzzy_results[0][1] if fuzzy_results else 0.0

        # Choose the better method
        if cluster_id != -1 and db_score >= fuzzy_score and db_score >= self.db_threshold:
            return ClusterMatchResult(
                original=ingredient,
                cleaned=cleaned_db,
                matched_food=matched_food,
                matched_cluster_id=cluster_id,
                score=db_score,
                method='cluster_database',
                confidence=self.get_confidence_level(db_score, 'database'),
                metadata={'fuzzy_score': fuzzy_score}
            )
        elif fuzzy_results and fuzzy_score >= self.fuzzy_threshold:
            return ClusterMatchResult(
                original=ingredient,
                cleaned=cleaned_fuzzy,
                matched_food=fuzzy_results[0][0],
                matched_cluster_id=-1,
                score=fuzzy_score,
                method='fuzzy',
                confidence=self.get_confidence_level(fuzzy_score, 'fuzzy'),
                metadata={'cluster_score': db_score}
            )
        else:
            # Return best available
            if cluster_id != -1 and db_score >= fuzzy_score:
                return ClusterMatchResult(
                    original=ingredient,
                    cleaned=cleaned_db,
                    matched_food=matched_food,
                    matched_cluster_id=cluster_id,
                    score=db_score,
                    method='cluster_database_low_conf',
                    confidence='low'
                )
            elif fuzzy_results:
                return ClusterMatchResult(
                    original=ingredient,
                    cleaned=cleaned_fuzzy,
                    matched_food=fuzzy_results[0][0],
                    matched_cluster_id=-1,
                    score=fuzzy_score,
                    method='fuzzy_low_conf',
                    confidence='low'
                )
            else:
                return ClusterMatchResult(
                    original=ingredient,
                    cleaned=cleaned_fuzzy,
                    matched_food=cleaned_fuzzy,
                    matched_cluster_id=-1,
                    score=0.0,
                    method='no_match',
                    confidence='very_low'
                )

    def _db_first_match(self, ingredient: str) -> ClusterMatchResult:
        """Try clustered database method first, fallback to fuzzy"""
        cleaned_db = self.clean_ingredient_db(ingredient)
        cleaned_fuzzy = self.clean_ingredient_fuzzy(ingredient)

        if self.ingredient_database:
            exact = self.exact_match(cleaned_fuzzy)
            if exact:
                return ClusterMatchResult(
                    original=ingredient,
                    cleaned=cleaned_fuzzy,
                    matched_food=exact,
                    matched_cluster_id=-1,
                    score=1.0,
                    method='exact',
                    confidence='very_high'
                )

        cluster_id, matched_food, db_score = self.database_search_optimized(ingredient)
        if cluster_id != -1 and db_score >= self.db_threshold:
            return ClusterMatchResult(
                original=ingredient,
                cleaned=cleaned_db,
                matched_food=matched_food,
                matched_cluster_id=cluster_id,
                score=db_score,
                method='cluster_database',
                confidence=self.get_confidence_level(db_score, 'database')
            )

        # Fallback to fuzzy
        if self.ingredient_database:
            fuzzy_results = self.fuzzy_match_optimized(cleaned_fuzzy)
            if fuzzy_results:
                return ClusterMatchResult(
                    original=ingredient,
                    cleaned=cleaned_fuzzy,
                    matched_food=fuzzy_results[0][0],
                    matched_cluster_id=-1,
                    score=fuzzy_results[0][1],
                    method='fuzzy_fallback',
                    confidence=self.get_confidence_level(fuzzy_results[0][1], 'fuzzy')
                )

        return ClusterMatchResult(
            original=ingredient,
            cleaned=cleaned_fuzzy,
            matched_food=cleaned_fuzzy,
            matched_cluster_id=-1,
            score=0.0,
            method='no_match',
            confidence='very_low'
        )

    def _fuzzy_first_match(self, ingredient: str) -> ClusterMatchResult:
        """Try fuzzy method first, fallback to clustered database"""
        cleaned_fuzzy = self.clean_ingredient_fuzzy(ingredient)
        cleaned_db = self.clean_ingredient_db(ingredient)

        if self.ingredient_database:
            exact = self.exact_match(cleaned_fuzzy)
            if exact:
                return ClusterMatchResult(
                    original=ingredient,
                    cleaned=cleaned_fuzzy,
                    matched_food=exact,
                    matched_cluster_id=-1,
                    score=1.0,
                    method='exact',
                    confidence='very_high'
                )

            fuzzy_results = self.fuzzy_match_optimized(cleaned_fuzzy)
            if fuzzy_results and fuzzy_results[0][1] >= self.fuzzy_threshold:
                return ClusterMatchResult(
                    original=ingredient,
                    cleaned=cleaned_fuzzy,
                    matched_food=fuzzy_results[0][0],
                    matched_cluster_id=-1,
                    score=fuzzy_results[0][1],
                    method='fuzzy',
                    confidence=self.get_confidence_level(fuzzy_results[0][1], 'fuzzy')
                )

        # Fallback to database
        cluster_id, matched_food, db_score = self.database_search_optimized(ingredient)
        if cluster_id != -1 and db_score > 0:
            return ClusterMatchResult(
                original=ingredient,
                cleaned=cleaned_db,
                matched_food=matched_food,
                matched_cluster_id=cluster_id,
                score=db_score,
                method='cluster_database_fallback',
                confidence=self.get_confidence_level(db_score, 'database')
            )

        return ClusterMatchResult(
            original=ingredient,
            cleaned=cleaned_fuzzy,
            matched_food=cleaned_fuzzy,
            matched_cluster_id=-1,
            score=0.0,
            method='no_match',
            confidence='very_low'
        )

    def match_ingredient_fast(self, ingredient: str) -> ClusterMatchResult:
        """Fast ingredient matching (for backwards compatibility)"""
        return self.match_ingredient_with_strategy(ingredient, 'adaptive')

    def match_ingredients_batch_optimized(self, ingredients: List[str],
                                          strategy: str = 'adaptive') -> List[ClusterMatchResult]:
        """Optimized batch processing with strategy support"""
        results = []

        # Process in chunks to avoid memory issues
        chunk_size = 100
        for i in range(0, len(ingredients), chunk_size):
            chunk = ingredients[i:i + chunk_size]

            for ingredient in chunk:
                result = self.match_ingredient_with_strategy(ingredient, strategy)
                results.append(result)

        return results

    def get_statistics(self, results: List[ClusterMatchResult]) -> Dict:
        """Generate comprehensive statistics from match results (RESTORED)"""
        total = len(results)
        methods = {}
        confidences = {}
        clustered_count = 0

        for result in results:
            methods[result.method] = methods.get(result.method, 0) + 1
            confidences[result.confidence] = confidences.get(result.confidence, 0) + 1
            if result.matched_cluster_id != -1:
                clustered_count += 1

        high_confidence = sum(1 for r in results if r.confidence in ['very_high', 'high'])
        avg_score = sum(r.score for r in results) / total if total > 0 else 0

        return {
            'total_ingredients': total,
            'clustered_ingredients': clustered_count,
            'clustering_rate': clustered_count / total if total > 0 else 0,
            'methods_used': methods,
            'confidence_levels': confidences,
            'high_confidence_rate': high_confidence / total if total > 0 else 0,
            'average_score': avg_score
        }

    def __del__(self):
        """Clean up database connection"""
        if hasattr(self, 'con') and self.con:
            self.con.close()


def process_recipes_optimized_comprehensive(strategy='adaptive', max_recipes=None):
    """
    ENHANCED processing function with full functionality from compare.py
    Supports all strategies: 'adaptive', 'best_of_both', 'db_first', 'fuzzy_first'
    """

    print("Initializing enhanced optimized matcher...")
    foods_db_path = "C://Users//User//Desktop//needle_in_haystack//NeadleInHaystack//archive//FINAL FOOD DATASET//FOOD-DATA-GROUP*.csv"
    clustered_ingredients_path = 'output_table.json'

    matcher = EnhancedOptimizedClusteredMatcher(
        foods_db_path=foods_db_path,
        clustered_ingredients_path=clustered_ingredients_path
    )

    recipe_file_path = '../whats-cooking/train.json/train.json'

    print(f"ENHANCED OPTIMIZED PROCESSING WITH {strategy.upper()} STRATEGY")
    print("=" * 60)

    # Load recipes
    try:
        with open(recipe_file_path, 'r', encoding='utf-8') as file:
            recipes_data = json.load(file)
        print(f"Loaded {len(recipes_data)} recipes")
    except Exception as e:
        print(f"Error loading recipes: {e}")
        return []

    # Limit recipes for testing if specified
    if max_recipes:
        recipes_data = recipes_data[:max_recipes]
        print(f"Processing first {max_recipes} recipes for testing")

    all_recipe_results = []
    total_processed = 0
    total_ingredients_processed = 0

    start_time = time.time()
    last_report_time = start_time

    print("\nStarting processing...")

    for recipe in recipes_data:
        try:
            if 'ingredients' not in recipe:
                continue

            ingredients = recipe['ingredients']
            if not ingredients:
                continue

            # Batch process all ingredients for this recipe with strategy
            results = matcher.match_ingredients_batch_optimized(ingredients, strategy)

            # Store results
            all_recipe_results.append({
                'recipe_id': recipe.get('id', str(total_processed)),
                'recipe_title': recipe.get('title', f'Recipe_{total_processed}'),
                'cuisine': recipe.get('cuisine', 'unknown'),
                'original_ingredients': ingredients,
                'matched_ingredients': [r.matched_cluster_id if r.matched_cluster_id != -1 else r.matched_food for r in
                                        results],
                'scores': [r.score for r in results],
                'methods': [r.method for r in results],
                'confidences': [r.confidence for r in results]
            })

            total_processed += 1
            total_ingredients_processed += len(ingredients)

            # Progress reporting every 1000 recipes or 10 seconds
            current_time = time.time()
            if total_processed % 1000 == 0 or (current_time - last_report_time) >= 10:
                elapsed = current_time - start_time
                rate = total_processed / elapsed if elapsed > 0 else 0
                print(
                    f"  Processed {total_processed:,} recipes ({total_ingredients_processed:,} ingredients) - {rate:.1f} recipes/sec")
                last_report_time = current_time

        except Exception as e:
            print(f"Error processing recipe {total_processed}: {e}")
            continue

    elapsed_time = time.time() - start_time

    # Save results
    output_file = f'enhanced_recipes_clustered_{strategy}.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(all_recipe_results, f, indent=2, ensure_ascii=False)

    # Print final statistics
    print(f"\n{'=' * 60}")
    print(f"PROCESSING COMPLETED!")
    print(f"{'=' * 60}")
    print(f"Total recipes processed: {total_processed:,}")
    print(f"Total ingredients processed: {total_ingredients_processed:,}")
    print(f"Processing time: {elapsed_time:.2f} seconds")
    print(f"Average rate: {total_processed / elapsed_time:.1f} recipes/sec")
    print(f"Average rate: {total_ingredients_processed / elapsed_time:.1f} ingredients/sec")
    print(f"Results saved to: {output_file}")

    # Calculate comprehensive statistics
    if total_ingredients_processed > 0:
        # Create flat results list for statistics
        all_results_flat = []
        for recipe_result in all_recipe_results:
            for i, original in enumerate(recipe_result['original_ingredients']):
                matched_ingredient = recipe_result['matched_ingredients'][i]
                # Determine cluster_id: if it's a number, use it; otherwise it's a food name (cluster_id = -1)
                cluster_id = matched_ingredient if isinstance(matched_ingredient, int) else -1
                matched_food = matched_ingredient if isinstance(matched_ingredient,
                                                                str) else f"cluster_{matched_ingredient}"

                all_results_flat.append(ClusterMatchResult(
                    original=original,
                    cleaned="",  # Not needed for stats
                    matched_food=matched_food,
                    matched_cluster_id=cluster_id,
                    score=recipe_result['scores'][i],
                    method=recipe_result['methods'][i],
                    confidence=recipe_result['confidences'][i]
                ))

        stats = matcher.get_statistics(all_results_flat)

        print(f"\nCOMPREHENSIVE STATISTICS:")
        print(f"Successfully clustered: {stats['clustered_ingredients']:,}")
        print(f"Clustering rate: {stats['clustering_rate']:.1%}")
        print(f"High confidence rate: {stats['high_confidence_rate']:.1%}")
        print(f"Average score: {stats['average_score']:.3f}")

        # Method statistics
        print(f"\nMETHOD BREAKDOWN:")
        for method, count in sorted(stats['methods_used'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_ingredients_processed) * 100
            print(f"  {method:<25} {count:>8,} ({percentage:>5.1f}%)")

        # Confidence statistics
        print(f"\nCONFIDENCE BREAKDOWN:")
        for confidence, count in sorted(stats['confidence_levels'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_ingredients_processed) * 100
            print(f"  {confidence:<15} {count:>8,} ({percentage:>5.1f}%)")

    return all_recipe_results


# def process_all_strategies_comprehensive():
#     """Process recipes with ALL strategies for comprehensive comparison (RESTORED FROM compare.py)"""
#
#     strategies = ['adaptive', 'best_of_both', 'db_first', 'fuzzy_first']
#
#     foods_db_path = "C://Users//User//Desktop//needle_in_haystack//NeadleInHaystack//archive//FINAL FOOD DATASET//FOOD-DATA-GROUP*.csv"
#     clustered_ingredients_path = 'output_table.json'
#
#     # Initialize matcher once
#     matcher = EnhancedOptimizedClusteredMatcher(
#         foods_db_path=foods_db_path,
#         clustered_ingredients_path=clustered_ingredients_path
#     )
#
#     recipe_file_path = '../whats-cooking/train.json/train.json'
#
#     print("\nPROCESSING ALL RECIPES WITH ALL METHODS (ENHANCED)")
#     print("=" * 60)
#
#     # Load recipes once
#     try:
#         with open(recipe_file_path, 'r', encoding='utf-8') as file:
#             recipes_data = json.load(file)
#         print(f"Loaded {len(recipes_data)} recipes from file")
#     except Exception as e:
#         print(f"ERROR loading recipe file: {e}")
#         return
#
#     all_results = {}
#     overall_stats = {}
#
#     for strategy in strategies:
#         print(f"\nProcessing with strategy: {strategy.upper()}")
#         print("-" * 40)
#
#         strategy_results = []
#         total_ingredients = 0
#         total_recipes = 0
#
#         start_time = time.time()
#
#         # Process all recipes in the list
#         for recipe_idx, recipe in enumerate(recipes_data[:10]):
#             try:
#                 recipe_title = recipe.get('title', recipe.get('id', f'Recipe_{recipe_idx}'))
#
#                 if 'ingredients' not in recipe:
#                     print(f"    WARNING: Recipe {recipe_idx} has no ingredients")
#                     continue
#
#                 # Get ingredients list
#                 ingredients = recipe['ingredients']
#
#                 # Handle case where ingredients might be nested
#                 if ingredients and isinstance(ingredients[0], list):
#                     # If ingredients are in format [["ingredient", "amount"], ...]
#                     ingredients = [item[0] if isinstance(item, list) else item for item in ingredients]
#
#                 print(f"    Recipe '{recipe_title}': {len(ingredients)} ingredients")
#                 if recipe_idx == 0:  # Show first recipe's ingredients
#                     print(f"      Sample ingredients: {ingredients[:3]}")
#
#                 # Match ingredients using current strategy
#                 results = matcher.match_ingredients_batch_optimized(ingredients, strategy)
#
#                 # Store results with cluster IDs as matched ingredients
#                 recipe_result = {
#                     'recipe_id': recipe.get('id', recipe_idx),
#                     'recipe_title': recipe_title,
#                     'cuisine': recipe.get('cuisine', 'unknown'),
#                     'original_ingredients': ingredients,
#                     'matched_ingredients': [r.matched_cluster_id if r.matched_cluster_id != -1 else r.matched_food for r
#                                             in results],
#                     'scores': [r.score for r in results],
#                     'methods': [r.method for r in results],
#                     'confidences': [r.confidence for r in results]
#                 }
#
#                 strategy_results.append(recipe_result)
#                 total_ingredients += len(ingredients)
#                 total_recipes += 1
#
#                 # Progress update every 1000 recipes
#                 if total_recipes % 1000 == 0:
#                     print(f"  Processed {total_recipes} recipes...")
#
#             except Exception as e:
#                 print(f"    ERROR processing recipe {recipe_idx}: {e}")
#                 continue
#
#         elapsed_time = time.time() - start_time
#
#         # Calculate statistics for this strategy
#         all_results_flat = []
#         for recipe_result in strategy_results:
#             for i, original in enumerate(recipe_result['original_ingredients']):
#                 matched_ingredient = recipe_result['matched_ingredients'][i]
#                 # Determine cluster_id: if it's a number, use it; otherwise it's a food name (cluster_id = -1)
#                 cluster_id = matched_ingredient if isinstance(matched_ingredient, int) else -1
#                 matched_food = matched_ingredient if isinstance(matched_ingredient,
#                                                                 str) else f"cluster_{matched_ingredient}"
#
#                 all_results_flat.append(ClusterMatchResult(
#                     original=original,
#                     cleaned="",  # Not needed for stats
#                     matched_food=matched_food,
#                     matched_cluster_id=cluster_id,
#                     score=recipe_result['scores'][i],
#                     method=recipe_result['methods'][i],
#                     confidence=recipe_result['confidences'][i]
#                 ))
#
#         stats = matcher.get_statistics(all_results_flat)
#
#         print(f"\nStrategy: {strategy.upper()} COMPLETED")
#         print(f"Recipes processed: {total_recipes}")
#         print(f"Total ingredients: {total_ingredients}")
#         print(f"Clustered ingredients: {stats['clustered_ingredients']}")
#         print(f"Clustering rate: {stats['clustering_rate']:.1%}")
#         print(f"High confidence rate: {stats['high_confidence_rate']:.1%}")
#         print(f"Average score: {stats['average_score']:.3f}")
#         print(f"Processing time: {elapsed_time:.2f} seconds")
#         print(f"Methods used: {stats['methods_used']}")
#
#         # Store results
#         all_results[strategy] = strategy_results
#         overall_stats[strategy] = {
#             'total_recipes': total_recipes,
#             'total_ingredients': total_ingredients,
#             'processing_time': elapsed_time,
#             'stats': stats
#         }
#
#     # Save all results to files
#     print(f"\nSAVING RESULTS...")
#
#     # Save detailed results for each strategy
#     for strategy, results in all_results.items():
#         output_file = f'enhanced_clustered_ingredient_matching_{strategy}_results.json'
#         try:
#             with open(output_file, 'w', encoding='utf-8') as f:
#                 json.dump(results, f, indent=2, ensure_ascii=False)
#             print(f"Successfully saved {strategy} results to {output_file}")
#         except Exception as e:
#             print(f"ERROR saving {strategy} results: {e}")
#
#     # Save summary statistics
#     try:
#         with open('trials/enhanced_clustered_ingredient_matching_summary.json', 'w', encoding='utf-8') as f:
#             json.dump(overall_stats, f, indent=2)
#         print(f"Successfully saved summary statistics")
#     except Exception as e:
#         print(f"ERROR saving summary: {e}")
#
#     # Create comparison report
#     create_enhanced_comparison_report(overall_stats)
#
#     print(f"\nPROCESSING COMPLETE!")
#     if overall_stats:
#         max_recipes = max(s['total_recipes'] for s in overall_stats.values())
#         print(f"Processed {max_recipes} total recipes")
#     print(f"Check output files for detailed results")
#
#     return overall_stats


def create_enhanced_comparison_report(overall_stats):
    """Create a comprehensive comparison report of all methods (RESTORED FROM compare.py)"""

    print(f"\n" + "=" * 100)
    print("COMPREHENSIVE ENHANCED METHOD COMPARISON REPORT")
    print("=" * 100)

    # Performance comparison table
    print(
        f"\n{'Method':<15} {'Recipes':<8} {'Ingredients':<12} {'Clustered':<10} {'Cluster %':<10} {'Time (s)':<10} {'High Conf %':<12} {'Avg Score':<10}")
    print("-" * 100)

    for strategy, stats in overall_stats.items():
        clustered_count = stats['stats']['clustered_ingredients']
        total_ingredients = stats['stats']['total_ingredients']
        cluster_rate = (clustered_count / total_ingredients * 100) if total_ingredients > 0 else 0

        print(f"{strategy:<15} "
              f"{stats['total_recipes']:<8} "
              f"{stats['total_ingredients']:<12} "
              f"{clustered_count:<10} "
              f"{cluster_rate:<10.1f} "
              f"{stats['processing_time']:<10.2f} "
              f"{stats['stats']['high_confidence_rate'] * 100:<12.1f} "
              f"{stats['stats']['average_score']:<10.3f}")

    # Method usage breakdown
    print(f"\nMETHOD USAGE BREAKDOWN:")
    print("-" * 40)
    for strategy, stats in overall_stats.items():
        print(f"\n{strategy.upper()}:")
        for method, count in stats['stats']['methods_used'].items():
            percentage = (count / stats['total_ingredients']) * 100
            print(f"  {method:<30} {count:>6} ({percentage:>5.1f}%)")

    # Find best performing method for clustering
    best_clustering_strategy = max(overall_stats.keys(),
                                   key=lambda s: overall_stats[s]['stats']['clustering_rate'])

    best_confidence_strategy = max(overall_stats.keys(),
                                   key=lambda s: overall_stats[s]['stats']['high_confidence_rate'])

    print(f"\nRECOMMENDATIONS:")
    print(f"Best clustering strategy: {best_clustering_strategy.upper()}")
    print(f"- Highest clustering rate: {overall_stats[best_clustering_strategy]['stats']['clustering_rate']:.1%}")
    print(f"- Average score: {overall_stats[best_clustering_strategy]['stats']['average_score']:.3f}")

    print(f"\nBest confidence strategy: {best_confidence_strategy.upper()}")
    print(f"- Highest confidence rate: {overall_stats[best_confidence_strategy]['stats']['high_confidence_rate']:.1%}")
    print(f"- Clustering rate: {overall_stats[best_confidence_strategy]['stats']['clustering_rate']:.1%}")

    return best_clustering_strategy, best_confidence_strategy


import json
import time
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List


def create_comprehensive_performance_report(overall_stats: Dict, output_dir: str = 'reports') -> None:
    """
    Creates comprehensive performance reports including:
    1. Detailed statistical comparison
    2. CSV export of key metrics
    3. Visual performance charts
    4. Executive summary
    """

    # Create output directory if it doesn't exist
    import os
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # 1. Generate detailed statistical report
    _create_detailed_statistical_report(overall_stats, output_dir, timestamp)

    # 2. Create CSV performance summary
    _create_csv_performance_summary(overall_stats, output_dir, timestamp)

    # 3. Generate visualization charts
    _create_performance_visualizations(overall_stats, output_dir, timestamp)

    # 4. Create executive summary
    _create_executive_summary(overall_stats, output_dir, timestamp)

    # 5. Create method-specific detailed reports
    _create_method_specific_reports(overall_stats, output_dir, timestamp)

    print(f"\n📊 COMPREHENSIVE PERFORMANCE REPORTS GENERATED")
    print(f"📁 Reports saved to: {output_dir}/")
    print(f"🏷️  Report timestamp: {timestamp}")


def _create_detailed_statistical_report(overall_stats: Dict, output_dir: str, timestamp: str) -> None:
    """Create a detailed text-based statistical report"""

    report_file = f"{output_dir}/detailed_statistical_report_{timestamp}.txt"

    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("=" * 120 + "\n")
        f.write("COMPREHENSIVE INGREDIENT MATCHING PERFORMANCE ANALYSIS\n")
        f.write("=" * 120 + "\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Overall performance table
        f.write("1. OVERALL PERFORMANCE COMPARISON\n")
        f.write("-" * 50 + "\n")

        # Headers
        f.write(f"{'Strategy':<15} {'Recipes':<8} {'Ingredients':<12} {'Clustered':<10} {'Cluster %':<10} ")
        f.write(f"{'Time (s)':<10} {'High Conf %':<12} {'Avg Score':<10} {'Rate (ing/s)':<12}\n")
        f.write("-" * 120 + "\n")

        for strategy, stats in overall_stats.items():
            clustered_count = stats['stats']['clustered_ingredients']
            total_ingredients = stats['stats']['total_ingredients']
            cluster_rate = (clustered_count / total_ingredients * 100) if total_ingredients > 0 else 0
            processing_rate = total_ingredients / stats['processing_time'] if stats['processing_time'] > 0 else 0

            f.write(f"{strategy:<15} ")
            f.write(f"{stats['total_recipes']:<8} ")
            f.write(f"{stats['total_ingredients']:<12} ")
            f.write(f"{clustered_count:<10} ")
            f.write(f"{cluster_rate:<10.1f} ")
            f.write(f"{stats['processing_time']:<10.2f} ")
            f.write(f"{stats['stats']['high_confidence_rate'] * 100:<12.1f} ")
            f.write(f"{stats['stats']['average_score']:<10.3f} ")
            f.write(f"{processing_rate:<12.1f}\n")

        # Method usage breakdown
        f.write("\n\n2. METHOD USAGE BREAKDOWN BY STRATEGY\n")
        f.write("-" * 50 + "\n")

        for strategy, stats in overall_stats.items():
            f.write(f"\n{strategy.upper()} STRATEGY:\n")
            f.write("  " + "-" * 45 + "\n")
            for method, count in sorted(stats['stats']['methods_used'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / stats['total_ingredients']) * 100
                f.write(f"  {method:<35} {count:>8,} ({percentage:>5.1f}%)\n")

        # Confidence distribution
        f.write("\n\n3. CONFIDENCE LEVEL DISTRIBUTION BY STRATEGY\n")
        f.write("-" * 50 + "\n")

        for strategy, stats in overall_stats.items():
            f.write(f"\n{strategy.upper()} STRATEGY:\n")
            f.write("  " + "-" * 35 + "\n")
            for confidence, count in sorted(stats['stats']['confidence_levels'].items(), key=lambda x: x[1],
                                            reverse=True):
                percentage = (count / stats['total_ingredients']) * 100
                f.write(f"  {confidence:<20} {count:>8,} ({percentage:>5.1f}%)\n")

        # Performance ranking
        f.write("\n\n4. PERFORMANCE RANKING ANALYSIS\n")
        f.write("-" * 50 + "\n")

        # Rank by clustering rate
        clustering_ranking = sorted(overall_stats.keys(),
                                    key=lambda s: overall_stats[s]['stats']['clustering_rate'], reverse=True)
        f.write("Ranking by Clustering Rate:\n")
        for i, strategy in enumerate(clustering_ranking, 1):
            rate = overall_stats[strategy]['stats']['clustering_rate']
            f.write(f"  {i}. {strategy:<15} ({rate:.1%})\n")

        # Rank by confidence
        confidence_ranking = sorted(overall_stats.keys(),
                                    key=lambda s: overall_stats[s]['stats']['high_confidence_rate'], reverse=True)
        f.write("\nRanking by High Confidence Rate:\n")
        for i, strategy in enumerate(confidence_ranking, 1):
            rate = overall_stats[strategy]['stats']['high_confidence_rate']
            f.write(f"  {i}. {strategy:<15} ({rate:.1%})\n")

        # Rank by average score
        score_ranking = sorted(overall_stats.keys(),
                               key=lambda s: overall_stats[s]['stats']['average_score'], reverse=True)
        f.write("\nRanking by Average Score:\n")
        for i, strategy in enumerate(score_ranking, 1):
            score = overall_stats[strategy]['stats']['average_score']
            f.write(f"  {i}. {strategy:<15} ({score:.3f})\n")

        # Processing speed ranking
        speed_ranking = sorted(overall_stats.keys(),
                               key=lambda s: overall_stats[s]['total_ingredients'] / overall_stats[s][
                                   'processing_time'],
                               reverse=True)
        f.write("\nRanking by Processing Speed (ingredients/second):\n")
        for i, strategy in enumerate(speed_ranking, 1):
            speed = overall_stats[strategy]['total_ingredients'] / overall_stats[strategy]['processing_time']
            f.write(f"  {i}. {strategy:<15} ({speed:.1f} ing/s)\n")


def _create_csv_performance_summary(overall_stats: Dict, output_dir: str, timestamp: str) -> None:
    """Create CSV summary for easy analysis in Excel/other tools"""

    # Main performance metrics
    performance_data = []
    for strategy, stats in overall_stats.items():
        clustered_count = stats['stats']['clustered_ingredients']
        total_ingredients = stats['stats']['total_ingredients']
        cluster_rate = (clustered_count / total_ingredients) if total_ingredients > 0 else 0
        processing_rate = total_ingredients / stats['processing_time'] if stats['processing_time'] > 0 else 0

        performance_data.append({
            'strategy': strategy,
            'total_recipes': stats['total_recipes'],
            'total_ingredients': stats['total_ingredients'],
            'clustered_ingredients': clustered_count,
            'clustering_rate': cluster_rate,
            'high_confidence_rate': stats['stats']['high_confidence_rate'],
            'average_score': stats['stats']['average_score'],
            'processing_time_seconds': stats['processing_time'],
            'processing_rate_ingredients_per_second': processing_rate
        })

    df_performance = pd.DataFrame(performance_data)
    df_performance.to_csv(f"{output_dir}/performance_summary_{timestamp}.csv", index=False)

    # Method usage breakdown
    method_data = []
    for strategy, stats in overall_stats.items():
        for method, count in stats['stats']['methods_used'].items():
            percentage = (count / stats['total_ingredients']) * 100
            method_data.append({
                'strategy': strategy,
                'method': method,
                'count': count,
                'percentage': percentage
            })

    df_methods = pd.DataFrame(method_data)
    df_methods.to_csv(f"{output_dir}/method_usage_breakdown_{timestamp}.csv", index=False)

    # Confidence levels breakdown
    confidence_data = []
    for strategy, stats in overall_stats.items():
        for confidence, count in stats['stats']['confidence_levels'].items():
            percentage = (count / stats['total_ingredients']) * 100
            confidence_data.append({
                'strategy': strategy,
                'confidence_level': confidence,
                'count': count,
                'percentage': percentage
            })

    df_confidence = pd.DataFrame(confidence_data)
    df_confidence.to_csv(f"{output_dir}/confidence_breakdown_{timestamp}.csv", index=False)


def _create_performance_visualizations(overall_stats: Dict, output_dir: str, timestamp: str) -> None:
    """Create visual performance comparison charts"""

    # Set up the plotting style
    plt.style.use('default')
    sns.set_palette("husl")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Ingredient Matching Performance Comparison', fontsize=16, fontweight='bold')

    strategies = list(overall_stats.keys())

    # 1. Clustering Rate Comparison
    clustering_rates = [overall_stats[s]['stats']['clustering_rate'] * 100 for s in strategies]
    axes[0, 0].bar(strategies, clustering_rates, color='skyblue', edgecolor='navy', linewidth=1.2)
    axes[0, 0].set_title('Clustering Rate by Strategy (%)', fontweight='bold')
    axes[0, 0].set_ylabel('Clustering Rate (%)')
    axes[0, 0].tick_params(axis='x', rotation=45)

    # Add value labels on bars
    for i, v in enumerate(clustering_rates):
        axes[0, 0].text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

    # 2. Processing Speed Comparison
    processing_speeds = [overall_stats[s]['total_ingredients'] / overall_stats[s]['processing_time']
                         for s in strategies]
    axes[0, 1].bar(strategies, processing_speeds, color='lightcoral', edgecolor='darkred', linewidth=1.2)
    axes[0, 1].set_title('Processing Speed by Strategy (ing/s)', fontweight='bold')
    axes[0, 1].set_ylabel('Ingredients per Second')
    axes[0, 1].tick_params(axis='x', rotation=45)

    # Add value labels
    for i, v in enumerate(processing_speeds):
        axes[0, 1].text(i, v + max(processing_speeds) * 0.01, f'{v:.1f}', ha='center', va='bottom', fontweight='bold')

    # 3. High Confidence Rate Comparison
    confidence_rates = [overall_stats[s]['stats']['high_confidence_rate'] * 100 for s in strategies]
    axes[1, 0].bar(strategies, confidence_rates, color='lightgreen', edgecolor='darkgreen', linewidth=1.2)
    axes[1, 0].set_title('High Confidence Rate by Strategy (%)', fontweight='bold')
    axes[1, 0].set_ylabel('High Confidence Rate (%)')
    axes[1, 0].tick_params(axis='x', rotation=45)

    # Add value labels
    for i, v in enumerate(confidence_rates):
        axes[1, 0].text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontweight='bold')

    # 4. Average Score Comparison
    avg_scores = [overall_stats[s]['stats']['average_score'] for s in strategies]
    axes[1, 1].bar(strategies, avg_scores, color='gold', edgecolor='orange', linewidth=1.2)
    axes[1, 1].set_title('Average Matching Score by Strategy', fontweight='bold')
    axes[1, 1].set_ylabel('Average Score')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].set_ylim(0, 1)

    # Add value labels
    for i, v in enumerate(avg_scores):
        axes[1, 1].text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/performance_comparison_charts_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()

    # Create method usage distribution chart
    _create_method_usage_chart(overall_stats, output_dir, timestamp)


def _create_method_usage_chart(overall_stats: Dict, output_dir: str, timestamp: str) -> None:
    """Create detailed method usage distribution charts"""

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Method Usage Distribution by Strategy', fontsize=16, fontweight='bold')

    strategies = list(overall_stats.keys())

    for i, strategy in enumerate(strategies):
        ax = axes[i // 2, i % 2]

        methods = list(overall_stats[strategy]['stats']['methods_used'].keys())
        counts = list(overall_stats[strategy]['stats']['methods_used'].values())

        # Create pie chart
        wedges, texts, autotexts = ax.pie(counts, labels=methods, autopct='%1.1f%%', startangle=90)
        ax.set_title(f'{strategy.upper()} Strategy', fontweight='bold')

        # Improve text readability
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/method_usage_distribution_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()


def _create_executive_summary(overall_stats: Dict, output_dir: str, timestamp: str) -> None:
    """Create executive summary with key findings and recommendations"""

    summary_file = f"{output_dir}/executive_summary_{timestamp}.txt"

    # Calculate rankings
    best_clustering = max(overall_stats.keys(),
                          key=lambda s: overall_stats[s]['stats']['clustering_rate'])
    best_confidence = max(overall_stats.keys(),
                          key=lambda s: overall_stats[s]['stats']['high_confidence_rate'])
    fastest_processing = max(overall_stats.keys(),
                             key=lambda s: overall_stats[s]['total_ingredients'] / overall_stats[s]['processing_time'])
    best_overall_score = max(overall_stats.keys(),
                             key=lambda s: overall_stats[s]['stats']['average_score'])

    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("📊 EXECUTIVE SUMMARY - INGREDIENT MATCHING PERFORMANCE ANALYSIS\n")
        f.write("=" * 80 + "\n")
        f.write(f"📅 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("🎯 KEY FINDINGS:\n")
        f.write("-" * 20 + "\n")

        total_recipes = max(s['total_recipes'] for s in overall_stats.values())
        total_ingredients = max(s['total_ingredients'] for s in overall_stats.values())

        f.write(f"• Dataset Size: {total_recipes:,} recipes, {total_ingredients:,} ingredients processed\n")
        f.write(f"• Strategies Evaluated: {len(overall_stats)} different approaches\n\n")

        f.write("🏆 PERFORMANCE LEADERS:\n")
        f.write("-" * 25 + "\n")
        f.write(f"• Best Clustering Rate: {best_clustering.upper()} ")
        f.write(f"({overall_stats[best_clustering]['stats']['clustering_rate']:.1%})\n")
        f.write(f"• Highest Confidence: {best_confidence.upper()} ")
        f.write(f"({overall_stats[best_confidence]['stats']['high_confidence_rate']:.1%})\n")
        f.write(f"• Fastest Processing: {fastest_processing.upper()} ")
        f.write(
            f"({overall_stats[fastest_processing]['total_ingredients'] / overall_stats[fastest_processing]['processing_time']:.1f} ing/s)\n")
        f.write(f"• Best Average Score: {best_overall_score.upper()} ")
        f.write(f"({overall_stats[best_overall_score]['stats']['average_score']:.3f})\n\n")

        f.write("📈 DETAILED PERFORMANCE BREAKDOWN:\n")
        f.write("-" * 40 + "\n")

        for strategy, stats in overall_stats.items():
            clustering_rate = stats['stats']['clustering_rate']
            confidence_rate = stats['stats']['high_confidence_rate']
            avg_score = stats['stats']['average_score']
            speed = stats['total_ingredients'] / stats['processing_time']

            f.write(f"{strategy.upper()} STRATEGY:\n")
            f.write(f"  • Clustering Rate: {clustering_rate:.1%}\n")
            f.write(f"  • High Confidence Rate: {confidence_rate:.1%}\n")
            f.write(f"  • Average Score: {avg_score:.3f}\n")
            f.write(f"  • Processing Speed: {speed:.1f} ingredients/second\n")
            f.write(f"  • Processing Time: {stats['processing_time']:.2f} seconds\n\n")

        f.write("💡 RECOMMENDATIONS:\n")
        f.write("-" * 20 + "\n")

        if best_clustering == best_confidence == best_overall_score:
            f.write(f"• CLEAR WINNER: {best_clustering.upper()} strategy excels across all key metrics\n")
            f.write("• Recommended for production deployment\n")
        else:
            f.write("• Performance varies by metric - consider use case requirements:\n")
            f.write(f"  - For maximum clustering: Use {best_clustering.upper()}\n")
            f.write(f"  - For highest confidence: Use {best_confidence.upper()}\n")
            f.write(f"  - For fastest processing: Use {fastest_processing.upper()}\n")
            f.write(f"  - For best overall scores: Use {best_overall_score.upper()}\n")

        f.write("\n📋 METHODOLOGY INSIGHTS:\n")
        f.write("-" * 30 + "\n")

        # Analyze method distributions
        most_db_heavy = max(overall_stats.keys(),
                            key=lambda s: overall_stats[s]['stats']['methods_used'].get('cluster_database', 0))
        most_fuzzy_heavy = max(overall_stats.keys(),
                               key=lambda s: overall_stats[s]['stats']['methods_used'].get('fuzzy', 0))

        f.write(f"• Most database-reliant strategy: {most_db_heavy.upper()}\n")
        f.write(f"• Most fuzzy-matching reliant strategy: {most_fuzzy_heavy.upper()}\n")

        f.write("\n📊 Report Components Generated:\n")
        f.write("-" * 40 + "\n")
        f.write("• Detailed statistical analysis (TXT)\n")
        f.write("• Performance metrics (CSV)\n")
        f.write("• Method usage breakdown (CSV)\n")
        f.write("• Confidence distribution (CSV)\n")
        f.write("• Performance comparison charts (PNG)\n")
        f.write("• Method distribution visualizations (PNG)\n")
        f.write("• Strategy-specific detailed reports (TXT)\n")


def _create_method_specific_reports(overall_stats: Dict, output_dir: str, timestamp: str) -> None:
    """Create detailed reports for each individual strategy"""

    for strategy, stats in overall_stats.items():
        report_file = f"{output_dir}/detailed_report_{strategy}_{timestamp}.txt"

        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(f"DETAILED PERFORMANCE REPORT: {strategy.upper()} STRATEGY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

            # Basic metrics
            f.write("📊 BASIC PERFORMANCE METRICS:\n")
            f.write("-" * 35 + "\n")
            f.write(f"Total Recipes Processed: {stats['total_recipes']:,}\n")
            f.write(f"Total Ingredients Processed: {stats['total_ingredients']:,}\n")
            f.write(f"Successfully Clustered: {stats['stats']['clustered_ingredients']:,}\n")
            f.write(f"Clustering Rate: {stats['stats']['clustering_rate']:.2%}\n")
            f.write(f"High Confidence Matches: {stats['stats']['high_confidence_rate']:.2%}\n")
            f.write(f"Average Matching Score: {stats['stats']['average_score']:.4f}\n")
            f.write(f"Total Processing Time: {stats['processing_time']:.2f} seconds\n")
            f.write(
                f"Processing Rate: {stats['total_ingredients'] / stats['processing_time']:.2f} ingredients/second\n\n")

            # Method breakdown
            f.write("🔧 METHOD USAGE BREAKDOWN:\n")
            f.write("-" * 30 + "\n")
            for method, count in sorted(stats['stats']['methods_used'].items(), key=lambda x: x[1], reverse=True):
                percentage = (count / stats['total_ingredients']) * 100
                f.write(f"{method:<35} {count:>8,} ({percentage:>6.2f}%)\n")

            # Confidence distribution
            f.write(f"\n🎯 CONFIDENCE LEVEL DISTRIBUTION:\n")
            f.write("-" * 40 + "\n")
            for confidence, count in sorted(stats['stats']['confidence_levels'].items(), key=lambda x: x[1],
                                            reverse=True):
                percentage = (count / stats['total_ingredients']) * 100
                f.write(f"{confidence:<20} {count:>8,} ({percentage:>6.2f}%)\n")

            # Strategy-specific insights
            f.write(f"\n💡 STRATEGY-SPECIFIC INSIGHTS:\n")
            f.write("-" * 35 + "\n")

            if strategy == 'adaptive':
                f.write("• Adaptive strategy balances database and fuzzy matching\n")
                f.write("• Automatically selects best method based on ingredient characteristics\n")
            elif strategy == 'best_of_both':
                f.write("• Runs both methods and selects best result\n")
                f.write("• Higher computational cost but potentially better accuracy\n")
            elif strategy == 'db_first':
                f.write("• Prioritizes database clustering with fuzzy fallback\n")
                f.write("• Good for ingredients likely to be in clustered database\n")
            elif strategy == 'fuzzy_first':
                f.write("• Prioritizes fuzzy matching with database fallback\n")
                f.write("• Good for complex or uncommon ingredient descriptions\n")


# Enhanced version of process_all_strategies_comprehensive with report generation
def process_all_strategies_comprehensive_with_reports():
    """
    Enhanced version that includes comprehensive performance reporting
    """

    print("Initializing comprehensive analysis with performance reporting...")

    # Run the existing comprehensive analysis
    overall_stats = process_all_strategies_comprehensive()

    if overall_stats:
        print("\n🚀 GENERATING COMPREHENSIVE PERFORMANCE REPORTS...")
        print("=" * 60)

        # Generate all performance reports
        create_comprehensive_performance_report(overall_stats)

        print("\n✅ ANALYSIS AND REPORTING COMPLETED SUCCESSFULLY!")
        print("📁 Check the 'reports/' directory for detailed performance analysis")

        return overall_stats
    else:
        print("❌ No statistics generated - analysis may have failed")
        return None


def process_all_strategies_comprehensive():
    """
    MODIFIED VERSION: Process recipes with ALL strategies for comprehensive comparison
    This is the original function with enhanced reporting integration
    """

    strategies = ['adaptive', 'best_of_both', 'db_first', 'fuzzy_first']

    foods_db_path = "C://Users//User//Desktop//needle_in_haystack//NeadleInHaystack//archive//FINAL FOOD DATASET//FOOD-DATA-GROUP*.csv"
    clustered_ingredients_path = 'output_table.json'

    # Initialize matcher once
    matcher = EnhancedOptimizedClusteredMatcher(
        foods_db_path=foods_db_path,
        clustered_ingredients_path=clustered_ingredients_path
    )

    # recipe_file_path = 'train_short.json'
    recipe_file_path = '../whats-cooking/train.json/train.json'

    print("\nPROCESSING ALL RECIPES WITH ALL METHODS (ENHANCED WITH REPORTING)")
    print("=" * 70)

    # Load recipes once
    try:
        with open(recipe_file_path, 'r', encoding='utf-8') as file:
            recipes_data = json.load(file)
        print(f"Loaded {len(recipes_data)} recipes from file")
    except Exception as e:
        print(f"ERROR loading recipe file: {e}")
        return None

    all_results = {}
    overall_stats = {}

    for strategy in strategies:
        print(f"\nProcessing with strategy: {strategy.upper()}")
        print("-" * 40)

        strategy_results = []
        total_ingredients = 0
        total_recipes = 0

        start_time = time.time()

        # Process recipes with current strategy
        for recipe_idx, recipe in enumerate(recipes_data):  # Limit for testing
            try:
                recipe_title = recipe.get('title', recipe.get('id', f'Recipe_{recipe_idx}'))

                if 'ingredients' not in recipe:
                    continue

                ingredients = recipe['ingredients']
                if not ingredients:
                    continue

                # Match ingredients using current strategy
                results = matcher.match_ingredients_batch_optimized(ingredients, strategy)

                # Store results
                recipe_result = {
                    'recipe_id': recipe.get('id', recipe_idx),
                    'recipe_title': recipe_title,
                    'cuisine': recipe.get('cuisine', 'unknown'),
                    'original_ingredients': ingredients,
                    'matched_ingredients': [r.matched_cluster_id if r.matched_cluster_id != -1 else r.matched_food for r
                                            in results],
                    'scores': [r.score for r in results],
                    'methods': [r.method for r in results],
                    'confidences': [r.confidence for r in results]
                }

                strategy_results.append(recipe_result)
                total_ingredients += len(ingredients)
                total_recipes += 1

                if total_recipes % 100 == 0:
                    print(f"  Processed {total_recipes} recipes...")

            except Exception as e:
                print(f"    ERROR processing recipe {recipe_idx}: {e}")
                continue

        elapsed_time = time.time() - start_time

        # Calculate statistics for this strategy
        all_results_flat = []
        for recipe_result in strategy_results:
            for i, original in enumerate(recipe_result['original_ingredients']):
                matched_ingredient = recipe_result['matched_ingredients'][i]
                cluster_id = matched_ingredient if isinstance(matched_ingredient, int) else -1
                matched_food = matched_ingredient if isinstance(matched_ingredient,
                                                                str) else f"cluster_{matched_ingredient}"

                all_results_flat.append(ClusterMatchResult(
                    original=original,
                    cleaned="",
                    matched_food=matched_food,
                    matched_cluster_id=cluster_id,
                    score=recipe_result['scores'][i],
                    method=recipe_result['methods'][i],
                    confidence=recipe_result['confidences'][i]
                ))

        stats = matcher.get_statistics(all_results_flat)

        print(f"\nStrategy: {strategy.upper()} COMPLETED")
        print(f"Clustering rate: {stats['clustering_rate']:.1%}")
        print(f"High confidence rate: {stats['high_confidence_rate']:.1%}")
        print(f"Processing time: {elapsed_time:.2f} seconds")

        # Store results
        all_results[strategy] = strategy_results
        overall_stats[strategy] = {
            'total_recipes': total_recipes,
            'total_ingredients': total_ingredients,
            'processing_time': elapsed_time,
            'stats': stats
        }

    # Save individual strategy results
    print(f"\nSAVING STRATEGY RESULTS...")
    for strategy, results in all_results.items():
        output_file = f'enhanced_clustered_ingredient_matching_{strategy}_results.json'
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Successfully saved {strategy} results to {output_file}")
        except Exception as e:
            print(f"ERROR saving {strategy} results: {e}")

    # Save summary statistics
    try:
        with open('enhanced_clustered_ingredient_matching_summary.json', 'w', encoding='utf-8') as f:
            json.dump(overall_stats, f, indent=2)
        print(f"Successfully saved summary statistics")
    except Exception as e:
        print(f"ERROR saving summary: {e}")

    print(f"\nPROCESSING COMPLETE!")
    if overall_stats:
        max_recipes = max(s['total_recipes'] for s in overall_stats.values())
        print(f"Processed {max_recipes} total recipes")
    print(f"Check output files for detailed results")

    return overall_stats


# Additional utility functions for the enhanced reporting system

def generate_latex_performance_report(overall_stats: Dict, output_dir: str = 'reports') -> None:
    """
    Generate a LaTeX-formatted performance report for academic/professional presentation
    """

    import os
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    latex_file = f"{output_dir}/performance_report_{timestamp}.tex"

    with open(latex_file, 'w', encoding='utf-8') as f:
        f.write(r"\documentclass[11pt]{article}" + "\n")
        f.write(r"\usepackage[margin=1in]{geometry}" + "\n")
        f.write(r"\usepackage{booktabs}" + "\n")
        f.write(r"\usepackage{amsmath}" + "\n")
        f.write(r"\usepackage{graphicx}" + "\n")
        f.write(r"\usepackage{float}" + "\n")
        f.write(r"\usepackage{caption}" + "\n")
        f.write(r"\title{Ingredient Matching Algorithm Performance Analysis}" + "\n")
        f.write(r"\date{\today}" + "\n")
        f.write(r"\begin{document}" + "\n")
        f.write(r"\maketitle" + "\n\n")

        f.write(r"\section{Executive Summary}" + "\n")
        f.write("This report presents a comprehensive performance analysis of four ingredient matching strategies ")
        f.write("evaluated on a dataset of recipe ingredients.\n\n")

        f.write(r"\section{Performance Metrics}" + "\n")
        f.write(r"\begin{table}[H]" + "\n")
        f.write(r"\centering" + "\n")
        f.write(r"\caption{Performance Comparison Across Strategies}" + "\n")
        f.write(r"\begin{tabular}{@{}lcccccc@{}}" + "\n")
        f.write(r"\toprule" + "\n")
        f.write("Strategy & Ingredients & Clustered & Rate (\%) & Confidence (\%) & Avg Score & Speed (ing/s) \\\\\n")
        f.write(r"\midrule" + "\n")

        for strategy, stats in overall_stats.items():
            clustered_count = stats['stats']['clustered_ingredients']
            total_ingredients = stats['stats']['total_ingredients']
            cluster_rate = (clustered_count / total_ingredients * 100) if total_ingredients > 0 else 0
            confidence_rate = stats['stats']['high_confidence_rate'] * 100
            avg_score = stats['stats']['average_score']
            speed = total_ingredients / stats['processing_time'] if stats['processing_time'] > 0 else 0

            f.write(f"{strategy.replace('_', r'\_')} & {total_ingredients:,} & {clustered_count:,} & ")
            f.write(f"{cluster_rate:.1f} & {confidence_rate:.1f} & {avg_score:.3f} & {speed:.1f} \\\\\n")

        f.write(r"\bottomrule" + "\n")
        f.write(r"\end{tabular}" + "\n")
        f.write(r"\end{table}" + "\n\n")

        f.write(r"\section{Key Findings}" + "\n")

        # Find best performing strategies
        best_clustering = max(overall_stats.keys(),
                              key=lambda s: overall_stats[s]['stats']['clustering_rate'])
        best_confidence = max(overall_stats.keys(),
                              key=lambda s: overall_stats[s]['stats']['high_confidence_rate'])

        f.write(f"The analysis reveals that the \\textbf{{{best_clustering.replace('_', r'\_')}}} strategy ")
        f.write(f"achieved the highest clustering rate of ")
        f.write(f"{overall_stats[best_clustering]['stats']['clustering_rate']:.1%}, ")
        f.write(f"while the \\textbf{{{best_confidence.replace('_', r'\_')}}} strategy ")
        f.write(f"demonstrated the highest confidence rate of ")
        f.write(f"{overall_stats[best_confidence]['stats']['high_confidence_rate']:.1%}.\n\n")

        f.write(r"\section{Mathematical Formulation}" + "\n")
        f.write("The clustering rate is defined as:\n")
        f.write(r"\begin{equation}" + "\n")
        f.write(
            r"\text{Clustering Rate} = \frac{\text{Number of Successfully Clustered Ingredients}}{\text{Total Number of Ingredients}}" + "\n")
        f.write(r"\end{equation}" + "\n\n")

        f.write("The confidence score is calculated based on fuzzy string matching and database similarity scores:\n")
        f.write(r"\begin{equation}" + "\n")
        f.write(r"\text{Confidence Level} = f(\text{similarity\_score}, \text{method\_type})" + "\n")
        f.write(r"\end{equation}" + "\n\n")

        f.write(r"\section{Conclusion}" + "\n")
        f.write("The performance analysis demonstrates significant variation across different matching strategies, ")
        f.write("with implications for optimal strategy selection based on specific use case requirements.\n\n")

        f.write(r"\end{document}" + "\n")

    print(f"LaTeX report generated: {latex_file}")


def create_performance_dashboard_html(overall_stats: Dict, output_dir: str = 'reports') -> None:
    """
    Create an interactive HTML dashboard for performance visualization
    """

    import os
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    html_file = f"{output_dir}/performance_dashboard_{timestamp}.html"

    # Prepare data for JavaScript
    strategies = list(overall_stats.keys())
    clustering_rates = [overall_stats[s]['stats']['clustering_rate'] * 100 for s in strategies]
    confidence_rates = [overall_stats[s]['stats']['high_confidence_rate'] * 100 for s in strategies]
    avg_scores = [overall_stats[s]['stats']['average_score'] for s in strategies]
    processing_speeds = [overall_stats[s]['total_ingredients'] / overall_stats[s]['processing_time']
                         for s in strategies]

    with open(html_file, 'w', encoding='utf-8') as f:
        f.write("""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Ingredient Matching Performance Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .header {
            text-align: center;
            color: #333;
            border-bottom: 2px solid #007bff;
            padding-bottom: 20px;
            margin-bottom: 30px;
        }
        .charts-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        .chart-container {
            background-color: #fafafa;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #ddd;
        }
        .chart-title {
            text-align: center;
            font-weight: bold;
            margin-bottom: 15px;
            color: #555;
        }
        .summary-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 20px;
        }
        .summary-table th, .summary-table td {
            padding: 12px;
            text-align: center;
            border-bottom: 1px solid #ddd;
        }
        .summary-table th {
            background-color: #007bff;
            color: white;
        }
        .summary-table tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        .best-performer {
            background-color: #d4edda !important;
            font-weight: bold;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Ingredient Matching Performance Dashboard</h1>
            <p>Comprehensive Analysis of Matching Strategies</p>
            <p><em>Generated: """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S') + """</em></p>
        </div>

        <div class="charts-grid">
            <div class="chart-container">
                <div class="chart-title">Clustering Rate by Strategy (%)</div>
                <canvas id="clusteringChart"></canvas>
            </div>
            <div class="chart-container">
                <div class="chart-title">High Confidence Rate (%)</div>
                <canvas id="confidenceChart"></canvas>
            </div>
            <div class="chart-container">
                <div class="chart-title">Average Matching Score</div>
                <canvas id="scoreChart"></canvas>
            </div>
            <div class="chart-container">
                <div class="chart-title">Processing Speed (ingredients/second)</div>
                <canvas id="speedChart"></canvas>
            </div>
        </div>

        <h2>📊 Performance Summary Table</h2>
        <table class="summary-table">
            <thead>
                <tr>
                    <th>Strategy</th>
                    <th>Total Ingredients</th>
                    <th>Clustered</th>
                    <th>Clustering Rate (%)</th>
                    <th>High Confidence (%)</th>
                    <th>Avg Score</th>
                    <th>Speed (ing/s)</th>
                </tr>
            </thead>
            <tbody>""")

        # Find best performers for highlighting
        best_clustering_idx = clustering_rates.index(max(clustering_rates))
        best_confidence_idx = confidence_rates.index(max(confidence_rates))
        best_score_idx = avg_scores.index(max(avg_scores))
        best_speed_idx = processing_speeds.index(max(processing_speeds))

        for i, strategy in enumerate(strategies):
            stats = overall_stats[strategy]
            row_class = ""
            if i in [best_clustering_idx, best_confidence_idx, best_score_idx, best_speed_idx]:
                row_class = ' class="best-performer"'

            f.write(f"""
                <tr{row_class}>
                    <td>{strategy.replace('_', ' ').title()}</td>
                    <td>{stats['total_ingredients']:,}</td>
                    <td>{stats['stats']['clustered_ingredients']:,}</td>
                    <td>{clustering_rates[i]:.1f}</td>
                    <td>{confidence_rates[i]:.1f}</td>
                    <td>{avg_scores[i]:.3f}</td>
                    <td>{processing_speeds[i]:.1f}</td>
                </tr>""")

        f.write("""
            </tbody>
        </table>
    </div>

    <script>
        // Chart.js configuration
        const chartConfig = {
            type: 'bar',
            options: {
                responsive: true,
                plugins: {
                    legend: {
                        display: false
                    }
                },
                scales: {
                    y: {
                        beginAtZero: true
                    }
                }
            }
        };

        // Data
        const strategies = """ + str(strategies) + """;
        const clusteringRates = """ + str(clustering_rates) + """;
        const confidenceRates = """ + str(confidence_rates) + """;
        const avgScores = """ + str(avg_scores) + """;
        const processingSpeeds = """ + str(processing_speeds) + """;

        // Color schemes
        const colors = ['#FF6384', '#36A2EB', '#FFCE56', '#4BC0C0'];

        // Create charts
        new Chart(document.getElementById('clusteringChart'), {
            ...chartConfig,
            data: {
                labels: strategies,
                datasets: [{
                    data: clusteringRates,
                    backgroundColor: colors,
                    borderColor: colors.map(c => c + '80'),
                    borderWidth: 1
                }]
            }
        });

        new Chart(document.getElementById('confidenceChart'), {
            ...chartConfig,
            data: {
                labels: strategies,
                datasets: [{
                    data: confidenceRates,
                    backgroundColor: colors,
                    borderColor: colors.map(c => c + '80'),
                    borderWidth: 1
                }]
            }
        });

        new Chart(document.getElementById('scoreChart'), {
            ...chartConfig,
            data: {
                labels: strategies,
                datasets: [{
                    data: avgScores,
                    backgroundColor: colors,
                    borderColor: colors.map(c => c + '80'),
                    borderWidth: 1
                }]
            },
            options: {
                ...chartConfig.options,
                scales: {
                    y: {
                        beginAtZero: true,
                        max: 1
                    }
                }
            }
        });

        new Chart(document.getElementById('speedChart'), {
            ...chartConfig,
            data: {
                labels: strategies,
                datasets: [{
                    data: processingSpeeds,
                    backgroundColor: colors,
                    borderColor: colors.map(c => c + '80'),
                    borderWidth: 1
                }]
            }
        });
    </script>
</body>
</html>""")

    print(f"Interactive HTML dashboard generated: {html_file}")


if __name__ == "__main__":
    # Run the enhanced version with comprehensive reporting
    print("🚀 STARTING COMPREHENSIVE ANALYSIS WITH ENHANCED REPORTING...")
    process_all_strategies_comprehensive_with_reports()
# if __name__ == "__main__":
#     # Test with a smaller subset first
#     # print("Running enhanced optimized processing on first 1000 recipes...")
#     # process_recipes_optimized_comprehensive('adaptive', max_recipes=1000)
#
#     # Uncomment to process all recipes with single strategy
#     # print("Running enhanced optimized processing on ALL recipes...")
#     # process_recipes_optimized_comprehensive('adaptive')
#
#     # Uncomment to process all recipes with ALL strategies
#     print("Running comprehensive analysis with ALL strategies...")
#     process_all_strategies_comprehensive()
#
