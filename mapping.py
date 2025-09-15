import pandas as pd
import sys
import re

from tqdm import tqdm

import duckdb
import shelve

from textblob import TextBlob

from importlib import resources
from symspellpy.symspellpy import SymSpell

sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
dictionary_path = resources.files("symspellpy").joinpath("frequency_dictionary_en_82_765.txt")
sym_spell.load_dictionary(str(dictionary_path), term_index=0, count_index=1)

dictionary_path = resources.files("symspellpy").joinpath("frequency_bigramdictionary_en_243_342.txt")
sym_spell.load_bigram_dictionary(str(dictionary_path), term_index=0, count_index=2)

def correct1(sentence: str) -> str:
    return sym_spell.word_segmentation(sentence).corrected_string

def correct(sentence: str) -> str:
    return str(TextBlob(sentence).correct())

import spellchecker

spell = spellchecker.SpellChecker('en')
spell.distance = 1

def correct2(sentence: str) -> str:
    words = sentence.split()
    corrected_words = []
    for word in words:
        correction = spell.correction(word)
        if correction == None:
            return sentence
        corrected_words.append(correction)

    return ' '.join(corrected_words)

def fix_ingredient_list(ingredient_list):
    """Clean each ingredient in a list"""
    if isinstance(ingredient_list, list):
        return [clean_ingredient_text(ingredient) for ingredient in ingredient_list]
    else:
        # If it's a single string, still clean it
        return clean_ingredient_text(ingredient_list)

def clean_ingredient_text(ingredient):
    """Clean ingredient by keeping only letters and spaces, and removing specific words"""
    if not isinstance(ingredient, str):
        return ""

    # More permissive: allow unicode letters
    cleaned = re.sub(r'[^\w\s]', '', ingredient, flags=re.UNICODE)
    cleaned = re.sub(r'[0-9]', '', cleaned)  # Remove numbers separately

    # Remove extra spaces and convert to lowercase
    cleaned = ' '.join(cleaned.split()).lower()

    # Remove specific words
    words_to_remove = ['aged', 'frozen', 'fresh', 'organic', 'chopped', 'ground',
                       'cubed', 'sliced', 'cooked', 'roasted', 'dried', 'fried',
                       'fully', 'flavored', 'unflavored', 'powdered', 'plain']

    for word in words_to_remove:
        cleaned = cleaned.replace(word, '')

    # Clean up any extra spaces
    cleaned = ' '.join(cleaned.split())
    return cleaned

INGREDIENTS_FILENAME = 'foodb/Food.json'

class Search(object):
    def __init__ (self, database_name: str = ':memory:'):
        super().__init__()
        self._database_name = database_name
        self.con = None

    def __enter__ (self):
        self.con = duckdb.connect(self._database_name)
        self.con.execute(rf'''
            INSTALL fts;
            LOAD fts;
            SET preserve_insertion_order=false;
            SET threads=1;
            CREATE TABLE ingredients AS
                SELECT *
                    FROM read_json_auto('{INGREDIENTS_FILENAME}');
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
        search = self.con.execute(rf'''
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

    def find_most_similar_word(self, string1: str) -> str:
        with shelve.open("cached_search") as db:
            result = db.get(string1, None)
            if result is not None:
                return result

        sc_string1 = correct(string1)

        if sc_string1 == 'milk':
            sc_string1 = "milk (cow)"
            string1 = sc_string1

        ret = ""
        sc_string1_chosen = False

        for i, str1 in enumerate([string1, sc_string1]):
            searched_strings = self.resolve_word(str1)
            if len(searched_strings) != 0:
                ret = searched_strings[0].lower()
                sc_string1_chosen = i == 1
                break

        if sc_string1_chosen and sc_string1 != string1:
            print("original: ", string1, "corrected: ", sc_string1, "ret: ", ret, file = sys.stderr)

        with shelve.open("cached_search") as db:
            db[string1] = ret

        if ret == "":
            print(string1, "not found")

        return ret

    def find_most_similar(self, list1) -> dict[str, str]:
        """Optimized version with word matching bonus - returns top 1 match only"""
        result = {}

        # Flatten and clean
        flattened_list1 = []
        for item in list1:
            if isinstance(item, list):
                flattened_list1.extend(item)
            else:
                flattened_list1.append(item)

        valid_strings1 = [s.strip().lower() for s in flattened_list1 if isinstance(s, str) and s.strip()]

        for string1 in valid_strings1:
            x = self.find_most_similar_word(string1)
            if x:
                result[string1] = x


        return result

def main():
    with Search() as search:
        recipe_chunks = pd.read_json(".//whats-cooking//train_lines.json", chunksize = 1024, lines = True)

        for recipes in tqdm(recipe_chunks, desc="Chunks"):
            recipes['ingredients'] = recipes['ingredients'].apply(fix_ingredient_list)

            best_match = recipes['ingredients'].apply(lambda x: list(search.find_most_similar(x).values()))
            print(best_match)

            recipes['ingredients'] = best_match

            recipes.to_json('train_lines.json', orient = 'records', lines = True, mode = 'a')

def main2():
    with Search() as search:
        recipes = pd.read_json(".//whats-cooking//train.json", lines = True)
        recipes['ingredients'] = recipes['ingredients'].apply(fix_ingredient_list)

        tmp1 = recipes.head(10)
        print(tmp1.iloc[0])

        result  = tmp1['ingredients'].apply(lambda x: search.find_most_similar(x))

        for res in result:
            print(res)

#
def resolve_words(input: list) -> list:
    with Search() as search:
        lst = pd.Series(input)
        lst = lst.apply(fix_ingredient_list)
        res = list(search.find_most_similar(lst).values())

        return res

def main4():
    with Search() as search:
        res = search.find_most_similar(['mayonaise'])
    print(res)

def test_search():
    with Search() as search:
        print(search.resolve_word('mayonaise'))

if __name__ == '__main__':
    main()

##baking powder
