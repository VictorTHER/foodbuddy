import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

import re
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from rapidfuzz import process, fuzz
from google.cloud import storage
from io import StringIO
import pandas as pd

def download_target_df():
    """
    Download targets df from GCS.
    1 second runtime !
    """
    # Initialize Google Cloud Storage client
    client = storage.Client()
    bucket_name = "recipes-dataset"
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"Targets/data_train_set.csv")
    content = blob.download_as_text()

    # Return df
    return pd.read_csv(StringIO(content))


def merge_target_recipes_ingredients():
    """
    Function will pull recipes, ingredients and target df from GCS
    and return a equivalents board.
    1 min runtime
    """
    from Recipes_to_ingredients import download_recipes_df
    recipes = download_recipes_df()
    recipe = recipes.copy()["recipe"]

    from Ingredients_to_nutrients import download_ingredients_df
    ingredients = download_ingredients_df()
    ingredient = ingredients.copy()["ingredient"]

    targets = download_target_df()
    target = targets.copy()["name_readable"]

    def clean_text(series):
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words('english'))

        def preprocess(food_item):
            food_item_strip_lowered = food_item.strip().lower()
            food_item_cleaned = ''.join(char for char in food_item_strip_lowered if char not in string.punctuation and not char.isdigit())
            food_item_tokenized = word_tokenize(food_item_cleaned)
            food_item_no_stop_words = [word for word in food_item_tokenized if word not in stop_words]
            food_item_lemmatize_verbs = [lemmatizer.lemmatize(word, pos='v') for word in food_item_no_stop_words]
            food_item_lemmatized = [lemmatizer.lemmatize(word, pos='n') for word in food_item_lemmatize_verbs]
            food_item_sorted = sorted(food_item_lemmatized)
            return ' '.join(food_item_sorted)

        return series.dropna().drop_duplicates().map(preprocess)

    def fuzzy_match_and_update(merged_df, recipe_cleaned, ingredient_cleaned):
        # Filter rows where "is_ok" is False
        rows_to_process = merged_df[merged_df["is_ok"] == False]

        # List to store updated rows
        updated_rows = []

        for index, row in rows_to_process.iterrows():
            target_name = row['target_cleaned']

            # Perform fuzzy matching on recipes
            recipe_match = process.extractOne(target_name, recipe_cleaned, scorer=fuzz.ratio, score_cutoff=60)

            # Perform fuzzy matching on ingredients
            ingredient_match = process.extractOne(target_name, ingredient_cleaned, scorer=fuzz.ratio, score_cutoff=60)

            # Update row if matches are found
            if recipe_match or ingredient_match:
                if recipe_match:
                    row['recipe_cleaned'] = recipe_match[0]  # Update recipe column
                if ingredient_match:
                    row['ingredient_cleaned'] = ingredient_match[0]  # Update ingredient column
                row['is_ok'] = True  # Set "is_ok" to True
            # Append the updated row
            updated_rows.append(row)

        # Convert updated rows back into a DataFrame
        updated_rows_df = pd.DataFrame(updated_rows)

        # Merge updated rows back into the original DataFrame
        merged_df.update(updated_rows_df)

        return merged_df

    target_cleaned = clean_text(target)
    ingredient_cleaned = clean_text(ingredient)
    recipe_cleaned = clean_text(recipe)

    # Convert cleaned series to DataFrames for merging
    target_df = pd.DataFrame({"target_cleaned": target_cleaned}).reset_index(drop=True)
    recipe_df = pd.DataFrame({"recipe_cleaned": recipe_cleaned}).reset_index(drop=True)
    ingredient_df = pd.DataFrame({"ingredient_cleaned": ingredient_cleaned}).reset_index(drop=True)

    # Merge all three based on name (exact matches)
    merged_df = target_df.merge(
        recipe_df, left_on="target_cleaned", right_on="recipe_cleaned", how="left"
    ).merge(
        ingredient_df, left_on="target_cleaned", right_on="ingredient_cleaned", how="left"
    )
    merged_df = merged_df.drop_duplicates()
    merged_df["is_ok"] = merged_df[["recipe_cleaned", "ingredient_cleaned"]].notna().any(axis=1)

    updated_to_clean = fuzzy_match_and_update(merged_df, recipe_cleaned.tolist(), ingredient_cleaned.tolist())

    return updated_to_clean


# result = merge_target_recipes_ingredients()
# print(result)
