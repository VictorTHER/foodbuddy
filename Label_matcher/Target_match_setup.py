import re
import os
import ast
import string
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from rapidfuzz import process, fuzz
from google.cloud import storage
from io import StringIO
import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

from Ingredients_list_setup import download_ingredients_df
from Recipes_list_setup import download_recipes_df

# DEFINE CACHE LOCATION
cache_1 = "Label_matcher/cache/cache_1.csv"
cache_2 = "Label_matcher/cache/cache_2.csv"
cache_3 = "Label_matcher/cache/cache_3.csv"
cleaned_ingredients_list = "Label_matcher/cache/cleaned_ingredients_list.csv"
cleaned_recipes_list = "Label_matcher/cache/cleaned_recipes_list.csv"


def clean_text(series):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    def preprocess(food_item):
        # Remove 'raw' and 'NFC' terms
        food_item = re.sub(r"\braw\b", "", food_item, flags=re.IGNORECASE)
        food_item = re.sub(r"\bnfs\b", "", food_item, flags=re.IGNORECASE)

        # Other cleaning
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
            row['is_ok'] = "fuzzy_matched"  # Set "is_ok" to "fuzzy_matched" for manual analysis
        # Append the updated row
        updated_rows.append(row)

    # Convert updated rows back into a DataFrame
    updated_rows_df = pd.DataFrame(updated_rows)

    # Merge updated rows back into the original DataFrame
    merged_df.update(updated_rows_df)

    return merged_df


def manual_review_fuzzy_matches(merged_df, save_interval=20):
    """
    Function to manually review rows where `is_ok` is 'fuzzy_matched'.
    Saves progress to a checkpoint file every `save_interval` rows.
    Displays ingredient match (if present) as priority or recipe match if ingredient is absent.
    """
    # Filter rows needing manual review
    fuzzy_rows = merged_df[merged_df["is_ok"] == "fuzzy_matched"]
    total = len(fuzzy_rows)
    print(f"Total rows to review: {total}\n")

    # Loop through each row for manual review
    for idx, (index, row) in enumerate(fuzzy_rows.iterrows(), start=1):
        # Determine which match to show
        match = row['ingredient_cleaned'] if pd.notna(row['ingredient_cleaned']) else row['recipe_cleaned']

        # Display review details in a clean format
        print(f"Reviewing {idx}/{total}")
        print("-" * 22)
        print(f"{row['target_cleaned']} --> {match}")
        print("-" * 22)

        # User decision
        decision = input("c to confirm, n to reject: ").strip().lower()

        if decision == 'c':
            # Approve match
            merged_df.at[index, "is_ok"] = True
        elif decision == 'n':
            # Reject match and clear columns
            merged_df.at[index, "is_ok"] = False
            if pd.notna(row['ingredient_cleaned']):
                merged_df.at[index, "ingredient_cleaned"] = None
            if pd.notna(row['recipe_cleaned']):
                merged_df.at[index, "recipe_cleaned"] = None
        else:
            print("Invalid input. Please enter 'c' or 'n'.")
            continue  # Repeat the current review if input is invalid

        # Save progress every `save_interval` rows
        if idx % save_interval == 0 or idx == total:
            merged_df.to_csv(cache_1, index=False)
            print(f"Progress saved to {cache_1")

        # Remaining rows to check
        remaining = total - idx
        print(f"\nRemaining rows to review: {remaining}\n")

    return merged_df


def get_target_match(target_list="data_train_set.csv"):
    """
    Input target list name from GCS, having "name_readable" column!
    Function will smartly keep track of progress using cache documents
    Function will go through the following steps:
    - Get target/recipes/ingredients from GCS
    - Clean labels to prepare for matching
    - Auto match using exact match + fuzzy matching
    - Output a file to check inacurate matches and rematch
    -
    Output a match list uploaded to GCS with ALL nutrients.
    """

    ### STEP 1: CHECK CURRENT PROGRESS USING CACHE ###

    progress = 0

    if os.path.exists(cache_1):
        progress = 1
        updated_to_clean = pd.read_csv(cache_1)

    if os.path.exists(cache_2):
        if input("Did you manualy match the missing recipes? 'y'/'n' ") == "y":
            progress = 2
            equivalents_table = pd.read_csv(cache_2)
        else :
            print("Please finish matching the recipes before launching this program.")
            return None

    if os.path.exists(cache_3):
        progress = 3
        cache = pd.read_csv(cache_3)

    if progress == 0 :

    ### STEP 2: DOWNLOAD DATA FROM GCS ###

        # Get targets list
        client = storage.Client()
        bucket_name = "recipes-dataset"
        bucket = client.bucket(bucket_name)
        blob = bucket.blob(f"Targets/data_train_set.csv")
        content = blob.download_as_text()
        target = pd.read_csv(StringIO(content))["name_readable"]
        print("Targets downloaded")

        # Get recipes list
        recipe = download_recipes_df()["recipe"]
        print("Recipes downloaded")

        # Get ingredients list
        ingredient = download_ingredients_df()["recipe"]
        print("Ingredients downloaded")

    ### STEP 3: CLEAN LABELS FOR MATCHING ###

        # Use clean_text function (basic NLP)
        target_cleaned = clean_text(target)
        ingredient_cleaned = clean_text(ingredient)
        recipe_cleaned = clean_text(recipe)

        # Save cleaned labels for final manual matching
        pd.Series(ingredient_cleaned).to_csv(cleaned_ingredients_list, index=False, header=False)
        pd.Series(recipe_cleaned).to_csv(cleaned_recipes_list, index=False, header=False)

        # Convert pd.Series to df for further processing
        target_df = pd.DataFrame({
            "original_target": target,
            "target_cleaned": target_cleaned
        }).reset_index(drop=True)
        recipe_df = pd.DataFrame({"recipe_cleaned": recipe_cleaned}).reset_index(drop=True)
        ingredient_df = pd.DataFrame({"ingredient_cleaned": ingredient_cleaned}).reset_index(drop=True)

     ### STEP 4: AUTO MERGING PROCESS ###

       # Match on exact name
        merged_df = target_df.merge(
            recipe_df,
            left_on="target_cleaned",
            right_on="recipe_cleaned",
            how="left"
        ).merge(
            ingredient_df,
            left_on="target_cleaned",
            right_on="ingredient_cleaned",
            how="left"
        )
        merged_df = merged_df.drop_duplicates()

        # Use new columns to keep track of matched things
        merged_df["is_ok"] = merged_df[["recipe_cleaned", "ingredient_cleaned"]].notna().any(axis=1)

        # Apply fuzzy matching to fill gaps
        updated_to_clean = fuzzy_match_and_update(merged_df, recipe_cleaned.tolist(), ingredient_cleaned.tolist())

        # Save progress as "cache_1"
        updated_to_clean.to_csv(cache_1, index=False)
        print(f"DataFrame to auto-review saved to {cache_1}.")
        progress = 1

    if progress == 1 :

    ### STEP 5: MANUAL MERGING PROCESS ###

        print("Auto-matches completed, now entering easy verification:")
        final_df = manual_review_fuzzy_matches(updated_to_clean,save_interval=20)

        final_df.to_csv(cache_2, index=False)
        print(f"""You've successfuly sorted auto-matches.\n\n
              Please use Excel/Pages to open the following documents \n
              - {cache_2}\n
              - {cleaned_ingredients_list}\n
              - {cleaned_recipes_list}\n
              And then fix the remaining matches by hand.""")

        progress = 2

    if progress == 2 :

    ### STEP 6: PREPARE MERGE ####

        # Download recipes/ingredients again
        recipe = download_recipes_df()["recipe"]
        ingredient = download_ingredients_df()["recipe"]
        print("recipes/ingredients nutrients database downloaded")

        # Clean recipes/ingredients labels
        ingredient["recipe"] = clean_text(ingredient["recipe"])
        recipe["recipe"] = clean_text(recipe["recipe"])
        print("Recipe/ingredients labels cleaned")

        # Quickly clean up manualy fixed table
        equivalents_table['recipe_cleaned'] = equivalents_table['recipe_cleaned'].astype(str).str.strip().str.lower()
        equivalents_table['ingredient_cleaned'] = equivalents_table['ingredient_cleaned'].astype(str).str.strip().str.lower()

        # Handle "nans" if not done correctly with Excel/Pages
        equivalents_table['recipe_cleaned'] = equivalents_table['recipe_cleaned'].replace("nan", np.nan)
        equivalents_table['ingredient_cleaned'] = equivalents_table['ingredient_cleaned'].replace("nan", np.nan)

        # Prioritize ingredients df over recipes = delete all recipe matches when ingredient has matched
        equivalents_table.loc[equivalents_table['ingredient_cleaned'].notna(), 'recipe_cleaned'] = np.nan

    ### STEP 7: MERGE EQUIVALENTS TABLE AND NUTRIENTS ####


        ### NAME ISSUE CHECK COLUMN NAMES!!!
        #
        #
        #
        #
        #

        # Merge data using cleaned equivalents table
        merged = equivalents_table.merge(
            ingredient,
            left_on='ingredient_cleaned',
            right_on='ingredient',
            how='left'
        ).merge(
            recipe,
            left_on='recipe_cleaned',
            right_on='recipe',
            how='left'
        )
        print("merge done")

        # Step 5: Drop dupplicates
        merged = merged.drop_duplicates(subset=["target_cleaned"], keep="first")
        print("File successfuly generated")

        # Step 6: Save locally
        merged.to_csv(cache_3, index=False)
        print(f"XXXXXX to {cache_3} for next steps.")


        #
        #
        #"original_target" column contains original name
        #
        #



    if progress == 3 :

    ### STEP X: XXXS ####


        target["name_readable_cleaned"] = clean_text(target["name_readable"])

        # Merge target names with cache
        merged = cache.merge(
            target[["name_readable", "name_readable_cleaned"]],
            left_on="target_cleaned",
            right_on="name_readable_cleaned",
            how="left"
        )
        print("Target names linked with equivalence board.")

        # Step 3: Split merged DataFrame into two parts
        merged_recipes = merged[~merged["recipe_cleaned"].isna()]  # Only rows with recipe_cleaned
        merged_ingredients = merged[~merged["ingredient_cleaned"].isna()]  # Only rows with ingredient_cleaned

        # Step 4: Perform merges for recipes and ingredients separately
        recipe_linked = merged_recipes.merge(
            recipe,
            left_on="recipe_cleaned",
            right_on="recipe",
            how="left"
        )
        print("Equivalence board linked with recipes.")

        ingredient_linked = merged_ingredients.merge(
            ingredient,
            left_on="ingredient_cleaned",
            right_on="recipe",
            how="left"
        )
        print("Equivalence board linked with ingredients.")

        # Step 5: Concatenate the two DataFrames
        final_merged = pd.concat([recipe_linked, ingredient_linked], ignore_index=True)
        print("Recipes and ingredients combined.")

        # Step 6: Remove duplicates based on 'name_readable' column
        final_merged = final_merged.drop_duplicates(subset=["name_readable"]).drop(columns=["recipe_cleaned", "ingredient_cleaned"])

        # Step 7: Keep only required columns (20+ columns)
        nutrient_columns_per_100g = [
            "Calcium_(MG)_per_100G", "Carbohydrates_(G)_per_100G", "Iron_(MG)_per_100G",
            "Lipid_(G)_per_100G", "Magnesium_(MG)_per_100G", "Protein_(G)_per_100G",
            "Sodium_(MG)_per_100G", "Vitamin_A_(UG)_per_100G", "Vitamin_C_(MG)_per_100G",
            "Vitamin_D_(UG)_per_100G"
        ]
        nutrient_columns_total = [
            "Calcium_(MG)_total", "Carbohydrates_(G)_total", "Iron_(MG)_total",
            "Lipid_(G)_total", "Magnesium_(MG)_total", "Protein_(G)_total",
            "Sodium_(MG)_total", "Vitamin_A_(UG)_total", "Vitamin_C_(MG)_total",
            "Vitamin_D_(UG)_total"
        ]

        required_columns = ["name_readable", "default_portion", "default_portion_in_grams"] + nutrient_columns_per_100g + nutrient_columns_total
        final_df = final_merged[required_columns].drop_duplicates()
        ### STEP X: UPLOAD MATCH TABLE TO GCS ###

        # Upload final DataFrame to GCS
        destination_blob_name = "Targets/target_to_nutrients.csv"
        file_name = "target_to_nutrients.csv"
        final_df.to_csv(file_name, index=False)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(file_name)

        print(f"Final linked data saved and uploaded to {destination_blob_name}")
        return final_df
















# OUTDATED!!! After matches are done, bring all data back together // ISSUE BC ingredients/recipe format has been changed since!!
def final_match():


    # Step 3: Load cached matches
    try:
        cache = pd.read_csv(cache_file)
        print("cache extracted")

        # Harmonize formats
        cache['recipe_cleaned'] = cache['recipe_cleaned'].astype(str).str.strip().str.lower()
        recipe['recipe'] = recipe['recipe'].astype(str).str.strip().str.lower()
        cache['ingredient_cleaned'] = cache['ingredient_cleaned'].astype(str).str.strip().str.lower()
        ingredient['ingredient'] = ingredient['ingredient'].astype(str).str.strip().str.lower()

        cache['recipe_cleaned'] = cache['recipe_cleaned'].replace("nan", np.nan)
        cache['ingredient_cleaned'] = cache['ingredient_cleaned'].replace("nan", np.nan)

        cache.loc[cache['ingredient_cleaned'].notna(), 'recipe_cleaned'] = np.nan
        print("NaNs handled")

        # Step 4: Merge data using merge key (priority FOR ingredients)
        merged = cache.merge(
            ingredient,
            left_on='ingredient_cleaned',
            right_on='ingredient',
            how='left'
        ).merge(
            recipe,
            left_on='recipe_cleaned',
            right_on='recipe',
            how='left'
        )
        print("merge done")

        # Step 5: Drop dupplicates
        merged = merged.drop_duplicates(subset=["target_cleaned"], keep="first")
        print("File successfuly generated")

        # Step 6: Save locally
        merged.to_csv(cache_file_final, index=False)
        print(f"Final df saved to {cache_file_final} for next steps.")

        return merged

    except FileNotFoundError:
        print(f"Cache file {cache_file_final} not found.")
        return None


# Filter ingredients list + recipes list BASED ON above generated match table
# Upload final MATCH table to cloud. in Targets/
# Don't forget to filter out of ALL the options :)
def final_merge():
    """
    Links target names with equivalence board and recipes/ingredients lists,
    including both per 100G and total columns.

    Parameters:
    - cache_file (str): Path to the cache file containing equivalence board.

    Returns:
    - pd.DataFrame: Final merged DataFrame linking targets, recipes, and ingredients.
    """

    # Step 1: Download and prepare data
    client = storage.Client()
    bucket_name = "recipes-dataset"

    # Download cleaned recipes
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"Recipes/cleaned_recipes_with_nutrients.csv")
    content = blob.download_as_text()
    recipe = pd.read_csv(StringIO(content))
    print("Recipes downloaded")

    # Download ingredients
    ingredients = download_ingredients_df()
    ingredient = ingredients.copy()
    print("Ingredients downloaded")

    # Download targets
    blob = bucket.blob(f"Targets/data_train_set.csv")
    content = blob.download_as_text()
    targets = pd.read_csv(StringIO(content))
    target = targets.copy()
    print("Targets downloaded")

    # Load cache equivalence board
    cache = pd.read_csv(cache_file)
    print("Equivalence board gotten from cache")

    ingredient["recipe"] = clean_text(ingredient["recipe"])
    target["name_readable_cleaned"] = clean_text(target["name_readable"])
