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
            print(f"Progress saved to {cache_1}")

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
    - Take manualy matched file and rematch names with nutrients list
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
        print("Labels cleaned")

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
        print("Nice tables generated")

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
        merged_df = merged_df.drop_duplicates(subset=["target_cleaned"], keep="first")

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
        recipe = download_recipes_df()
        ingredient = download_ingredients_df()
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

        # Combine "recipe_cleaned" and "ingredient_cleaned" into a single "recipe" column
        equivalents_table['recipe'] = equivalents_table['recipe_cleaned']
        equivalents_table.loc[equivalents_table['ingredient_cleaned'].notna(), 'recipe'] = equivalents_table['ingredient_cleaned']
        equivalents_table = equivalents_table.drop(columns=['target_cleaned','recipe_cleaned', 'ingredient_cleaned','is_ok'])

    ### STEP 7: MERGE EQUIVALENTS TABLE AND NUTRIENTS ####

        # Concatenate recipe and ingredient DataFrames
        combined = pd.concat([recipe, ingredient], ignore_index=True)
        combined = combined.drop_duplicates(subset=["recipe"], keep="first")
        print("recipe and ingredient tables joined")

        # Merge equivalents table with nutrients
        merged = equivalents_table.merge(
            combined,
            on='recipe',
            how='left'
        )
        merged = merged.drop_duplicates(subset=["recipe"], keep="first")
        merged = merged.drop(columns=["recipe"])
        merged = merged.rename(columns={"original_target":"recipe"})
        print("merge done")

    ### STEP 8: SAVE AND UPLOAD TO GCS ####

        # Reorganize columns
        ordered_columns = ['recipe', 'default_portion', 'default_portion_in_grams', 'Calcium_(MG)_total', 'Carbohydrates_(G)_total', 'Iron_(MG)_total', 'Lipid_(G)_total', 'Magnesium_(MG)_total', 'Protein_(G)_total', 'Sodium_(MG)_total', 'Vitamin_A_(UG)_total', 'Vitamin_C_(MG)_total', 'Vitamin_D_(UG)_total', 'Calcium_(MG)_per_100G', 'Carbohydrates_(G)_per_100G', 'Iron_(MG)_per_100G', 'Lipid_(G)_per_100G', 'Magnesium_(MG)_per_100G', 'Protein_(G)_per_100G', 'Sodium_(MG)_per_100G', 'Vitamin_A_(UG)_per_100G', 'Vitamin_C_(MG)_per_100G', 'Vitamin_D_(UG)_per_100G']
        merged = merged[ordered_columns]

        # Save locally
        merged.to_csv(cache_3, index=False)
        print(f"Saved final target to nutrients file to {cache_3}.")

        # Upload final DataFrame to GCS
        client = storage.Client()
        bucket_name = "recipes-dataset"
        bucket = client.bucket(bucket_name)
        destination_blob_name = "Targets/target_to_nutrients.csv"
        file_name = "target_to_nutrients.csv"
        merged.to_csv(file_name, index=False)
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(file_name)

        print(f"Final linked data saved and uploaded to {destination_blob_name}")
        return merged

    if progress == 3 :
        print("'Targets/target_to_nutrients.csv' file is in GCS. Please remove it and remove 'cache_3' to remake the file.")

        return cache

    return None


def download_target_match():
    """
    Download targets with nutrients data frame from GCS.
    1 second runtime !
    """
    # Initialize Google Cloud Storage client
    client = storage.Client()
    bucket_name = "recipes-dataset"
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"Targets/target_to_nutrients.csv")
    content = blob.download_as_text()

    # Return df
    return pd.read_csv(StringIO(content))
