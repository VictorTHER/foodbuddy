from google.cloud import storage
from io import StringIO
import pandas as pd
import numpy as np
import re
import ast
from Recipes import ingredients_per_recipe_dictionary

def clean_recipes():
    """
    1. Process the raw recipe table.
    2. Generate a dictionary storing the nutrients-per-ingredient datasets for each recipe, by calling the parse_ingredients function.
    """
    # Data loading
    #Load data
    # Initialize Google Cloud Storage client
    client = storage.Client()
    bucket_name = "recipes-dataset"
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"Recipes/recipes.csv")
    content = blob.download_as_text()

    data = pd.read_csv(StringIO(content))

    # Data cleaning (update : optional)
    # data.drop(columns=['directions','rating'], inplace=True)

    # Calling the Recipes module to generate a dictionary of parsed ingredients per recipe
    dict_recipes = ingredients_per_recipe_dictionary(data)

    # Flatten the dictionary into a single DataFrame
    all_recipes = []
    for recipe_name, df in dict_recipes.items():
        df["recipe_name"] = recipe_name  # Add recipe name to each ingredient row
        all_recipes.append(df)
    flat_recipes_df = pd.concat(all_recipes, ignore_index=True)

    # Reorganize df
    flat_recipes_df = flat_recipes_df[["recipe_name","name","quantity","grammage","unit"]]
    flat_recipes_df = flat_recipes_df.rename(columns={"recipe_name":"recipe","name":"ingredient"})

    # Upload to GCS
    destination_blob_name = "Recipes/cleaned_recipes_with_ingredients.csv"
    file_name = "cleaned_recipes_with_ingredients.csv"
    flat_recipes_df.to_csv(file_name, index=False)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_name)

    print(f"File {file_name} successfully uploaded to GCS as {destination_blob_name}.")
    return None


def download_recipes_df():
    """
    Download cleaned recipes with nutrients data frame from GCS.
    1 second runtime !
    """
    # Initialize Google Cloud Storage client
    client = storage.Client()
    bucket_name = "recipes-dataset"
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"Recipes/cleaned_recipes_with_ingredients.csv")
    content = blob.download_as_text()

    # Return df
    return pd.read_csv(StringIO(content))


# clean_recipes()
# parse_ingredient(ingredient)
# download_recipes_df()
