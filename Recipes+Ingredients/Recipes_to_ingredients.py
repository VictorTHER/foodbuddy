from google.cloud import storage
from io import StringIO
import pandas as pd
import numpy as np
import re
import ast


def parse_ingredient(ingredient):
    """
    Input : Ingredient cell of a recipe row from the recipe dataset
    Output : A dataframe-like dictionary where each each pair will be the future column/cell value of each ingredient from the same recipe, where :
        'quantity': How many portion
        'grammage': Grammage depending on the unit provided
        'unit': Unit of the grammage
        'name': Name of the ingredient
    Purpose : Output will be used to generate a proper dataframe using pd.DataFrame
    """
    VALID_UNITS = {'g', 'tbsp', 'tsp', 'tspn', 'cup', 'ml', 'l', 'kg', 'oz', 'fl oz'}
    # Preprocessing to remove "/xoz" patterns and fractions like "½oz" when g is already provided
    ingredient = re.sub(r'/\d+oz', '', ingredient)  # Remove patterns like "/9oz"
    ingredient = re.sub(r'/\d+fl oz', '', ingredient)  # Remove patterns like "/9fl oz"
    ingredient = re.sub(r'/\d+[½⅓¼¾]+oz', '', ingredient)  # Remove fractions before "oz"

    # Regex to capture quantity, unit, and name
    pattern = r'(?:(\d+)(?:\s*x\s*(\d+))?)?\s*([a-zA-Z%½⅓¼]+)?\s*(.*)'
    match = re.match(pattern, ingredient)

    if match:
        quantity, sub_quantity, unit, name = match.groups()

        # Default values
        grammage = None
        portion_quantity = 1  # Default quantity if not provided

        # Handle the case of "2 x 80g"
        if sub_quantity:
            portion_quantity = int(quantity)
            grammage = int(sub_quantity)
        elif quantity and unit:
            grammage = int(quantity)
        elif quantity:
            portion_quantity = int(quantity)

        # If no grammage or unit is provided
        if not unit and not grammage:
            name = ingredient.strip()  # Full ingredient name as name

        # Debugging exception : Handling cases where the detected unit is actually the first word of the ingredient name
        if unit and unit not in VALID_UNITS:
            # Move the incorrectly detected unit back into the beggining of the name
            name = f"{unit} {name}".strip()
            unit = None  # Clear the unit, since it's invalid

        # Exception when a fraction of quantity is provided
        # Output example before fixing : 1       NaN     unit  ½ leftover roast chicken, torn into pieces
        # Fix : Check if a fraction is at the beginning of the name and adjust quantity
        fraction_pattern = r'^([½⅓¼¾])\s*(.*)'
        fraction_match = re.match(fraction_pattern, name)
        if fraction_match and portion_quantity == 1:
            fraction, remaining_name = fraction_match.groups()
            try:
                # Fraction to decimal dictionary
                fraction_value = {
                    "½": 0.5,
                    "⅓": 0.33,
                    "¼": 0.25,
                    "¾": 0.75
                }[fraction]
                portion_quantity = fraction_value  # Replacing quantity with the decimal
                name = remaining_name.strip()  # Removing the fraction from the name
            except KeyError:
                pass  # Keep running the code if error

        return {
            'quantity': float(portion_quantity),
            'grammage': grammage,
            'unit': unit,
            'name': name.strip()
        }
    # if no pattern is recognized eventually -> Default return
    return {
        'quantity': 1,
        'grammage': None,
        'unit': None,
        'name': ingredient.strip()
    }


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

    # Generate a dictionary of parsed ingredients
    dict_recipes = {}
    for i, row in data.iterrows():
        ingredients = ast.literal_eval(row['ingredients'])
        parsed_ingredients = [parse_ingredient(ingredient) for ingredient in ingredients]
        dict_recipes[row['title']] = pd.DataFrame(parsed_ingredients)

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
