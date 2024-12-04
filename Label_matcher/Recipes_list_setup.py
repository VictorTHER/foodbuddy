import re
import os
import ast
import string
from collections import Counter
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from rapidfuzz import process, fuzz
from google.cloud import storage
from io import StringIO
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

from foodbuddy.Label_matcher.Ingredients_list_setup import download_ingredients_df


def parse_ingredient(ingredient):
    """
    Input: Ingredient cell of a recipe row from the recipe dataset
    Output: A tuple (quantity, grammage, unit, name) where:
        - 'quantity': How many portions
        - 'grammage': Grammage depending on the unit provided
        - 'unit': Unit of the grammage
        - 'name': Name of the ingredient
    Purpose: Output will be used to generate a proper DataFrame using pd.DataFrame.
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

        return (float(portion_quantity), grammage, unit, name.strip())

    # If no pattern is recognized -> Default return
    return (1, None, None, ingredient.strip())


def clean_ingredients(series):
    """
    Input a series containing recipe/ingredient/target names
    Remove all the unecessary stuff (non-food words, spaces, lemmatize)
    Output the series with cleaned names
    """
    non_food_items = [
    # Verbs
    'add', 'adjust', 'approx', 'blend', 'buy', 'clean', 'combine', 'contain', 'cook', 'crush',
    'cut', 'dice', 'discard', 'drain', 'drizzle', 'grate', 'halve', 'measure', 'mix', 'prepare',
    'refrigerate', 'serve', 'slice', 'spread', 'stir', 'use', 'wrap', 'yield', 'bake', 'chop',
    'freeze', 'grind', 'mince', 'pack', 'shred', 'soften', 'taste', 'adjustable', 'assemble',
    'blanch', 'break', 'expose', 'filter', 'press', 'reduce', 'remove', 'reserve', 'rinse', 'rise',
    'shake', 'squeeze', 'store', 'trim', 'try', 'wash', 'whip', 'whisk',

    # Adjectives / Descriptions
    'additional', 'big', 'bitter', 'black', 'brown', 'cold', 'coarse', 'crispy', 'dry', 'extra',
    'fine', 'firm', 'fresh', 'freshly', 'hard', 'heavy', 'hot', 'large', 'light', 'little', 'long',
    'medium', 'necessary', 'new', 'optional', 'possible', 'raw', 'rich', 'rough', 'simple', 'smooth',
    'soft', 'solid', 'sweet', 'tepid', 'thick', 'thin', 'tender', 'warm', 'whole', 'without',

    # Units / Measurements
    'approx', 'cup', 'degree', 'dozen', 'fluid', 'g', 'gm', 'gram', 'inch', 'intact', 'kg',
    'kilogram', 'lb', 'liter', 'litre', 'milliliter', 'ml', 'ounce', 'oz', 'ozs', 'part',
    'percent', 'piece', 'pinch', 'portion', 'pound', 'quart', 'quarter', 'tablespoon', 'tbsp',
    'tbsps', 'teaspoon', 'tsp', 'unit', 'weight',

    # Non-food items / Tools
    'bag', 'bottle', 'bowl', 'brush', 'can', 'carton', 'case', 'container', 'core', 'cover',
    'equipment', 'foil', 'glass', 'jar', 'knife', 'lid', 'mold', 'paper', 'peeler', 'plate',
    'pot', 'processor', 'punnet', 'scoop', 'sheet', 'skewer', 'spoon', 'stick', 'thermometer',
    'tin', 'tray', 'wrapper',

    # Miscellaneous
    'across', 'active', 'allpurpose', 'also', 'although', 'amount', 'around', 'assembly',
    'averna', 'available', 'best', 'bit', 'bitesize', 'block', 'break', 'capacity', 'choice',
    'color', 'concentrate', 'condense', 'count', 'couple', 'decorate', 'degree', 'diagonal',
    'diamond', 'dish', 'double', 'etc', 'favorite', 'find', 'form', 'free', 'great', 'guide',
    'handful', 'head', 'inspire', 'instant', 'keep', 'leave', 'lengthways', 'lengthwise', 'like',
    'loosely', 'may', 'mixture', 'natural', 'note', 'overly', 'plain', 'plenty', 'prefer',
    'preferably', 'purpose', 'quality', 'range', 'really', 'recommend', 'right', 'roughly',
    'shoot', 'side', 'solution', 'square', 'starter', 'strip', 'strong', 'substitute',
    'temperature', 'thats', 'third', 'tip', 'total', 'ultrapasteurized', 'unbleached',
    'uncooked', 'unflavored', 'unpeeled', 'unripe', 'variety', 'vie', 'well', 'whatever',
    'wild', 'wing', '“', '”', '’', '°', '¼', '¼oz', '½', 'ème', '–'
    ]
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

        # Remove non food items
        food_item_filtered = [word for word in food_item_lemmatized if word not in non_food_items]

        # Sort and join
        food_item_sorted = sorted(food_item_filtered)
        return ' '.join(food_item_sorted)

    return series.dropna().map(preprocess)


def clean_ingredient_names(recipes, ingredients):
    """
    Input recipes and ingredients dfs
    Get the 1000 most common ingredient words in recipes
    Keep only recipe ingredients that contain the 1000 words
    Remove all other recipe ingredients
    Output recipes with only cleaned ingredients
    """
    print("cleaning ingredients list")
    ingredients["recipe"] = clean_ingredients(ingredients["recipe"])
    print("cleaning recipes ingredients list")
    recipes["ingredient"] = clean_ingredients(recipes["ingredient"])

    # Count word occurrences in recipe content
    word_counter = Counter()
    for content in recipes["ingredient"].dropna():
        word_counter.update(content.split())
    print("couted 1000 most common ingredients")

    # Keep only the top 1000 most common ingredient words
    top_1000_words = {word for word, _ in word_counter.most_common(1000)}

    # Filter content by valid words
    def filter_content(content):
        words = content.split()
        filtered_words = [word for word in words if word in top_1000_words]
        return " ".join(filtered_words) if filtered_words else None

    recipes["ingredient"] = recipes["ingredient"].apply(lambda x: filter_content(x) if pd.notna(x) else None)
    print("kept only food items")
    return recipes.dropna(subset=["ingredient"])


def fuzzy_match_and_update(filtered_recipe, ingredient):
    """
    Input recipes with names filtered and ingredients df
    Do fuzzy matching to match recipe ingredients with the USDA ingredients
    Output recipes with only cleaned ingredients
    """
    # Prepare the list of USDA ingredient names for fuzzy matching
    ingredient_list = ingredient['recipe'].dropna().tolist()

    # Define a helper function for fuzzy matching
    def match_ingredient(content):
        if pd.notna(content):
            # Perform fuzzy matching with RapidFuzz
            ingredient_match = process.extractOne(
                content,
                ingredient_list,
                scorer=fuzz.ratio,
                score_cutoff=60
            )
            if ingredient_match:
                return ingredient_match[0], "fuzzy_matched"
        return content, "not_matched"

    # Apply the helper function to the "ingredient" column using pandas apply
    filtered_recipe = filtered_recipe.copy()
    filtered_recipe[['ingredient_cleaned', 'is_ok']] = filtered_recipe['ingredient'].apply(
        lambda x: pd.Series(match_ingredient(x))
    )

    return filtered_recipe


def preprocess_merged_df(matched_recipes, ingredients):
    """
    Merge matched recipes with ingredients and calculate the default portion in grams.
    If unit has been specified in the recipe as "grams", keep the recipe grammage
    Otherwise, take USDA standard portion
    Output df with calculations done on portion weight.
    """
    # Merge matched recipes with ingredient details
    ingredients = ingredients.rename(columns={"recipe": "content"})
    merged_df = matched_recipes.merge(
        ingredients, left_on="ingredient_cleaned", right_on="content", how="left"
    )

    # Calculate portion using vectorized operations
    merged_df["calculated_portion"] = merged_df["grammage"].where(
        merged_df["unit"] == "g"
    )  # Keep grammage if unit is "g"

    # Multiply default portion by quantity if grammage is not provided
    merged_df["calculated_portion"] = merged_df["calculated_portion"].fillna(
        merged_df["default_portion_in_grams"]
    )

    # Drop rows where portion could not be calculated
    merged_df = merged_df.dropna(subset=["calculated_portion"]).reset_index(drop=True)

    # Remove unnecessary columns
    columns_to_remove = ["content", "quantity", "grammage", "unit", "ingredient_cleaned", "is_ok"]
    merged_df = merged_df.drop(columns=columns_to_remove, errors="ignore")

    return merged_df


def calculate_total_per_person(df):
    """
    Input list of recipes with all recipe ingredients and nutrients
    Perform calculations based on portion size
    Output done calculations
    """
    # Dynamically identify nutrient columns ending with '_per_100G'
    nutrient_columns = [col for col in df.columns if col.endswith('_per_100G')]

    df['portion_per_person'] = df['calculated_portion'] / 4

    # Calculate total nutrients for one person using vectorized operations
    for col in nutrient_columns:
        total_col_name = col.replace('_per_100G', '_total')
        # Vectorized calculation of total nutrients
        df[total_col_name] = (df[col] * df['portion_per_person']) / 100

    return df


def consolidate_recipe_nutrients(df):
    """
    Input recipes & their ingredients df with all calculations done
    Sum the all recipe ingredients nutrients together
    Output recipes with nutrients without ingredients
    """
    # List of nutrient columns
    nutrient_columns_per_100G = [
        'Calcium_(MG)_per_100G', 'Carbohydrates_(G)_per_100G', 'Iron_(MG)_per_100G',
        'Lipid_(G)_per_100G', 'Magnesium_(MG)_per_100G', 'Protein_(G)_per_100G',
        'Sodium_(MG)_per_100G', 'Vitamin_A_(UG)_per_100G', 'Vitamin_C_(MG)_per_100G',
        'Vitamin_D_(UG)_per_100G'
    ]

    nutrient_columns_total = [col.replace('_per_100G', '_total') for col in nutrient_columns_per_100G]

    # Ensure default_portion_in_grams is non-zero and replace NaN values with a default value
    df['portion_per_person'] = df['portion_per_person'].fillna(0)

    # Calculate the weighted average for nutrients per 100g
    weighted_averages = (
        df.groupby("recipe").apply(
            lambda group: pd.Series({
                col: (group[col] * group['portion_per_person']).sum() /
                     max(group['portion_per_person'].sum(), 1)  # Prevent division by zero
                for col in nutrient_columns_per_100G
            })
        )
    ).reset_index()

    # Calculate the total nutrients for 1-person portion
    total_nutrients = (
        df.groupby("recipe")[nutrient_columns_total].sum().reset_index()
    )

    # Calculate total weight per recipe
    total_weights = (
        df.groupby("recipe")['portion_per_person'].sum().reset_index()
    )
    total_weights.rename(columns={'default_portion_in_grams': 'total_portion_weight'}, inplace=True)

    # Merge all results together
    consolidated_df = weighted_averages.merge(total_nutrients, on="recipe")
    consolidated_df = consolidated_df.merge(total_weights, on="recipe")

    # Add a default portion description column
    consolidated_df["default_portion"] = "recipe_1_person"

    # Rename total weight column to match the output format
    consolidated_df.rename(columns={"portion_per_person": "default_portion_in_grams"}, inplace=True)

    return consolidated_df


def generate_recipe_list():
    """
    Pull original recipes.csv list with 200k recipes from GCS
    Get ingredients with nutrients list from GCS (made with Ingredients_list_setup.py)
    Use parser to get all ingredients for each recipe with portion size
    match recipe ingredient names with USDA ingredients list with nutrients
    Save to GCS for future use
    This takes a LOT of time! 30mins for 200k recipes...
    """
    print("Warning! This function may take a lot of time to run :) No worries, progress will be saved throughout the process")

    # Step 1: Download data
    client = storage.Client()
    bucket_name = "recipes-dataset"
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"Recipes/recipes.csv")
    content = blob.download_as_text()
    recipes = pd.read_csv(StringIO(content))
    recipe = recipes.copy().drop_duplicates(subset=["title"], keep='first')
    print("Recipes downloaded")

    ingredients = download_ingredients_df()
    print("Ingredients downloaded")

    # Step 2: Process ingredients
    all_parsed_ingredients = [
        (title, *parse_ingredient(ingredient))
        for title, ingreds in zip(recipe["title"], recipe["ingredients"])
        for ingredient in ast.literal_eval(ingreds)
        ]
    print("All ingredients of all recipes pulled")

    # Convert to DataFrame directly
    flat_recipes_df = pd.DataFrame(
        all_parsed_ingredients,
        columns=["recipe", "quantity", "grammage", "unit", "ingredient"]
        )
    print("Parsed ingredients combined into a DataFrame")

    local_csv = "temp_1.csv"
    flat_recipes_df.to_csv(local_csv, index=False)
    print(f"Data saved locally as {local_csv}")

    # Step 2: Clean and filter recipes
    filtered_recipes = clean_ingredient_names(flat_recipes_df, ingredients)
    print("cleaned and filtered recipes")

    local_csv = "temp_2.csv"
    filtered_recipes.to_csv(local_csv, index=False)
    print(f"Data saved locally as {local_csv}")

    # Step 3: Fuzzy matching and merging
    updated_recipes = fuzzy_match_and_update(filtered_recipes, ingredients)
    print("fuzzy matched recipes")

    local_csv = "temp_3.csv"
    updated_recipes.to_csv(local_csv, index=False)
    print(f"Data saved locally as {local_csv}")

    matched_recipes = updated_recipes[updated_recipes["is_ok"]=="fuzzy_matched"]
    merged_df = preprocess_merged_df(matched_recipes, ingredients)
    print("Matched ingredients together")

    local_csv = "temp_4.csv"
    merged_df.to_csv(local_csv, index=False)
    print(f"Data saved locally as {local_csv}")

    # Step 4: Calculate totals and consolidate
    merged_df = calculate_total_per_person(merged_df)

    local_csv = "temp_5.csv"
    merged_df.to_csv(local_csv, index=False)
    print(f"Data saved locally as {local_csv}")

    consolidated_df = consolidate_recipe_nutrients(merged_df)
    print("Calculated nutrients")

    # Step 5: Save results to CSV
    local_csv = "cleaned_recipes_with_nutrients.csv"
    consolidated_df.to_csv(local_csv, index=False)
    print(f"Data saved locally as {local_csv}")

    # Upload to Google Cloud Storage
    client = storage.Client()
    bucket_name = "recipes-dataset"
    bucket = client.bucket(bucket_name)
    blob_name = "Recipes/cleaned_recipes_with_nutrients.csv"
    blob = bucket.blob(blob_name)
    blob.upload_from_filename("cleaned_recipes_with_nutrients.csv")
    print(f"Recipes with nutrients successfully uploaded to GCS as {blob_name}.")

    # delete cache!
    for i in range(1, 6):
        temp_file = f"temp_{i}.csv"
        try:
            os.remove(temp_file)
            print(f"Temporary file {temp_file} deleted.")
        except OSError as e:
            print(f"Error: {temp_file} could not be deleted. {e}")

    return consolidated_df


def download_recipes_df():
    """
    Download cleaned recipes with nutrients data frame from GCS.
    1 second runtime !
    """
    # Initialize Google Cloud Storage client
    client = storage.Client()
    bucket_name = "recipes-dataset"
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"Recipes/cleaned_recipes_with_nutrients.csv")
    content = blob.download_as_text()

    # Return df
    return pd.read_csv(StringIO(content))
