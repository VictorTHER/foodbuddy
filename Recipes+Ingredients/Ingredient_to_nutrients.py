from google.cloud import storage
from io import StringIO
import pandas as pd
import numpy as np

def clean_ingredients_list():
    """
    Take USA GOV nutrients dataset from GCS and clean them up
    Then reupload 4k ingredients list with OUR 10 selected nutrients!
    """
    # Initialize Google Cloud Storage client
    client = storage.Client()
    bucket_name = "recipes-dataset"
    bucket = client.bucket(bucket_name)

    # Load each file into pandas DataFrames
    def load_csv_from_gcs(file_name):
        # Access the blob
        blob = bucket.blob(f"Ingredients/{file_name}")
        # Download the content as text
        content = blob.download_as_text()
        # Use StringIO to create a file-like object
        return pd.read_csv(StringIO(content))

    # Download dfs from GCS
    food_org = load_csv_from_gcs("food.csv")[["fdc_id","description"]]
    food_nutrient_org = load_csv_from_gcs("food_nutrient.csv")[["fdc_id","nutrient_id","amount"]]
    food_portion_org = load_csv_from_gcs("food_portion.csv")[["fdc_id","portion_description","gram_weight"]]
    nutrient_org = load_csv_from_gcs("nutrient.csv")[["nutrient_nbr","name","unit_name"]]

    # Prepare data frames
    food = food_org.copy()

    food_portion = food_portion_org.copy()
    food_portion = food_portion.drop_duplicates(subset="fdc_id", keep="first")

    nutrient = nutrient_org.copy()
    nutrients = pd.DataFrame({
        "name": [
            "Carbohydrates",
            "Protein",
            "Lipid",
            "Calcium",
            "Iron",
            "Magnesium",
            "Sodium",
            "Vitamin_C",
            "Vitamin_D",
            "Vitamin_A"
        ],
        "nutrient_id": [205, 203, 204, 301, 303, 304, 307, 401, 328, 320],
        "unit_name": ["G", "G", "G", "MG", "MG", "MG", "MG", "MG", "UG", "UG"]
    })

    food_nutrient = food_nutrient_org.copy()
    food_nutrient = food_nutrient[food_nutrient["nutrient_id"].isin(nutrients["nutrient_id"])]

    # Merge all 4 dfs
    merged_data = pd.merge(food_nutrient, nutrients, how="left", on="nutrient_id")
    merged_data = pd.merge(merged_data, food, how="left", on="fdc_id")
    merged_data = pd.merge(merged_data, food_portion, how="left", on="fdc_id")

    # Prepare for 10 nutrients pivot
    merged_data["name_with_unit"] = merged_data["name"] + "_(" + merged_data["unit_name"] + ")_per_100G"
    merged_data = merged_data.drop(columns=["nutrient_id", "name", "unit_name"])
    merged_data = merged_data[[
        "description",
        "portion_description",
        "gram_weight",
        "fdc_id",
        "name_with_unit",
        "amount"
        ]]

    # Pivot table
    pivot_data = merged_data.pivot_table(
        index=["description", "portion_description", "gram_weight", "fdc_id"],
        columns="name_with_unit",
        values="amount",
        aggfunc="first"
    ).reset_index()
    pivot_data.columns.name = None
    pivot_data.columns = [col if isinstance(col, str) else col for col in pivot_data.columns]

    # Final cleanup!
    pivot_data = pivot_data.rename(columns={
        "description": "ingredient",
        "portion_description": "default_portion",
        "gram_weight": "default_portion_in_grams"
    })
    pivot_data = pivot_data.drop(columns=["fdc_id"])

    # Upload to GCS
    destination_blob_name = "Ingredients/cleaned_ingredients_with_nutrients.csv"
    file_name = "cleaned_ingredients_with_nutrients.csv"
    pivot_data.to_csv(file_name, index=False)
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(file_name)

    print(f"File {file_name} successfully uploaded to GCS as {destination_blob_name}.")
    return None


def download_ingredients_df():
    """
    Download cleaned ingredients with nutrients data frame from GCS.
    1 second runtime !
    """
    # Initialize Google Cloud Storage client
    client = storage.Client()
    bucket_name = "recipes-dataset"
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"Ingredients/cleaned_ingredients_with_nutrients.csv")
    content = blob.download_as_text()

    # Return df
    return pd.read_csv(StringIO(content))


def nutrition_facts(ingredient, weight):
    """
    Calculate nutrition facts for a given ingredient and weight.

    Parameters:
    - ingredient (str): The name of the ingredient.
    - weight (float or NaN): The weight of the ingredient in grams. If NaN, use default portion weight.
    - ingredients_df (DataFrame): The DataFrame containing ingredient data.

    Returns:
    - dict: Nutrition facts including ingredient, weight, and nutrient amounts.
    """
    ingredients_df = download_ingredients_df()
    
    ### STEP 1: MATCH INGREDIENT NAME ###
    try:
        # option 1 = exact match (shouldn't be dupplicates :) )
        ingredient_data = ingredients_df.loc[ingredients_df['ingredient'] == ingredient]
        if ingredient_data.empty:
            raise ValueError(f"Ingredient '{ingredient}' not found in the dataset.")
        # option 2 = DL matching model (for later)

        ### STEP 2: CHECK WEIGHT/CONVERT UNITS ###
        weight = weight if pd.notna(weight) else float(ingredient_data['default_portion_in_grams'].iloc[0])


        ### STEP 3: CALCULATE NUTRIENT AMOUNTS ###
        result = {'ingredient': ingredient, 'weight': weight}
        nutrient_columns = [
            col for col in ingredients_df.columns
            if col not in ['ingredient', 'default_portion', 'default_portion_in_grams']
        ]
        for nutrient in nutrient_columns:
            result[nutrient] = round(float((ingredient_data[nutrient].iloc[0] / 100) * weight), 2)

        key_mapping = {
            'Calcium_(MG)_per_100G': 'Calcium (MG)',
            'Carbohydrates_(G)_per_100G': 'Carbohydrates (G)',
            'Iron_(MG)_per_100G': 'Iron (MG)',
            'Lipid_(G)_per_100G': 'Lipid (G)',
            'Magnesium_(MG)_per_100G': 'Magnesium (MG)',
            'Protein_(G)_per_100G': 'Protein (G)',
            'Sodium_(MG)_per_100G': 'Sodium (MG)',
            'Vitamin_A_(UG)_per_100G': 'Vitamin A (UG)',
            'Vitamin_C_(MG)_per_100G': 'Vitamin C (MG)',
            'Vitamin_D_(UG)_per_100G': 'Vitamin D (UG)',
        }

        renamed_result = {
            key_mapping.get(key, key): value for key, value in result.items()
        }
        return renamed_result

    except Exception as e:
        return {"error": str(e)}


# clean_ingredients_list()
# ingredients = download_ingredients_df()
# print(nutrition_facts("Abalone",10,ingredients))
