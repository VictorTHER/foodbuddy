from google.cloud import storage
from io import StringIO
import pandas as pd
import numpy as np


def generate_ingredients_list():
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

        # Ensure column consistency
    pivot_data.rename(columns={
        "description": "recipe",
        "portion_description": "default_portion",
        "gram_weight": "default_portion_in_grams"
    }, inplace=True)

    # Handle default portion in grams
    pivot_data["default_portion_in_grams"] = pivot_data["default_portion_in_grams"].fillna(100).replace(0, 100)

    # Calculate total nutrients for the default portion
    nutrient_columns = [
        "Carbohydrates_(G)_per_100G", "Protein_(G)_per_100G", "Lipid_(G)_per_100G",
        "Calcium_(MG)_per_100G", "Iron_(MG)_per_100G", "Magnesium_(MG)_per_100G",
        "Sodium_(MG)_per_100G", "Vitamin_C_(MG)_per_100G", "Vitamin_D_(UG)_per_100G",
        "Vitamin_A_(UG)_per_100G"
    ]
    for nutrient in nutrient_columns:
        total_col = nutrient.replace("_per_100G", "_total")
        pivot_data[total_col] = (pivot_data[nutrient] * pivot_data["default_portion_in_grams"]) / 100

    # Add custom entry for "Flour"
    flour_data = {
        "recipe": "Flour",
        "default_portion": "100 grams",
        "default_portion_in_grams": 100,
        "Carbohydrates_(G)_per_100G": 76,
        "Protein_(G)_per_100G": 10,
        "Lipid_(G)_per_100G": 1,
        "Calcium_(MG)_per_100G": 15,
        "Iron_(MG)_per_100G": 1.2,
        "Magnesium_(MG)_per_100G": 22,
        "Sodium_(MG)_per_100G": 2,
        "Vitamin_C_(MG)_per_100G": 0,
        "Vitamin_D_(UG)_per_100G": 0,
        "Vitamin_A_(UG)_per_100G": 0
    }
    for nutrient in nutrient_columns:
        total_col = nutrient.replace("_per_100G", "_total")
        flour_data[total_col] = (flour_data[nutrient] * 100) / 100  # Default portion is 100g

    # Append to pivot_data
    pivot_data = pd.concat([pivot_data, pd.DataFrame([flour_data])], ignore_index=True)

    # Final cleanup
    pivot_data = pivot_data.drop(columns=["fdc_id"], errors="ignore")

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

