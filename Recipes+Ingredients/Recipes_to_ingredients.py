from google.cloud import storage
from io import StringIO
import pandas as pd
import numpy as np


def download_recipes_df():
    """
    Download cleaned recipes with nutrients data frame from GCS.
    1 second runtime !
    """
    # Initialize Google Cloud Storage client
    client = storage.Client()
    bucket_name = "recipes-dataset"
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(f"Recipes/recipes.csv")
    content = blob.download_as_text()

    # Return df
    return pd.read_csv(StringIO(content))

# recipes = download_recipes_df()
