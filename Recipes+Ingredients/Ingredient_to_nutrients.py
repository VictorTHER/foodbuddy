from google.cloud import storage
from io import StringIO
import pandas as pd

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
ingredients = load_csv_from_gcs("cleaned_ingredients_with_nutrients.csv")
