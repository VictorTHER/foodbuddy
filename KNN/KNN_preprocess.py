#Package import
import pandas as pd
import numpy as np
# np.set_printoptions(legacy='1.25') # Making sure float and integers won't show as 'np.float(64)', etc. 
import pandas as pd
from google.cloud import storage
from io import StringIO


def load_recipes_KNN_data():
    """
    WIP : Load the dataset from GCP instead of a local git-ignored source. 

    1. Import data recipe
    2. Standardize the structure
    3. Return standardized dataset for later recipe prediction (KNN)
    """
    # Initializing Google Cloud Storage client
    client = storage.Client()
    bucket_name = "recipes-dataset"
    bucket = client.bucket(bucket_name)

    # Loading the file into a pandas DataFrame
    # Accessing the blob
    blob = bucket.blob(f"Recipes/cleaned_recipes_with_nutrients.csv")
    # Downloading the content as text
    content = blob.download_as_text()
    # Using StringIO to create a file-like object
    data=pd.read_csv(StringIO(content))    

    # Load recipe/nutrients data
    # (local for now)
    # data=pd.read_csv("../raw_data/Recipes_cleaned_recipes_with_nutrients.csv")

    # Deleting null rows    
    data.dropna(inplace=True)

    # Resetting indexes to become sequential again
    data = data.reset_index(drop=True)

    # Cleaning the new database for preprocessing
    # Identifying nutrient columns
    nutrient_columns = [col for col in data.columns if col.endswith('_total')]
    nutrient_columns.insert(0,'recipe')

    # print(data.columns)

    for i,col in enumerate(nutrient_columns):
        col=col.lower()
        nutrient_columns[i]=col.replace("_total","").replace("(","").replace(")","")

    # Standardizing columns names for the next processing steps 
    rename_mapping = {
    'Calcium_(MG)_total': 'calcium_mg',
    'Iron_(MG)_total': 'iron_mg',
    'Magnesium_(MG)_total': 'magnesium_mg',
    'Sodium_(MG)_total': 'sodium_mg',
    'Vitamin_C_(MG)_total': 'vitamin_c_mg',
    'Vitamin_D_(UG)_total': 'vitamin_d_ug',
    'Vitamin_A_(UG)_total': 'vitamin_a_ug',
    'Carbohydrates_(G)_total': 'carbohydrates_g',
    'Protein_(G)_total': 'protein_g',
    'Lipid_(G)_total': 'lipid_g',
    'recipe': 'recipe'}


    data.rename(columns=rename_mapping, inplace=True)

    # Shuffling column order for the KNN model
    standard_data_columns= ['recipe',
    'carbohydrates_g', # macronutrients come first for readability
    'protein_g',
    'lipid_g',
    'calcium_mg', # micronutrients
    'iron_mg',
    'magnesium_mg',
    'sodium_mg',
    'vitamin_a_ug',
    'vitamin_c_mg',
    'vitamin_d_ug']

    #Setting the final dataframe
    data=data[standard_data_columns]

    print ("Recipe and nutrients dataset successfully imported and standardized.")
    return data


