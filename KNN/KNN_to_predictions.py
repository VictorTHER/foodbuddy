#Package import
import pandas as pd
import numpy as np
# np.set_printoptions(legacy='1.25') # Making sure float and integers won't show as 'np.float(64)', etc.
import pandas as pd

from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

from KNN.KNN_model import weighting_nutrients
from Label_matcher.Recipes_list_setup import download_recipes_df




def load_model():
    """Load fitted KNN model"""
    if os.path.exists('KNN/fitted_model.pkl'):
        with open('KNN/fitted_model.pkl', "rb") as f:
            model = pickle.load(f)
        print("Données chargées :", model)
    else:
        print(f"Fichier introuvable : 'fitted_model.pkl'")
    return model

def load_scaler():
    """Load fitted KNN scaler"""
    if os.path.exists('KNN/fitted_scaler.pkl'):
        with open('KNN/fitted_scaler.pkl', "rb") as f:
            scaler = pickle.load(f)
        print("Données chargées :", scaler)
    else:
        print(f"Fichier introuvable : 'fitted_scaler.pkl'")
    return scaler

def load_user_data(scaler:StandardScaler, weights=None, recommended_intake=None,consumed_intake=None):
    """
    WIP : Dummy user data is being used
    => Replace with user function

    1. Load user consumed and current intake
    2. Calculate the remaining nutrient intake to match the recommended intakes
    """

    # Dummy recommended data for testing
    if recommended_intake is None:
        recommended_intake=np.array([[560.,   4., 224., 840.,  50.,   8., 504., 303., 121.,  81.]])

    # Initializing consumed_intakes
    # consumed_intake = np.zeros(len(recommended_intake))
    # Would increment each recipe intake depending on the moment of the day

    # Dummy consumed data for testing
    if consumed_intake is None:
        consumed_intake=np.array([[280. ,   2. , 112. , 420. ,  25. ,   4. , 252. , 151.,  60., 40.]])

    # Calculate the remaining intake
    X_remaining=recommended_intake-consumed_intake

    # Scaling the nutrients
    # X_remaining_scaled=scaler.transform(X_remaining)

    # Weighting the nutrients
    """WIP To do later : Weight should be customized at some point over each user's importance of reaching certains nutrients"""
    # weights= # Put the user adapted weight algorithm below and before calling weighting_nutrients + Put weights as argument
    X_remaining_weighted=weighting_nutrients(X_remaining,weights)

    print('Successfully loaded and weighted the user nutritional data')
    return X_remaining_weighted

def predict_KNN_model():
    """MAIN FUNCTION"""
    """Prediction and Nutrition calculation"""
    models=load_model()
    scaler=load_scaler()
    X_remaining_weighted = load_user_data(scaler)

    # Making sure that X is a 2D array:
    if X_remaining_weighted.shape!=(1,10):
        X_remaining_weighted = X_remaining_weighted.reshape(1, -1)

    # Predicting the most nutritious recipes
    y_pred= models.kneighbors(X_remaining_weighted)

    print('Successfully predicted the ideal recipes')

    # # Displaying to the terminal the raw KNN output
    # print("Here are the raw prediction outputs (recipe matching 'distances', and recipe index in the recipes dataset) :",y_pred)

    # # Displaying to the terminal the selected recipes
    # # (UTD 12/02/2024 - before GitHub push to main) Making a Python object for Streamlit & API pipeline with the recipe names
    # """First step : Loading the recipe names from the model's trained targets, as they contain both recipes indexes and titles"""
    # y=pd.read_csv('./','recipe_titles.csv')

    # print("Here are the selected recipes by order of matching :")
    # recommended_recipes_names=[]
    # predicted_recipes=y_pred[1][0]
    # for i,recipe_index in enumerate(predicted_recipes) :
    #     print(f"""The recommended recipe n°{i+1} is : {data.iloc[recipe_index]['recipe']}.""") # Printing by matching order the selected recipe
    #     recommended_recipes_names.append(data.iloc[recipe_index]['recipe']) # Generating the list of recipe names by matching order for later use
    # return y_pred, recommended_recipes_names
    return y_pred

def predict_and_sort(model=load_model(), scaler=load_scaler()):

    column_mapping = {
    'Carbohydrates_(G)_total': 'carbohydrates_g',
    'Protein_(G)_total': 'protein_g',
    'Lipid_(G)_total': 'lipid_g',
    'Calcium_(MG)_total': 'calcium_mg',
    'Iron_(MG)_total': 'iron_mg',
    'Magnesium_(MG)_total': 'magnesium_mg',
    'Sodium_(MG)_total': 'sodium_mg',
    'Vitamin_A_(UG)_total': 'vitamin_a_ug',
    'Vitamin_C_(MG)_total': 'vitamin_c_mg',
    'Vitamin_D_(UG)_total': 'vitamin_d_ug'
    }

    X_remaining_weighted = load_user_data(scaler, recommended_intake=np.array([np.array([292, 117, 78, 697, 6, 279, 1046, 627, 63, 10])]), consumed_intake=np.array([18, 113, 35, 107, 4, 132, 1616, 574, 9.3, 0.3]))

    if X_remaining_weighted.shape!=(1,10):
        X_remaining_weighted = X_remaining_weighted.reshape(1, -1)

    y_pred= model.kneighbors(X_remaining_weighted)

    # Process des dataframes

    standard_data_columns= [
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

    remaining = pd.DataFrame(X_remaining_weighted, columns=standard_data_columns)

    recipes_df = download_recipes_df()
    recipes_df = recipes_df[['recipe', 'Calcium_(MG)_total', 'Carbohydrates_(G)_total', 'Iron_(MG)_total', 'Lipid_(G)_total', 'Magnesium_(MG)_total', 'Protein_(G)_total', 'Sodium_(MG)_total', 'Vitamin_A_(UG)_total', 'Vitamin_C_(MG)_total', 'Vitamin_D_(UG)_total']]
    recipes_df = recipes_df.rename(columns=column_mapping)
    final_columns = ['recipe', 'carbohydrates_g', 'protein_g', 'lipid_g', 'calcium_mg', 'iron_mg', 'magnesium_mg', 'sodium_mg', 'vitamin_a_ug', 'vitamin_c_mg', 'vitamin_d_ug']
    recipes_df = recipes_df[final_columns]

    recipe_indices = y_pred[1][0]

    matching_recipes = recipes_df.loc[recipe_indices]

    ordered_nutrients = [
    'protein_g',
    'iron_mg',
    'vitamin_d_ug',
    'calcium_mg',
    'lipid_g',
    'magnesium_mg',
    'vitamin_a_ug',
    'vitamin_c_mg',
    'sodium_mg',
    'carbohydrates_g'
    ]

    filtered_nutrients = [nutrient for nutrient in ordered_nutrients if remaining[nutrient].iloc[0] >= 0]
    print(filtered_nutrients)

    diff_df = pd.DataFrame()

    for nutrient in filtered_nutrients:
        matching_recipes[f'diff_{nutrient}'] = matching_recipes[nutrient] - remaining[nutrient].iloc[0]

        diff_df = matching_recipes.iloc[:, 11:]

    # Filtrer les recettes où la différence est positive
        diff_df = diff_df[diff_df[f'diff_{nutrient}'] >= 0]

    # Si le nombre de recettes restantes est inférieur à 5, ajouter une logique de récupération
        if len(diff_df) < 5:
        # Identifier les indices des recettes rejetées
            rejected_indices = matching_recipes.index.difference(diff_df.index)

        # Réintégrer les recettes rejetées pour garantir un minimum de 5 recettes
            missing_count = 5 - len(diff_df)
            additional_recipes = matching_recipes.loc[rejected_indices].iloc[:missing_count]

        # Ajouter les recettes manquantes à diff_df
            diff_df = pd.concat([diff_df, additional_recipes])

    # Mettre à jour `matching_recipes` pour conserver uniquement les recettes restantes
        matching_recipes = matching_recipes.loc[diff_df.index]

    # Arrêter la boucle si le nombre de recettes restantes est inférieur ou égal à 5
        if len(diff_df) <= 5:
            break

# Finaliser la liste des recettes restantes
    recipes = matching_recipes['recipe'].tolist()

    print(diff_df)
    print(matching_recipes)
    print(remaining)
    return recipes

y_pred=predict_KNN_model()

print(y_pred)
