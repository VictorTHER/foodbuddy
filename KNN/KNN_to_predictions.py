#Package import
import pandas as pd
import numpy as np
import pandas as pd

from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

from KNN_model import weighting_nutrients



def load_model():
    """Load fitted KNN model"""
    if os.path.exists('fitted_model.pkl'):
        with open('fitted_model.pkl', "rb") as f:
            model = pickle.load(f)
        print("Données chargées :", model)
    else:
        print(f"Fichier introuvable : 'fitted_model.pkl'")
    return model

def load_scaler():
    """Load fitted KNN scaler"""
    if os.path.exists('fitted_scaler.pkl'):
        with open('fitted_scaler.pkl', "rb") as f:
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
        recommended_intake=np.array([[303., 121.,   81., 560., 4.,  224.,   840., 504., 50., 8.]])

    # Initializing consumed_intakes # Temporarily disabled: Can be used if multiple recipes or more than one lunch is allowed 
    # consumed_intake = np.zeros(len(recommended_intake))
    # Would increment each recipe intake depending on the moment of the day

    # Dummy consumed data for testing
    if consumed_intake is None:
        consumed_intake=np.array([[151.,  60.,  40., 280.,   2., 112., 420., 252., 25.,   4.]])

    # Calculate the remaining intake
    X_remaining=recommended_intake-consumed_intake

    # Scaling the nutrients
    X_remaining_scaled=scaler.transform(X_remaining)
    
    # Weighting the nutrients
    # """WIP To do later : Weight should be customized at some point over each user's importance of reaching certains nutrients"""
    # weights= # Put the user adapted weight algorithm below and before calling weighting_nutrients + Put weights as argument
    # X_remaining_scaled=weighting_nutrients(X_remaining_scaled,weights) # Disabled : Weighting no longer used, preferring validating selected recipes' nutrition after using the model. 
    
    print('Successfully loaded and weighted the user nutritional data')
    return X_remaining_scaled

def predict_KNN_model():
    """MAIN FUNCTION"""
    """Prediction and Nutrition calculation
    - Loads and scales the users' remaining nutrients of the day.
    - Inputs the KNN model to predicts the closest recipes to fulfill the recommended intakes
    - Returns a selection of those recipes: database indexes, recipe names, and a clean terminal output ranking them 
    """
    # Loading the model, scaler and user's scaled remaining intakes
    model=load_model()
    scaler=load_scaler()
    X_remaining_scaled = load_user_data(scaler)

    # Making sure that X is a 2D array:
    if X_remaining_scaled.shape!=(1,10):
        X_remaining_scaled = X_remaining_scaled.reshape(1, -1)

    # Predicting the most nutritious recipes
    y_pred= model.kneighbors(X_remaining_scaled)
    
    print('Successfully predicted the ideal recipes')

    # Displaying to the terminal the raw KNN output
    print("Here are the raw prediction outputs (recipe matching 'distances', and recipe index in the recipes dataset) :",y_pred)

    # Displaying to the terminal the selected recipes
    # (UTD 12/02/2024 - before GitHub push to main) Making a Python object for Streamlit & API pipeline with the recipe names
    """First step : Loading the recipe names from the model's trained targets, as they contain both recipes indexes and titles"""
    y=pd.read_csv('./recipe_titles.csv')

    print("Here are the selected recipes by order of matching :")
    recommended_recipes_names=[]
    predicted_recipes=y_pred[1][0]
    for i,recipe_index in enumerate(predicted_recipes) :
        print(f"""The recommended recipe n°{i+1} is : {y.loc[recipe_index]['recipe']}.""") # Printing by matching order the selected recipe
        recommended_recipes_names.append(y.loc[recipe_index]['recipe']) # Generating the list of recipe names by matching order for later use 
    return y_pred, recommended_recipes_names

if __name__=='__main__':
    predict_KNN_model()


"""Suggestions for the output : 

1. After-recommended-lunch nutrition validation
2. Data display to the user (To Be Aligned with API project work)

"""

""" Under the hood : Nutrition fulfilling validation 

1. (Ceiling Threshold) Are some recipes overreaching the recommended intakes ?
2. (Floor Threshold) Are there still some nutrient deficiency after taking a recommended recipe ?
3. Cleaning the recipes, then give recommended
4. Recommend a top 3 or 5 list of recipes to the user

"""



