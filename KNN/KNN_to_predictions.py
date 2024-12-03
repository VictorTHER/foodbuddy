#Package import
import pandas as pd
import numpy as np
# np.set_printoptions(legacy='1.25') # Making sure float and integers won't show as 'np.float(64)', etc. 
import pandas as pd

from sklearn.neighbors import KNeighborsRegressor, NearestNeighbors
from sklearn.model_selection import train_test_split,cross_validate
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from KNN_preprocess import load_recipes_KNN_data


"""DEV NOTES :
----
12/02/2024 :
1. Packaging is done 
2. Now running with Google Cloud Storage platform
3. Pushed to GitHub for sharing with teammates
WIP 1: MLFlow push
WIP 2: Fine-tuning recommendations with nutrient deficiency/overreach thresholds (probably in predictions_to_output.py) 


11/28/2024 : 
1. Summarizing the KNN notebook to streamline the code
2. Will package it later into functions
 """

def preprocessing():
    """
    1. Load the recipe's data using the KNN_preprocess.py
    2. Create and scaling the nutrient features for the KNN model
    """
    # Calling the dataset function
    data=load_recipes_KNN_data()

    # Loading the features
    X=data.drop(columns=['recipe']) 
    y=data.recipe

    ## Scaling the nutrients features
    scaler=StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print('Successfully preprocessed the recipe/nutrient data')
    return X_scaled,y,scaler,data

## Weighting the nutrients over their importance
def weighting_nutrients(X_scaled,weights=None):
    """
    WIP : Will change the default argument to make it easier to change for probable fine-tuning

    1. Giving weight to some nutrients more than others.
    2. Applying weights to the scaled features to be training the KNN model.
    """
    if weights is None : 
        weights = np.array([1.5, 1.5, 1.5, 0.8, 1.2, 1.5, 1.2, 0.8, 1.5, 1.2])
    # Multiply features after scaling
    # Subjective factor to determine gropingly
    X_weighted=X_scaled*weights
    print('Successfully weighted the recipe/nutrient data')
    return X_weighted


def KNN_model(X_weighted,y):
    """Instantiate and fit a model to predict the recipes that would match the remaining recommended nutrient intakes
    1. Instantiate a KNN Regressor model
    2. Fitting it with a Recipe/Nutrient dataset
    3. Return a trained model
    """
    # Model initialization and fitting
    KNN_regressor_model = KNeighborsRegressor(n_neighbors=10)
    KNN_regressor_model.fit(X_weighted, y)
    print('Successfully intialized and trained the KNN model')
    return KNN_regressor_model

def load_user_data(scaler:StandardScaler,recommended_intake=None,consumed_intake=None):
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
        consumed_intake=np.array([[280. ,   2. , 112. , 420. ,  25. ,   4. , 252. , 151.,  60.,40.]])

    # Calculate the remaining intake
    X_remaining=recommended_intake-consumed_intake

    # Scaling/Weighting the nutrients
    X_remaining_scaled=scaler.transform(X_remaining)
    
    X_remaining_weighted=weighting_nutrients(X_remaining_scaled)
    
    print('Successfully loaded and weighted the user nutritional data')
    return X_remaining_weighted

def predict_KNN_model(KNN_regressor_model:KNeighborsRegressor,X_remaining_weighted):
    """Prediction and Nutrition calculation"""

    # Making sure that X is a 2D array:
    if X_remaining_weighted.shape!=(1,10):
        X_remaining_weighted = X_remaining_weighted.reshape(1, -1)

    # Predicting the most nutritious recipes
    y_pred= KNN_regressor_model.kneighbors(X_remaining_weighted)
    
    print('Successfully predicted the ideal recipes')
    return y_pred


def run_KNN_workflow():
    """
    1. Running back-end the KNN workflow
    2. Generating unprocessed predictions
    3. (TBD) Terminal-displaying unprocessed output (y_pred : List of 2 array -> recipes matching distance to reco-intakes & recipes indexes)
    4. (TBD) Terminal-displaying the selected recipes
    """
    # if not KNN_regressor_model : # KNN_regressor_model is instantiated and fitted once
    X_scaled,y,scaler,data=preprocessing()
    X_weighted=weighting_nutrients(X_scaled)
    KNN_regressor_model=KNN_model(X_weighted,y)
    X_remaining_weighted,=load_user_data(scaler) #Load recommended, consumed and remaining intakes
    y_pred=predict_KNN_model(KNN_regressor_model,X_remaining_weighted) 

    # Displaying to the terminal the raw KNN output
    print("Here are the raw prediction outputs (recipe matching 'distances', and recipe index in the recipes dataset) :",y_pred)

    # Displaying to the terminal the selected recipes
    # (UTD 12/02/2024 - before GitHub push to main) Making a Python object for Streamlit & API pipeline with the recipe names
    print("Here are the selected recipes by order of matching :")
    recommended_recipes_names=[]
    predicted_recipes=y_pred[1][0]
    for i,recipe_index in enumerate(predicted_recipes) :
        print(f"""The recommended recipe n°{i+1} is : {data.iloc[recipe_index]['recipe']}.""") # Printing by matching order the selected recipe
        recommended_recipes_names.append(data.iloc[recipe_index]['recipe']) # Generating the list of recipe names by matching order for later use 
    return y_pred, recommended_recipes_names

y_pred, recommended_recipe_names=run_KNN_workflow() 
print(recommended_recipe_names)


    



