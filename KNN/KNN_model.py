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

from KNN.KNN_preprocess import load_recipes_KNN_data


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
    print('Successfully preprocessed and standardized the recipe/nutrient data.')
    print('Successfully instantiated the nutrient scaler future user inputs.')
    return X_scaled,y,scaler

## Weighting the nutrients over their importance
def weighting_nutrients(X_scaled,weights=None):
    """
    WIP : Will change the default argument to make it easier to change for probable fine-tuning

    1. Giving weight to some nutrients more than others.
    2. Applying weights to the scaled features to be training the KNN model.
    """
    if weights is None :
        weights = np.array([0.8, 1.5, 1.2, 1.5, 1.5, 1.5, 0.8, 1.2, 1.2, 1.5])
    # Multiply features after scaling
    # Subjective factor to determine gropingly
    X_weighted=X_scaled*weights
    print('Successfully weighted the recipe/nutrient data')
    return X_weighted


def KNN_model():
    """MAIN FUNCTION"""

    """Instantiate and fit a model to predict the recipes that would match the remaining recommended nutrient intakes
    1. Instantiate a KNN Regressor model
    2. Fitting it with a Recipe/Nutrient dataset
    3. Return a trained model
    4. Save fitted model into a pkl file that will be called thereafter for the user's 'predictions' (recipe recommendations)
    """
    # Running the previous functions
    X_scaled,y,scaler=preprocessing()
    X_weighted=weighting_nutrients(X_scaled)

    # Model initialization and fitting
    model = KNeighborsRegressor(n_neighbors=10)
    model.fit(X_weighted, y)
    print('Successfully intialized and trained the KNN model')

    # Saving the fitted model into a .pkl file
    # fitted_model_path="./KNN/fitted_model.pkl"
    with open('fitted_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print(f"Processed dataset saved at foodbuddy/KNN.")

    # Saving the fitted scaler into a .pkl file
    # fitted_scaler_path="./KNN/fitted_scaler.pkl"
    with open('fitted_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print(f"Fitted scaler saved at foodbuddy/KNN.")

    # """Update 12/03/2024 : Recipe names have to be called during the predictions
    # => Fix : Saving the target into a csv, that will be called for indexation in the final recommendation output"""
    # y.to_csv()

    return None

if __name__=='__main__':
    KNN_model()
