#Package import
import pandas as pd
import numpy as np
# np.set_printoptions(legacy='1.25') # Making sure float and integers won't show as 'np.float(64)', etc. 
import pandas as pd

from KNN_to_predictions import run_KNN_workflow 


"""WORK IN PROGRESS : 

1. After-recommended-lunch nutrition validation
2. Data display to the user (To Be Aligned with API project work)

"""

y_pred,recommended_recipes_names=run_KNN_workflow()

print(y_pred)

""" Under the hood : Nutrition fulfilling validation 

1. (Ceiling Threshold) Are some recipes overreaching the recommended intakes ?
2. (Floor Threshold) Are there still some nutrient deficiency after taking a recommended recipe ?
3. Cleaning the recipes, then give recommended
4. Recommend a top 3 or 5 list of recipes to the user

"""


"""Recommendation displaying """

# Processing the models output
# Validate coherence


# Showing the users the recommended recipe 
## 
# data.iloc[recipe_index]    