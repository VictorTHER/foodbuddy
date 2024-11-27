import pandas as pd
import numpy as np
import pandas as pd
import re #import regex for text data cleaning
import ast #package whose literal_eval method will help us remove the string type

#Load data
data=pd.read_csv("./raw_data/new_crawl_recipe(basic).csv")
df=data.copy()
df.drop(columns=['directions','rating'])

#Function
def parse_ingredient(ingredient):
    """
    Input : Ingredient cell of a recipe row from the recipe dataset
    Output : A dataframe-like dictionary where each each pair will be the future column/cell value of each ingredient from the same recipe, where :
        'quantity': How many portion
        'grammage': Grammage depending on the unit provided
        'unit': Unit of the grammage
        'name': Name of the ingredient
    Purpose : Output will be used to generate a proper dataframe using pd.DataFrame
    """
    VALID_UNITS = {'g', 'tbsp', 'tsp', 'tspn', 'cup', 'ml', 'l', 'kg', 'oz', 'unit'}
    # Preprocessing to remove "/xoz" patterns and fractions like "½oz" when g is already provided
    ingredient = re.sub(r'/\d+oz', '', ingredient)  # Remove patterns like "/9oz"
    ingredient = re.sub(r'/\d+[½⅓¼¾]+oz', '', ingredient)  # Remove fractions before "oz"

    # Regex to capture quantity, unit, and name
    pattern = r'(?:(\d+)(?:\s*x\s*(\d+))?)?\s*([a-zA-Z%½⅓¼]+)?\s*(.*)'
    match = re.match(pattern, ingredient)

    if match:
        quantity, sub_quantity, unit, name = match.groups()

        # Default values
        grammage = None
        portion_quantity = 1  # Default quantity if not provided

        # Handle the case of "2 x 80g"
        if sub_quantity:
            portion_quantity = int(quantity)
            grammage = int(sub_quantity)
        elif quantity and unit:
            grammage = int(quantity)
        elif quantity:
            portion_quantity = int(quantity)

        # If no grammage or unit is provided
        if not unit and not grammage:
            unit = 'unit'  # Default unit
            name = ingredient.strip()  # Full ingredient name as name
        # Debugging exception : Handling cases where the name has no quantity -> Otherwise first word would be wrongly interpreted as unit
        if unit and not grammage and not quantity:
            name = f"{unit} {name}"  # Combine the unit back into the name
            unit = 'unit'  # Clear the unit column

        return {
            'quantity': portion_quantity,
            'grammage': grammage,
            'unit': unit,
            'name': name.strip()
        }
    return {
        'quantity': 1,
        'grammage': None,
        'unit': 'unit',
        'name': ingredient.strip()
    }


"""

TODO 27/11

- Créer une fonction main
- Y mettre tout le code en dessous

- Elle sera called dans un notebook avec "from recipe import main; dico=main ..."

"""

#Generate a dictionary where indexes are recipe names :
dict_recipes={}
for i,row in df.iterrows():
    ingredients = ast.literal_eval(row['ingredients'])
    parsed_ingredients=[parse_ingredient(ingredient) for ingredient in ingredients] #processed ingredients
    dict_recipes[row['title']]=pd.DataFrame(parsed_ingredients) #index=ingredient name -> For querying image-recognized recipes
# print(dict_recipes)

# # Display the result in the terminal
# for key, value in dict_recipes.items():
#     print(f"\nRecipe index: {key}")
#     print(value)
