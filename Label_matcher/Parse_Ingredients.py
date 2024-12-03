import pandas as pd
import numpy as np
import pandas as pd
import re #import regex for text data cleaning
import ast #package whose literal_eval method will help us remove the string type


def parse_ingredient_original(ingredient):
    """
    Input : Ingredient cell of a recipe row from the recipe dataset
    Output : A dataframe-like dictionary where each each pair will be the future column/cell value of each ingredient from the same recipe, where :
        'quantity': How many portion
        'grammage': Grammage depending on the unit provided
        'unit': Unit of the grammage
        'name': Name of the ingredient
    Purpose : Output will be used to generate a proper dataframe using pd.DataFrame
    """
    VALID_UNITS = {'g', 'tbsp', 'tsp', 'tspn', 'cup', 'ml', 'l', 'kg', 'oz', 'fl oz'}
    # Preprocessing to remove "/xoz" patterns and fractions like "½oz" when g is already provided
    ingredient = re.sub(r'/\d+oz', '', ingredient)  # Remove patterns like "/9oz"
    ingredient = re.sub(r'/\d+fl oz', '', ingredient)  # Remove patterns like "/9fl oz"
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
            name = ingredient.strip()  # Full ingredient name as name

        # Debugging exception : Handling cases where the detected unit is actually the first word of the ingredient name
        if unit and unit not in VALID_UNITS:
            # Move the incorrectly detected unit back into the beggining of the name
            name = f"{unit} {name}".strip()
            unit = None  # Clear the unit, since it's invalid

        # Exception when a fraction of quantity is provided
        # Output example before fixing : 1       NaN     unit  ½ leftover roast chicken, torn into pieces
        # Fix : Check if a fraction is at the beginning of the name and adjust quantity
        fraction_pattern = r'^([½⅓¼¾])\s*(.*)'
        fraction_match = re.match(fraction_pattern, name)
        if fraction_match and portion_quantity == 1:
            fraction, remaining_name = fraction_match.groups()
            try:
                # Fraction to decimal dictionary
                fraction_value = {
                    "½": 0.5,
                    "⅓": 0.33,
                    "¼": 0.25,
                    "¾": 0.75
                }[fraction]
                portion_quantity = fraction_value  # Replacing quantity with the decimal
                name = remaining_name.strip()  # Removing the fraction from the name
            except KeyError:
                pass  # Keep running the code if error

        return {
            'quantity': float(portion_quantity),
            'grammage': grammage,
            'unit': unit,
            'name': name.strip()
        }
    # if no pattern is recognized eventually -> Default return
    return {
        'quantity': 1,
        'grammage': None,
        'unit': None,
        'name': ingredient.strip()
    }


def ingredients_per_recipe_dictionary(data):
    """
    1. Process the raw recipe table.
    2. Generate a dictionary storing the nutrients-per-ingredient datasets for each recipe, by calling the parse_ingredients function.
    """
    #Generate a dictionary where indexes are recipe names :
    dict_recipes={}
    for i,row in data.iterrows():
        ingredients = ast.literal_eval(row['ingredients'])
        parsed_ingredients=[parse_ingredient(ingredient) for ingredient in ingredients] # structured ingredient information for each recipe are stored in dictionaries
        dict_recipes[row['title']]=pd.DataFrame(parsed_ingredients) #index=ingredient name -> For querying image-recognized recipes
    return dict_recipes
