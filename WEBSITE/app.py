import streamlit as st
import pandas as pd
import requests
from PIL import Image
import toml
import time
import random

# Load the .toml file
config = toml.load("WEBSITE/.secrets.toml")
MODEL_API_URL = config["model"]["MODEL_API_URL"]

### TITLE AND DESCRIPTION ###
st.markdown(
    """
    <h1 style='text-align: center;'>
    FoodBuddy™ Calculator
    </h1>
    """,
    unsafe_allow_html=True,
)
st.text("Welcome to FoodBuddy™! Our unique model can analize your meal and let you know its nutritional intake.")

# Food picture
st.image("WEBSITE/HealthyMeal.jpg", caption="Healthy Meal!")

### STEP 1: USER DETAILS FORM ###
st.header("Step 1: Personal Details")

# Collect user inputs
age = st.number_input("Age (years)", min_value=1, max_value=120, step=1)
gender = st.radio("Gender", ["Male", "Female"])
weight = st.number_input("Weight (kg)", min_value=1, max_value=200, step=1)
height = st.number_input("Height (cm)", min_value=50, max_value=250, step=1)

st.header("Activity Level")
activity_level = st.selectbox(
    "Choose your activity level",
    [
        "Sedentary (little or no exercise)",
        "Lightly active (light exercise/sports 1-3 days/week)",
        "Moderately active (moderate exercise/sports 3-5 days/week)",
        "Very active (hard exercise/sports 6-7 days a week)",
        "Super active (very hard exercise/physical job)",
    ],
)

# Map activity levels to multipliers
activity_multipliers = {
    "Sedentary (little or no exercise)": 1.2,
    "Lightly active (light exercise/sports 1-3 days/week)": 1.375,
    "Moderately active (moderate exercise/sports 3-5 days/week)": 1.55,
    "Very active (hard exercise/sports 6-7 days a week)": 1.725,
    "Super active (very hard exercise/physical job)": 1.9,
}

if st.button("Calculate"):
    # Calculate BMR using Harris-Benedict equation
    if gender == "Male":
        bmr = 88.362 + (13.397 * weight) + (4.799 * height) - (5.677 * age)
    else:
        bmr = 447.593 + (9.247 * weight) + (3.098 * height) - (4.330 * age)

    # Adjust for activity level
    activity_multiplier = activity_multipliers[activity_level]
    daily_caloric_needs = bmr * activity_multiplier

    # Macro split percentages
    macros = {
        "Carbohydrates": 0.5,  # 50% of total calories
        "Proteins": 0.2,       # 20% of total calories
        "Fats": 0.3,           # 30% of total calories
    }

    # Calculate intake for each category
    nutritional_intake = {
        "Category": ["Carbohydrates", "Proteins", "Fats"],
        "Percentage of Calories": [macros["Carbohydrates"], macros["Proteins"], macros["Fats"]],
        "Calories (kcal)": [
            daily_caloric_needs * macros["Carbohydrates"],
            daily_caloric_needs * macros["Proteins"],
            daily_caloric_needs * macros["Fats"],
        ],
        "Grams per Day": [
            (daily_caloric_needs * macros["Carbohydrates"]) / 4,  # 1g of carbs = 4 kcal
            (daily_caloric_needs * macros["Proteins"]) / 4,       # 1g of protein = 4 kcal
            (daily_caloric_needs * macros["Fats"]) / 9,           # 1g of fat = 9 kcal
        ],
    }

    # Convert to a DataFrame
    df = pd.DataFrame(nutritional_intake)

    # Display results
    st.subheader("Your Daily Nutritional Intake")
    st.write(f"**Basal Metabolic Rate (BMR):** {bmr:.2f} kcal/day")
    st.write(f"**Total Daily Caloric Needs:** {daily_caloric_needs:.2f} kcal/day")
    st.dataframe(df)

    st.markdown(
        "_Note: These calculations are estimates and may vary based on individual needs._"
    )



### STEP 2: PHOTO UPLOAD ###
st.header("Step 2: Upload a Photo")

# Split into two columns
col1, col2 = st.columns([1, 1])

with col1:
    # Create a placeholder for the upload button
    upload_placeholder = st.empty()
    uploaded_file = upload_placeholder.file_uploader(
        "Upload a photo of your meal", type=["jpg", "png", "jpeg"]
    )

    if uploaded_file:
        # Hide the upload button
        upload_placeholder.empty()

        # Show the uploaded image
        uploaded_image = Image.open(uploaded_file)
        st.image(
            uploaded_image,
            caption="Uploaded Image",
            use_container_width=True,
        )

        # Add a mask or overlay text to indicate processing
        st.markdown(
            """
            <div style="text-align:center; background: rgba(0, 0, 0, 0.6); color: white; padding: 10px; margin-top: -50px;">
            <strong>Analyzing...</strong>
            </div>
            """,
            unsafe_allow_html=True,
        )

with col2:
    st.subheader("Processing Status")

    if uploaded_file:
        # Placeholder for cumulative updates
        status_placeholder = st.empty()

        # List to store messages
        status_messages = []

        # Simulate processing steps with random delays
        steps = [
            "Analyzing your meal...",
            "Identifying ingredients...",
            "Checking nutritional value...",
            "Recommending recipes...",
            "Finalizing results...",
        ]

        for step in steps:
            # Simulate a random delay
            time.sleep(random.uniform(0, 5))

            # Add the new step to the list of messages
            status_messages.append(f"✅ {step}")

            # Add line breaks and update placeholder
            formatted_messages = "<br>".join(status_messages)
            status_placeholder.markdown(
                f"<div style='line-height: 1.8;'>{formatted_messages}</div>",
                unsafe_allow_html=True,
            )

        # Final success message
        st.error("[ERR0R: No response from FoodBuddy™ API]")
