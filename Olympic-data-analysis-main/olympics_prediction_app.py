from ast import main
from datetime import MAXYEAR, MINYEAR
import streamlit as st
import pandas as pd
import preprocessor
import helper
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier  # Import your chosen model class here

# Load the data and preprocess it
df = pd.read_csv(r'C:\Users\saite\OneDrive\Desktop\archive\athlete_events.csv')
region_df = pd.read_csv(r'C:\Users\saite\OneDrive\Desktop\archive\noc_regions.csv')
df = preprocessor.preprocess(df, region_df)

# Replace these placeholders with your actual data
sport_list_placeholder = df['Sport'].unique().tolist()
country_list_placeholder = df['region'].unique().tolist()
MINYEAR = df['Year'].min()
MAXYEAR = df['Year'].max()
current_year_placeholder = MAXYEAR  # Set it to the maximum year by default

# Streamlit Sidebar
st.sidebar.title("Olympics Analysis")
st.sidebar.image('https://e7.pngegg.com/pngimages/1020/402/png-clipart-2024-summer-olympics-brand-circle-area-olympic-rings-olympics-logo-text-sport.png')
user_menu = st.sidebar.radio(
    'Select an Option',
    ('Medal Tally', 'Overall Analysis', 'Country-wise Analysis', 'Athlete wise Analysis', 'Predict Medal')  # Added 'Predict Medal' option
)

# Function to prepare data for prediction
def prepare_data_for_prediction(selected_sport, selected_country, selected_year):
    # Filter the DataFrame based on selected sport, country, and year
    filtered_df = df[(df['Sport'] == selected_sport) & (df['region'] == selected_country) & (df['Year'] == selected_year)]

    # Extract relevant features for prediction
    feature_vector = filtered_df[['Age', 'Height', 'Weight']]  # Replace with your actual features

    return feature_vector

# Function to make predictions
def make_prediction(model, feature_vector):
    # Replace this with your actual prediction code
    # Example: Assuming model is trained and has a predict method
    prediction = model.predict(feature_vector)
    return prediction

if user_menu == 'Medal Tally':
    st.sidebar.header("Medal Tally")
    years, countries = helper.country_year_list(df)

    selected_year = st.sidebar.selectbox("Select Year", years)
    selected_country = st.sidebar.selectbox("Select Country", countries)

    medal_tally = helper.fetch_medal_tally(df, selected_year, selected_country)

    st.title(f"Medal Tally in {selected_year} Olympics for {selected_country}")
    st.table(medal_tally)

# Code for Overall Analysis (your existing code here)...

if user_menu == 'Country-wise Analysis':
    # Code for Country-wise Analysis
    st.sidebar.title('Country-wise Analysis')

    country_list = df['region'].dropna().unique().tolist()
    country_list.sort()

    selected_country = st.sidebar.selectbox('Select a Country', country_list)

if user_menu == 'Athlete wise Analysis':
    # Code for Athlete-wise Analysis (your existing code here)...

# Add prediction functionality
    if user_menu == 'Predict Medal':
        st.sidebar.header("Predict Medal")
    
    # Collect user inputs required for prediction
    selected_sport_for_prediction = st.sidebar.selectbox("Select Sport", sport_list_placeholder)
    selected_country_for_prediction = st.sidebar.selectbox("Select Country", country_list_placeholder)
    selected_year_for_prediction = st.sidebar.slider("Select Year", MINYEAR, MAXYEAR, current_year_placeholder)  # Adjust min, max, and current year
    
    # Prepare the data for prediction based on user inputs
    feature_vector = prepare_data_for_prediction(selected_sport_for_prediction, selected_country_for_prediction, selected_year_for_prediction)
    
    # Load the trained machine learning model (replace with your model loading logic)
    # Example using RandomForestClassifier
    model = RandomForestClassifier()
    
    # Make predictions based on user inputs
    prediction = make_prediction(model, feature_vector)
    
    # Display the prediction result
    st.title("Medal Prediction Result")
    st.write(f"Predicted Medal for {selected_country_for_prediction} in {selected_year_for_prediction} for {selected_sport_for_prediction}: {prediction}")

# Run the Streamlit app
if __name__ == "__main__":
    st.title("Olympics Data Analysis")
    main()
