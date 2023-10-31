import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import streamlit as st

# Load and preprocess the data
styles_df = pd.read_csv('styles.csv')
purchase_history_df = pd.read_csv('purchase_history.csv')

# Encode categorical variables
label_encoders = {}
categorical_cols = ['gender', 'season', 'usage', 'baseColour']
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    styles_df[col] = label_encoders[col].fit_transform(styles_df[col])

# Streamlit UI for user input
st.title("Outfit Recommendations")
st.write("Enter your preferences:")

user_id = st.text_input("User ID")


genders = ['Men', 'Women', 'Boys', 'Girls', 'Unisex']
gender = st.selectbox("Gender", genders)

occasions = ['Casual', 'Ethnic', 'Formal', 'Sports', 'Smart Casual', 'Travel', 'Party', 'Home']
occasion = st.selectbox("Occasion", occasions)

seasons = ['Fall', 'Summer', 'Winter', 'Spring']
season = st.selectbox("Season", seasons)

unique_colors = styles_df['baseColour'].unique()
preferred_color = st.selectbox("Preferred Color", unique_colors)


# Function to get recommendations
def get_outfit_recommendations(gender, occasion, season, preferred_color):
    # Encode user input
    user_gender_encoded = label_encoders['gender'].transform([gender])[0]
    user_season_encoded = label_encoders['season'].transform([season])[0]
    user_usage_encoded = label_encoders['usage'].transform([occasion])[0]
    user_preferred_color_encoded = label_encoders['baseColour'].transform([preferred_color])[0]

    # Prepare input data
    user_input_data = np.array([[user_gender_encoded, user_season_encoded, user_usage_encoded, user_preferred_color_encoded]])

    # Build the model
    user_input = Input(shape=(4,))
    output_layer = Dense(1, activation='linear')(user_input)
    model = Model(inputs=user_input, outputs=output_layer)
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Training the model is required here, but I'll use a dummy dataset
    # You should replace this with your actual training data
    X = styles_df[['gender', 'season', 'usage', 'baseColour']]
    Y = styles_df['id']
    model.fit(X, Y, epochs=10, batch_size=64)

    # Make 50 recommendations based on the user's preferences
    recommendations = model.predict(user_input_data)
    
    # If the user has previous history, get 10 recommendations based on history
    user_history = purchase_history_df[purchase_history_df['UserID'] == int(user_id)]
    if not user_history.empty:
        user_history_items = user_history['ImageID']
        recommended_items = styles_df[~styles_df['id'].isin(user_history_items)]['id'].head(10)
    else:
        recommended_items = styles_df['id'].sample(10)
    
    return recommendations, recommended_items

# Button to trigger recommendations
if st.button("Generate Outfit Recommendations"):
    recommendations, recommended_items = get_outfit_recommendations(gender, occasion, season, preferred_color)
    
    st.title("Top 50 Recommendations:")
    st.write(recommendations)  # Display the top 50 recommendations

    st.title("Top 10 Recommendations (Based on User History if Available):")
    st.write(recommended_items)  # Display the top 10 recommendations based on user history

