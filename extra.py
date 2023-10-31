import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
import streamlit as st

# Load and preprocess the data
styles_df = pd.read_csv('styles.csv')
images_df = pd.read_csv('images.csv')
purchase_history_df = pd.read_csv('purchase_history.csv')


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

# Encode categorical variables
label_encoders = {}
categorical_cols = ['gender', 'season', 'usage', 'baseColour']
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    styles_df[col] = label_encoders[col].fit_transform(styles_df[col])

# Encode user input
user_gender_encoded = label_encoders['gender'].transform([gender])[0]
user_season_encoded = label_encoders['season'].transform([season])[0]
user_usage_encoded = label_encoders['usage'].transform([occasion])[0]
user_preferred_color_encoded = label_encoders['baseColour'].transform([preferred_color])[0]

# Function to get recommendations
def get_outfit_recommendations():
    # Build the model
    user_input = Input(shape=(4,))  # Four features for user input (gender, season, usage, preferred_color)
    output_layer = Dense(1, activation='linear')(user_input)

    model = Model(inputs=user_input, outputs=output_layer)
    model.compile(loss='mean_squared_error', optimizer='adam')

    # Prepare input and output data
    merged_data = pd.merge(purchase_history_df, styles_df, left_on='ImageID', right_on='id')
    X = merged_data[['gender', 'season', 'usage', 'baseColour']]
    Y = merged_data['ImageID']

    # Split the data into training and validation sets
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Train the model
    model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=10, batch_size=64)

    # Make recommendations for the user using the model
    user_input_data = np.array([[user_gender_encoded, user_season_encoded, user_usage_encoded, user_preferred_color_encoded]])
    predicted_image_id = model.predict(user_input_data)

    # Filter out items based on user's preferences
    filtered_items = merged_data
    if user_gender_encoded != -1:
        filtered_items = filtered_items[filtered_items['gender'] == user_gender_encoded]
    if user_preferred_color_encoded != -1:
        filtered_items = filtered_items[filtered_items['baseColour'] == user_preferred_color_encoded]
    if user_season_encoded != -1:
        filtered_items = filtered_items[filtered_items['season'] == user_season_encoded]
    if user_usage_encoded != -1:
        filtered_items = filtered_items[filtered_items['usage'] == user_usage_encoded]

    # Remove items already purchased by the user
    user_purchased_items = purchase_history_df[purchase_history_df['UserID'] == int(user_id)]['ImageID'].unique()
    recommended_items = filtered_items[~filtered_items['ImageID'].isin(user_purchased_items)][['id', 'productDisplayName']]

    return recommended_items

# Button to trigger recommendations
if st.button("Generate Outfit Recommendations"):
    recommended_items = get_outfit_recommendations()

    st.title("Recommended items for the user:")
    
    if recommended_items.empty:
        st.write("No items matching your preferences were found.")
        st.write("Here are some items based on your preferences")
        user_gender_encoded = label_encoders['gender'].transform([gender])[0]
        user_preferred_color_encoded = label_encoders['baseColour'].transform([preferred_color])[0]
        # Find items based on the user's preferred gender and color
        similar_items = styles_df[(styles_df['gender'] == user_gender_encoded) & (styles_df['baseColour'] == user_preferred_color_encoded)][['id', 'productDisplayName']]
        user_purchased_items = purchase_history_df[purchase_history_df['UserID'] == int(user_id)]['ImageID'].unique()
        similar_items = similar_items[~similar_items['id'].isin(user_purchased_items)]

        if not similar_items.empty:
            st.title("Similar items based on your preferences:")
            # Display similar items with images
            count = 0  # Initialize a count
            unique_item_ids = set()  # Initialize a set to track unique items

            # Create 2 columns to display images side by side
            col1, col2 = st.columns(2)

            for _, row in similar_items.iterrows():
                if count >= 10:  # Display only the top 10 similar items
                    break

                item_id = row['id']
                
                # Check if the item_id is already displayed, skip if it's a duplicate
                if item_id in unique_item_ids:
                    continue

                product_display_name = row['productDisplayName']
                
                # Check if the image link exists in the images.csv file
                image_link_row = images_df[images_df['filename'] == f"{item_id}.jpg"]
                if not image_link_row.empty:
                    image_link = image_link_row['link'].values[0]
                    
                    # Add the image to the appropriate column
                    if count % 2 == 0:
                        with col1:
                            st.image(image_link, use_column_width=True, caption=f"Product ID: {item_id}")
                    else:
                        with col2:
                            st.image(image_link, use_column_width=True, caption=f"Product ID: {item_id}")
                    
                    count += 1
                    unique_item_ids.add(item_id)  # Add the item_id to the set of unique items

                else:
                        st.write("No similar items found based on your preferences.")
    else:
        # Display recommended items with images (top 10 unique products)
        count = 0  # Initialize a count
        unique_item_ids = set()  # Initialize a set to track unique items

        # Create 2 columns to display images side by side
        col1, col2 = st.columns(2)

        for _, row in recommended_items.iterrows():
            if count >= 10:  # Display only the top 10 products
                break

            item_id = row['id']
            
            # Check if the item_id is already displayed, skip if it's a duplicate
            if item_id in unique_item_ids:
                continue

            product_display_name = row['productDisplayName']
            
            # Check if the image link exists in the images.csv file
            image_link_row = images_df[images_df['filename'] == f"{item_id}.jpg"]
            if not image_link_row.empty:
                image_link = image_link_row['link'].values[0]
                
                # Add the image to the appropriate column
                if count % 2 == 0:
                    with col1:
                        st.image(image_link, use_column_width=True, caption=f"Product ID: {item_id}")
                else:
                    with col2:
                        st.image(image_link, use_column_width=True, caption=f"Product ID: {item_id}")
                
                count += 1
                unique_item_ids.add(item_id)  # Add the item_id to the set of unique items
            else:
                # Handle the case when the image link is not found
                st.write(product_display_name)
                st.write("Image not found for Product ID:", item_id)
                count += 1
