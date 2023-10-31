import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load and preprocess the data
styles_df = pd.read_csv('styles.csv')
purchase_history_df = pd.read_csv('purchase_history.csv')

# Merge the datasets
merged_data = pd.merge(purchase_history_df, styles_df, left_on='ImageID', right_on='id')

# Encode categorical variables
label_encoders = {}
categorical_cols = ['gender', 'season', 'usage']
for col in categorical_cols:
    label_encoders[col] = LabelEncoder()
    merged_data[col] = label_encoders[col].fit_transform(merged_data[col])

# Define user input
user_id = 1999  # Replace with the user's actual ID
user_gender = 'Men'  # Replace with the user's actual gender
user_season = 'Summer'  # Replace with the user's actual season
user_usage = 'Casual'  # Replace with the user's actual usage

# Encode user input
user_gender_encoded = label_encoders['gender'].transform([user_gender])[0]
user_season_encoded = label_encoders['season'].transform([user_season])[0]
user_usage_encoded = label_encoders['usage'].transform([user_usage])[0]

# Prepare input and output data
X = merged_data[['gender', 'season', 'usage']]
Y = merged_data['ImageID']

# Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42)

# Sequence length for user purchase history
sequence_length = 10  # Adjust as needed

# Prepare sequences for user purchase history
user_sequences = []
user_ids = []

for user_id, group in merged_data.groupby('UserID'):
    user_sequence = group['ImageID'].values[-sequence_length:]
    user_sequences.append(user_sequence)
    user_ids.append(user_id)

user_sequences = pad_sequences(user_sequences, maxlen=sequence_length, padding='pre')

# Filter Y_train based on user IDs
Y_train_filtered = Y_train[Y_train.index.isin(user_ids)]

# Build the RNN model
model = Sequential()
model.add(LSTM(100, input_shape=(sequence_length, 1)))
model.add(Dense(1, activation='linear'))

model.compile(loss='mean_squared_error', optimizer='adam')

# Ensure that Y_train_filtered has the same length as user_sequences
Y_train_filtered = Y_train_filtered.values

# Train the RNN model
model.fit(user_sequences, Y_train_filtered, validation_data=(user_sequences, Y_val), epochs=10, batch_size=64)

# Make recommendations for the user based on the last user_sequence
user_sequence = user_sequences[user_id]  # Replace 'user_id' with the actual user's index
user_sequence = user_sequence.reshape(1, sequence_length, 1)
predicted_image_id = model.predict(user_sequence)

# Post-process recommendations, e.g., remove items already purchased by the user
recommended_items = merged_data[merged_data['ImageID'] != predicted_image_id[0][0]]['productDisplayName'].unique()

# Display the recommendations to the user
print("Recommended items for the user:")
print(recommended_items)
