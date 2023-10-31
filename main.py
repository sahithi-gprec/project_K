import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical

# Load and preprocess the data
styles_df = pd.read_csv('styles.csv')
purchase_history_df = pd.read_csv('purchase_history.csv')

# Create a LabelEncoder for categorical variables
label_encoders = {}
categorical_columns = ['gender', 'season', 'usage']

for column in categorical_columns:
    le = LabelEncoder()
    le.fit(styles_df[column])
    styles_df[column] = le.transform(styles_df[column])
    label_encoders[column] = le

# Merge data to associate user attributes with purchase history
merged_df = pd.merge(purchase_history_df, styles_df, left_on='ImageID', right_on='id', how='inner')

# Sort the data by UserID and Timestamp (if available)
merged_df.sort_values(['UserID'], inplace=True)

# Create sequences of purchases for each user
sequences = []
user_ids = merged_df['UserID'].unique()

for user_id in user_ids:
    user_data = merged_df[merged_df['UserID'] == user_id]
    user_sequence = user_data['id'].tolist()
    sequences.append(user_sequence)

# Convert the sequences into input and target data for the RNN
X = [sequence[:-1] for sequence in sequences]
y = [sequence[1:] for sequence in sequences]

# Pad sequences to a fixed length (you can choose an appropriate length)
sequence_length = 10
X = tf.keras.preprocessing.sequence.pad_sequences(X, maxlen=sequence_length, padding='post')
y = tf.keras.preprocessing.sequence.pad_sequences(y, maxlen=sequence_length, padding='post')

# Create an RNN model
model = Sequential()
model.add(Embedding(input_dim=len(styles_df), output_dim=32, input_length=sequence_length))
model.add(LSTM(64, return_sequences=True))
model.add(Dense(len(styles_df), activation='softmax', input_dim=32))  # Change output size to match the number of products

model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=0.001))

# Ensure your target data (y) is one-hot encoded
y_onehot = [to_categorical(seq, num_classes=len(styles_df)) for seq in y]

# Train the model
model.fit(X, np.array(y_onehot), epochs=10, batch_size=32)


# Make recommendations for a user (e.g., User 2)
user_id = 2
user_data = styles_df[styles_df['id'].isin(sequences[user_id])]
user_attributes = user_data[['gender', 'season', 'usage']]
user_attributes_encoded = np.array([label_encoders[column].transform(user_attributes[column]) for column in categorical_columns])
recommended_sequence = model.predict(user_attributes_encoded.reshape(1, -1))

# Find the top-K recommended product IDs
K = 5
recommended_product_ids = np.argsort(recommended_sequence[0, -1, :])[-K:]

# Print the recommended product IDs
print("Recommended Product IDs:", recommended_product_ids)
