import os
import pandas as pd
import numpy as np
import tensorflow as tf
from pycaret.classification import *
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import sys

path = '/home/thiago/projects/deep_learning/Trabalho-de-deep-learning/datasets/data'

if len(sys.argv) == 3:
    encoding_dim = int(sys.argv[1])
    month = int(sys.argv[2])
elif len(sys.argv) == 2:
    encoding_dim = int(sys.argv[1])
    month = 9 # setembro
else:
    encoding_dim = 10
    month = 9 # setembro

data_path_mapping = {
    7: "jul",
    8: "ago",
    9: "set",
}

month_str = data_path_mapping[month]
data_path = f"{path}/{month_str}_2024.csv"

print (f"Using encoding_dim = {encoding_dim}")
print (f"Encoding data from month = {month_str}")

# Define the autoencoder
def build_autoencoder(input_dim, encoding_dim):
    """Build and return an autoencoder model."""
    # Encoder
    input_layer = Input(shape=(input_dim,))
    encoded = Dense(encoding_dim, activation="relu")(input_layer)

    # Decoder
    decoded = Dense(input_dim, activation="sigmoid")(encoded)

    # Autoencoder Model
    autoencoder = Model(inputs=input_layer, outputs=decoded)

    # Encoder Model (for extracting latent representations)
    encoder = Model(inputs=input_layer, outputs=encoded)

    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
    return autoencoder, encoder

def scale_data(data):
    """Scale the data to [0, 1] range."""
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    return scaled_data, scaler



set_data = pd.read_csv(data_path)
#data_concat = pd.concat([set_data, ago_data, jul_data], ignore_index=True)
data_concat = set_data
print("Tamanho do dataset combinado:", data_concat.shape)
data_concat.head()

columns_to_use = [
    "failure",
    "smart_1_normalized", "smart_5_normalized", "smart_7_normalized",  "smart_184_normalized",
    "smart_187_normalized", "smart_188_normalized", "smart_189_normalized", "smart_190_normalized",
    "smart_193_normalized", "smart_194_normalized", "smart_197_normalized", "smart_198_normalized",
    "smart_240_normalized", "smart_241_normalized", "smart_242_normalized"
]

data = data_concat[columns_to_use]

print("Dados combinados carregados:")
print(data.head())
print("Tamanho do dataset combinado:", data.shape)

train_data = data.sample(frac=0.8, random_state=42)  # 80% para treinamento
test_data = data.drop(train_data.index)              # 20% para teste

print(f"Tamanho do conjunto de treinamento: {train_data.shape}")
print(f"Tamanho do conjunto de teste: {test_data.shape}")

smart_cols = [c for c in data_concat.columns if c.startswith('smart')]
std_smart_cols = [np.std(data_concat[c]) for c in smart_cols]
columns_to_use = [c for c, std in zip(smart_cols, std_smart_cols) if std > 0] + ['failure']

data = data_concat[columns_to_use]

data = data.select_dtypes(include=[np.number]).dropna()

scaled_data, scaler = scale_data(data)
data = pd.DataFrame(scaled_data, columns=data.columns, index=data.index)


print("Dados combinados carregados:")
print(data.head())
print("Tamanho do dataset combinado:", data.shape)

train_data = data.sample(frac=0.8, random_state=42)  # 80% para treinamento
test_data = data.drop(train_data.index)

print('Train')
print(len(train_data[train_data['failure']==1]))
print(len(train_data[train_data['failure']==1])/len(train_data)*100)

print('Test')
print(len(test_data[test_data['failure']==1]))
print(len(test_data[test_data['failure']==1])/len(test_data)*100)

# to simplify
X_train = train_data
X_test = test_data

print(f"X_train.shape = {X_train.shape}")
print(f"X_test.shape = {X_test.shape}")

# Autoencoder parameters
input_dim = X_train.shape[1]

# Build the autoencoder
autoencoder, encoder = build_autoencoder(input_dim, encoding_dim)

# Train the autoencoder
autoencoder.fit(
    X_train, X_train,
    epochs=5,
    batch_size=32,
    shuffle=True,
    validation_data=(X_test, X_test)
)

# Extract latent representations
latent_representations = encoder.predict(X_train)

# Save the latent representations to a CSV file
folder = f"latent/{month_str}"
filepath = f"{folder}/{month_str}_latent_{encoding_dim}.csv"
latent_df = pd.DataFrame(latent_representations, columns=[f"Feature_{i}" for i in range(encoding_dim)])
if not os.path.isdir(folder):
    os.mkdir(folder)

latent_df.to_csv(filepath, index=False)

print(f"Latent representations saved to {filepath}")
