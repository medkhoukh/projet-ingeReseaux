import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM

# 1. Charger les données
file_path = '/Users/simo/Downloads/PRED_TRAFFIC-3/nb_variable_utilisateurs/scenario6/rx_throughput.csv'
data = pd.read_csv(file_path, sep=';')

# 2. Préparer les données
values = data['throughput (kbps)'].values.reshape(-1, 1)
scaler = MinMaxScaler()
scaled_values = scaler.fit_transform(values)

# Fonction pour créer des séquences (X, y)
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

seq_length = 10  # nombre de pas de temps pour la séquence
X, y = create_sequences(scaled_values, seq_length)

# Séparer en train/test
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 3. Créer le modèle NN (LSTM)
model = Sequential([
    LSTM(50, activation='relu', input_shape=(seq_length, 1)),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# 4. Entraîner le modèle
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# 5. Prédire sur le test set
predictions = model.predict(X_test)

# Inverser la normalisation
predicted_traffic = scaler.inverse_transform(predictions)
real_traffic = scaler.inverse_transform(y_test)

# 6. Plot
plt.figure(figsize=(12, 6))
plt.plot(real_traffic, label='Données réelles')
plt.plot(predicted_traffic, label='Prédictions NN')
plt.title('Comparaison du trafic réel et prédit (NN)')
plt.xlabel('Temps (échantillons)')
plt.ylabel('Throughput (kbps)')
plt.legend()
plt.tight_layout()
plt.show()

# Pour exécuter :
# pip install numpy pandas matplotlib scikit-learn tensorflow
# python3.11 predict_traffic_nn.py 