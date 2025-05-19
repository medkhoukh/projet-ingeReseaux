import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.stattools import adfuller

# 1. Charger les données
file_path = '/Users/simo/Downloads/PRED_TRAFFIC-3/nb_variable_utilisateurs/scenario6/rx_throughput.csv' 
data = pd.read_csv(file_path, sep=';')

# 2. Préparer les données
values = data['throughput (kbps)'].values
time_values = data['time'].values  # Stocker les valeurs de temps réelles

# 3. Tester la stationnarité
result = adfuller(values)
print(f'Test ADF p-value: {result[1]}')
print(f'La série est {"stationnaire" if result[1] < 0.05 else "non-stationnaire"}')

# 4. Utiliser un échantillon pour accélérer les calculs
sample_size = 1000
train_size = int(0.7 * sample_size)

# Éventuellement, différencier la série si nécessaire
diff = False
if result[1] > 0.05:  # Série non stationnaire
    print("Application d'une différenciation pour rendre la série stationnaire")
    diff = True
    values_diff = np.diff(values[:sample_size])
    train, test = values_diff[:train_size-1], values_diff[train_size-1:]
    # Valeurs originales pour la reconstruction
    train_orig, test_orig = values[:sample_size][:train_size], values[:sample_size][train_size:]
    # Valeurs de temps correspondantes
    time_train, time_test = time_values[:sample_size][:train_size], time_values[:sample_size][train_size:]
else:
    train, test = values[:sample_size][:train_size], values[:sample_size][train_size:]
    # Valeurs de temps correspondantes
    time_train, time_test = time_values[:sample_size][:train_size], time_values[:sample_size][train_size:]

# 5. Définir les paramètres ARMA
# p: ordre autorégressif, q: ordre moyenne mobile
p, q = 5, 2  # Ces valeurs doivent être ajustées selon vos données

# 6. Créer et ajuster le modèle ARMA (ARIMA sans différenciation d=0)
if diff:
    # ARIMA avec d=0 sur données différenciées
    model = ARIMA(train, order=(p, 0, q))
else:
    # ARIMA avec d=0 sur données originales
    model = ARIMA(train, order=(p, 0, q))

model_fit = model.fit()
print(model_fit.summary())

# 7. Prédiction pas à pas pour ARMA
predictions = []
history = list(train)

# Prédire un pas à la fois
for t in range(len(test)):
    # Ajuster le modèle à l'historique
    if diff:
        model = ARIMA(history, order=(p, 0, q))
        model_fit = model.fit()
        # Prédire le prochain changement
        output = model_fit.forecast()[0]
        # Ajouter le changement prédit à la dernière valeur originale
        if len(predictions) == 0:
            # Premier point
            yhat = train_orig[-1] + output
        else:
            # Points suivants
            yhat = predictions[-1] + output
        obs = test_orig[t]  # Valeur originale
    else:
        model = ARIMA(history, order=(p, 0, q))
        model_fit = model.fit()
        output = model_fit.forecast()[0]
        yhat = output
        obs = test[t]
    
    predictions.append(yhat)
    # Ajouter l'observation à l'historique pour la prochaine itération
    if diff:
        history.append(test[t])  # Ajouter la différence observée
    else:
        history.append(test[t])
    
    # Afficher la progression
    if t % 20 == 0:
        print(f'Prédiction {t}/{len(test)}')

# 8. Évaluer les performances
mse = mean_squared_error(test_orig if diff else test, predictions)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_orig if diff else test, predictions)

print('Performances du modèle ARMA')
print(f'MSE: {mse:.2f}')
print(f'RMSE: {rmse:.2f}')
print(f'MAE: {mae:.2f}')

# 9. Visualiser les résultats avec axe de temps réel
plt.figure(figsize=(12, 6))
plt.plot(time_test, test_orig if diff else test, label='Données réelles')
plt.plot(time_test, predictions, label='Prédictions ARMA', color='red')
plt.title('Prédiction du trafic avec modèle ARMA')
plt.xlabel('Temps (secondes)')
plt.ylabel('Throughput (kbps)')
plt.legend()
plt.tight_layout()
plt.savefig('arma_predictions.png')
plt.show()

# Zoom sur une section des données avec axe de temps réel
plt.figure(figsize=(12, 6))
zoom_length = min(100, len(test))
plt.plot(time_test[:zoom_length], test_orig[:zoom_length] if diff else test[:zoom_length], label='Données réelles')
plt.plot(time_test[:zoom_length], predictions[:zoom_length], label='Prédictions ARMA', color='red')
plt.title('Zoom sur les 100 premiers points (ARMA)')
plt.xlabel('Temps (secondes)')
plt.ylabel('Throughput (kbps)')
plt.legend()
plt.tight_layout()
plt.savefig('arma_predictions_zoom.png')
plt.show()

# Pour exécuter :
# pip install numpy pandas matplotlib statsmodels scikit-learn
# python predict_traffic_arma_final.py