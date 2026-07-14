"""
Étape 3b : Apprentissage du réseau de neurones avec l'architecture
optimale déterminée à l'étape 3a.

Architecture retenue : fenêtre = 2, couches cachées = (20, 10)
(à adapter ci-dessous si votre résultat de l'étape 3a diffère)
"""
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import joblib
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_TEST = 30       # dernières valeurs réservées (jamais vues à l'entraînement)
WINDOW = 2        # <-- résultat de l'étape 3a
HIDDEN_LAYERS = (20, 10)  # <-- résultat de l'étape 3a

# ---------------------------------------------------------------
# 1. Chargement des données
# ---------------------------------------------------------------
data = np.loadtxt("henon_data.csv", delimiter=",", skiprows=1)
x = data[:, 0]

x_train_full = x[: len(x) - N_TEST]  # 470 premiers points


def make_supervised(series, window):
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i : i + window])
        y.append(series[i + window])
    return np.array(X), np.array(y)


X_train, y_train = make_supervised(x_train_full, WINDOW)
print(f"Nombre d'exemples d'entraînement : {len(X_train)}")

# ---------------------------------------------------------------
# 2. Normalisation
# ---------------------------------------------------------------
scaler_X = MinMaxScaler()
X_train_s = scaler_X.fit_transform(X_train)

# ---------------------------------------------------------------
# 3. Apprentissage du réseau final
# ---------------------------------------------------------------
model = MLPRegressor(
    hidden_layer_sizes=HIDDEN_LAYERS,
    activation="tanh",
    solver="lbfgs",
    max_iter=5000,
    random_state=RANDOM_STATE,
)
model.fit(X_train_s, y_train)

# ---------------------------------------------------------------
# 4. Évaluation sur les données d'entraînement (erreur d'ajustement)
# ---------------------------------------------------------------
y_train_pred = model.predict(X_train_s)
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
mae_train = np.mean(np.abs(y_train - y_train_pred))

print(f"\n=== Performance sur les données d'entraînement ===")
print(f"MSE  = {mse_train:.8e}")
print(f"RMSE = {rmse_train:.8e}")
print(f"MAE  = {mae_train:.8e}")

# ---------------------------------------------------------------
# 5. Sauvegarde du modèle et du scaler pour les étapes suivantes
# ---------------------------------------------------------------
joblib.dump(model, "results/model_henon.joblib")
joblib.dump(scaler_X, "results/scaler_henon.joblib")

with open("results/training_report.txt", "w") as f:
    f.write(f"Fenêtre                : {WINDOW}\n")
    f.write(f"Architecture            : {HIDDEN_LAYERS}\n")
    f.write(f"Nb exemples entraînement: {len(X_train)}\n")
    f.write(f"MSE  (train)            : {mse_train:.8e}\n")
    f.write(f"RMSE (train)            : {rmse_train:.8e}\n")
    f.write(f"MAE  (train)            : {mae_train:.8e}\n")

print("\nModèle sauvegardé : results/model_henon.joblib")
print("Scaler sauvegardé  : results/scaler_henon.joblib")
print("Rapport sauvegardé : results/training_report.txt")