"""
Étape 3c : Prédictions à 1 pas.

On prédit 10 valeurs de x_n, chacune à partir des 2 vraies valeurs
précédentes (fenêtre glissante sur les données réelles, jamais sur les
prédictions précédentes). On compare aux 10 valeurs réellement attendues.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

WINDOW = 2
N_TEST = 30
N_PRED = 10  # nombre de valeurs prédites à 1 pas

# ---------------------------------------------------------------
# 1. Chargement des données, du modèle et du scaler
# ---------------------------------------------------------------
data = np.loadtxt("henon_data.csv", delimiter=",", skiprows=1)
x = data[:, 0]

model = joblib.load("results/model_henon.joblib")
scaler_X = joblib.load("results/scaler_henon.joblib")

# Les N_TEST dernières valeurs n'ont jamais été vues à l'entraînement.
# On prédit les N_PRED premières d'entre elles, à 1 pas.
test_start = len(x) - N_TEST  # index de la 1ère valeur de test

y_true = []
y_pred = []

for k in range(N_PRED):
    idx_target = test_start + k          # indice de la valeur à prédire
    window_vals = x[idx_target - WINDOW : idx_target]  # 2 vraies valeurs précédentes
    window_scaled = scaler_X.transform(window_vals.reshape(1, -1))
    pred = model.predict(window_scaled)[0]

    y_true.append(x[idx_target])
    y_pred.append(pred)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
errors = y_true - y_pred
rel_errors = np.abs(errors) / np.abs(y_true) * 100

# ---------------------------------------------------------------
# 2. Affichage des résultats (8 chiffres significatifs)
# ---------------------------------------------------------------
print(f"{'n':>4s}  {'x_attendu':>14s}  {'x_predit':>14s}  {'erreur abs':>14s}  {'erreur rel(%)':>13s}")
for k in range(N_PRED):
    print(
        f"{test_start+k:4d}  {y_true[k]: .8g}  {y_pred[k]: .8g}  "
        f"{errors[k]: .8g}  {rel_errors[k]:8.4f}"
    )

mse = np.mean(errors ** 2)
rmse = np.sqrt(mse)
mae = np.mean(np.abs(errors))
max_err = np.max(np.abs(errors))

print(f"\n=== Erreurs globales (prédiction à 1 pas, {N_PRED} valeurs) ===")
print(f"MSE          = {mse:.8e}")
print(f"RMSE         = {rmse:.8e}")
print(f"MAE          = {mae:.8e}")
print(f"Erreur max   = {max_err:.8e}")

# ---------------------------------------------------------------
# 3. Sauvegarde des résultats numériques
# ---------------------------------------------------------------
with open("results/prediction_1step.csv", "w") as f:
    f.write("n,x_attendu,x_predit,erreur_abs,erreur_rel_pct\n")
    for k in range(N_PRED):
        f.write(
            f"{test_start+k},{y_true[k]:.8g},{y_pred[k]:.8g},"
            f"{errors[k]:.8g},{rel_errors[k]:.4f}\n"
        )

with open("results/prediction_1step_summary.txt", "w") as f:
    f.write(f"MSE  = {mse:.8e}\n")
    f.write(f"RMSE = {rmse:.8e}\n")
    f.write(f"MAE  = {mae:.8e}\n")
    f.write(f"Erreur max = {max_err:.8e}\n")

# ---------------------------------------------------------------
# 4. Graphique : valeurs prédites vs valeurs attendues
# ---------------------------------------------------------------
n_axis = np.arange(test_start, test_start + N_PRED)

plt.figure(figsize=(9, 5))
plt.plot(n_axis, y_true, "o-", color="darkblue", label="Valeurs attendues (réelles)")
plt.plot(n_axis, y_pred, "s--", color="crimson", label="Valeurs prédites (1 pas)")
plt.title("Prédiction à 1 pas de $x_n$ : valeurs prédites vs valeurs attendues")
plt.xlabel("n")
plt.ylabel("$x_n$")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figures/prediction_1step.png", dpi=150)
print("\nFigure sauvegardée : figures/prediction_1step.png")