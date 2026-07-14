"""
Étape 3d : Prédictions à 3, 10 et 20 pas en avant.

Contrairement à la prédiction à 1 pas (étape 3c), ici la prédiction est
RÉCURSIVE : pour prédire loin dans le futur, le réseau réutilise ses
propres prédictions précédentes comme entrées (et non les vraies
valeurs). L'erreur se propage donc et s'amplifie avec l'horizon de
prédiction, illustrant la sensibilité aux conditions initiales propre
aux systèmes chaotiques.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

WINDOW = 2
N_TEST = 30
HORIZONS = [3, 10, 20]

# ---------------------------------------------------------------
# 1. Chargement des données, du modèle et du scaler
# ---------------------------------------------------------------
data = np.loadtxt("henon_data.csv", delimiter=",", skiprows=1)
x = data[:, 0]

model = joblib.load("results/model_henon.joblib")
scaler_X = joblib.load("results/scaler_henon.joblib")

test_start = len(x) - N_TEST  # 1ère valeur jamais vue à l'entraînement


def predict_recursive(x_full, start_idx, horizon, window, model, scaler):
    """
    Prédit `horizon` valeurs futures à partir de l'indice start_idx,
    en réinjectant récursivement les prédictions précédentes.

    La fenêtre initiale (les `window` valeurs juste avant start_idx)
    est constituée de VRAIES valeurs (dernières valeurs connues).
    """
    window_vals = list(x_full[start_idx - window : start_idx])  # vraies valeurs
    preds = []
    for _ in range(horizon):
        inp = np.array(window_vals[-window:]).reshape(1, -1)
        inp_scaled = scaler.transform(inp)
        p = model.predict(inp_scaled)[0]
        preds.append(p)
        window_vals.append(p)  # on réinjecte la prédiction, pas la vraie valeur
    return np.array(preds)


results_summary = []

fig, axes = plt.subplots(len(HORIZONS), 1, figsize=(9, 4 * len(HORIZONS)))

for ax, horizon in zip(axes, HORIZONS):
    y_pred = predict_recursive(x, test_start, horizon, WINDOW, model, scaler_X)
    y_true = x[test_start : test_start + horizon]

    errors = y_true - y_pred
    rel_errors = np.abs(errors) / np.abs(y_true) * 100
    mse = np.mean(errors ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(errors))
    max_err = np.max(np.abs(errors))

    print(f"\n=== Prédiction à {horizon} pas en avant (récursive) ===")
    print(f"{'n':>4s}  {'x_attendu':>14s}  {'x_predit':>14s}  {'erreur abs':>14s}  {'erreur rel(%)':>13s}")
    for k in range(horizon):
        print(
            f"{test_start+k:4d}  {y_true[k]: .8g}  {y_pred[k]: .8g}  "
            f"{errors[k]: .8g}  {rel_errors[k]:8.4f}"
        )
    print(f"MSE = {mse:.8e}   RMSE = {rmse:.8e}   MAE = {mae:.8e}   err_max = {max_err:.8e}")

    results_summary.append(
        {"horizon": horizon, "mse": mse, "rmse": rmse, "mae": mae, "max_err": max_err}
    )

    # Sauvegarde CSV détaillé
    with open(f"results/prediction_{horizon}step.csv", "w") as f:
        f.write("n,x_attendu,x_predit,erreur_abs,erreur_rel_pct\n")
        for k in range(horizon):
            f.write(
                f"{test_start+k},{y_true[k]:.8g},{y_pred[k]:.8g},"
                f"{errors[k]:.8g},{rel_errors[k]:.4f}\n"
            )

    # Graphique
    n_axis = np.arange(test_start, test_start + horizon)
    ax.plot(n_axis, y_true, "o-", color="darkblue", label="Valeurs attendues (réelles)")
    ax.plot(n_axis, y_pred, "s--", color="crimson", label="Valeurs prédites (récursif)")
    ax.set_title(f"Prédiction récursive à {horizon} pas en avant")
    ax.set_xlabel("n")
    ax.set_ylabel("$x_n$")
    ax.legend()
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("figures/prediction_multistep.png", dpi=150)
print("\nFigure sauvegardée : figures/prediction_multistep.png")

# ---------------------------------------------------------------
# Récapitulatif comparatif des 3 horizons
# ---------------------------------------------------------------
print("\n=== RÉCAPITULATIF : dégradation de l'erreur avec l'horizon ===")
print(f"{'Horizon':>8s}  {'MSE':>14s}  {'RMSE':>14s}  {'MAE':>14s}  {'err_max':>14s}")
for r in results_summary:
    print(
        f"{r['horizon']:8d}  {r['mse']:.8e}  {r['rmse']:.8e}  "
        f"{r['mae']:.8e}  {r['max_err']:.8e}"
    )

with open("results/prediction_multistep_summary.csv", "w") as f:
    f.write("horizon,mse,rmse,mae,max_err\n")
    for r in results_summary:
        f.write(
            f"{r['horizon']},{r['mse']:.8e},{r['rmse']:.8e},"
            f"{r['mae']:.8e},{r['max_err']:.8e}\n"
        )

# Graphique récapitulatif : MSE en fonction de l'horizon
plt.figure(figsize=(7, 5))
horizons_arr = [r["horizon"] for r in results_summary]
mse_arr = [r["mse"] for r in results_summary]
plt.plot(horizons_arr, mse_arr, "o-", color="darkgreen")
plt.yscale("log")
plt.xlabel("Horizon de prédiction (pas en avant)")
plt.ylabel("MSE (échelle log)")
plt.title("Dégradation de l'erreur de prédiction avec l'horizon")
plt.grid(alpha=0.3, which="both")
plt.tight_layout()
plt.savefig("figures/mse_vs_horizon.png", dpi=150)
print("Figure sauvegardée : figures/mse_vs_horizon.png")