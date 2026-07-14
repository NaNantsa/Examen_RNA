"""
Étape 3a : Détermination de l'architecture optimale du réseau de neurones
pour la prédiction de la série x_n de Hénon.

Méthode : on transforme la série x_n en un problème supervisé
(fenêtre glissante de taille w -> valeur suivante), puis on compare
plusieurs architectures de perceptron multicouche (MLPRegressor) par
validation croisée temporelle (TimeSeriesSplit), pour différentes
tailles de fenêtre w.
"""
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import itertools
import warnings
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
N_TEST = 30  # dernières valeurs réservées (test final, non utilisées ici)

# ---------------------------------------------------------------
# 1. Chargement des données
# ---------------------------------------------------------------
data = np.loadtxt("henon_data.csv", delimiter=",", skiprows=1)
x = data[:, 0]  # on modélise uniquement la série x_n, comme demandé

# On réserve les N_TEST dernières valeurs pour les étapes 3c/3d
# (elles ne doivent PAS être vues pendant la recherche d'architecture)
x_search = x[: len(x) - N_TEST]


def make_supervised(series, window):
    """Transforme une série 1D en (X, y) avec fenêtre glissante."""
    X, y = [], []
    for i in range(len(series) - window):
        X.append(series[i : i + window])
        y.append(series[i + window])
    return np.array(X), np.array(y)


# ---------------------------------------------------------------
# 2. Grille de recherche
# ---------------------------------------------------------------
window_sizes = [2, 3, 4, 5, 6]
architectures = [
    (5,),
    (10,),
    (20,),
    (5, 5),
    (10, 5),
    (10, 10),
    (20, 10),
]

tscv = TimeSeriesSplit(n_splits=5)

results = []

for window in window_sizes:
    X, y = make_supervised(x_search, window)

    for arch in architectures:
        fold_mses = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Normalisation (fit uniquement sur le train du fold)
            scaler_X = MinMaxScaler()
            X_train_s = scaler_X.fit_transform(X_train)
            X_val_s = scaler_X.transform(X_val)

            model = MLPRegressor(
                hidden_layer_sizes=arch,
                activation="tanh",
                solver="lbfgs",
                max_iter=2000,
                random_state=RANDOM_STATE,
            )
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_val_s)
            fold_mses.append(mean_squared_error(y_val, y_pred))

        mean_mse = np.mean(fold_mses)
        std_mse = np.std(fold_mses)
        n_params = (
            sum(
                (window if k == 0 else arch[k - 1]) * n + n
                for k, n in enumerate(arch)
            )
            + arch[-1]
            + 1
        )
        results.append(
            {
                "window": window,
                "architecture": arch,
                "mean_val_mse": mean_mse,
                "std_val_mse": std_mse,
                "n_params": n_params,
            }
        )
        print(
            f"fenêtre={window}  archi={str(arch):12s}  "
            f"MSE_val moyenne={mean_mse:.8e}  (+/- {std_mse:.2e})"
        )

# ---------------------------------------------------------------
# 3. Sélection de la meilleure architecture
# ---------------------------------------------------------------
results_sorted = sorted(results, key=lambda r: r["mean_val_mse"])
best = results_sorted[0]

print("\n=== TOP 5 architectures (par MSE de validation croisée) ===")
for r in results_sorted[:5]:
    print(
        f"fenêtre={r['window']}  archi={str(r['architecture']):12s}  "
        f"MSE_val={r['mean_val_mse']:.8e}  n_params={r['n_params']}"
    )

print(f"\n>>> MEILLEURE ARCHITECTURE RETENUE :")
print(f"    Taille de fenêtre (entrées)  : {best['window']}")
print(f"    Couches cachées              : {best['architecture']}")
print(f"    MSE de validation croisée    : {best['mean_val_mse']:.8e}")

# Sauvegarde des résultats complets
with open("results/architecture_search.csv", "w") as f:
    f.write("window,architecture,mean_val_mse,std_val_mse,n_params\n")
    for r in results_sorted:
        arch_str = "-".join(map(str, r["architecture"]))
        f.write(
            f"{r['window']},{arch_str},{r['mean_val_mse']:.8e},"
            f"{r['std_val_mse']:.8e},{r['n_params']}\n"
        )

# Sauvegarde du choix retenu pour les scripts suivants
with open("results/best_architecture.txt", "w") as f:
    f.write(f"window={best['window']}\n")
    f.write(f"architecture={'-'.join(map(str, best['architecture']))}\n")
    f.write(f"mean_val_mse={best['mean_val_mse']:.8e}\n")

print("\nRésultats complets sauvegardés dans results/architecture_search.csv")
print("Meilleure architecture sauvegardée dans results/best_architecture.txt")