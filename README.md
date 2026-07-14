# Prédiction et modélisation de séries temporelles par réseaux de neurones artificiels multicouches

**Institut Supérieur Polytechnique de Madagascar**
Mini-projet d'application — Juin 2026
Filières : ESIIA4 - IGGLIA4 - IMTICIA4 - ISAIA4

## Objectif

Étudier la série chaotique de Hénon, générée par la relation de récurrence :

```
x(n+1) = y(n) + 1 - a * x(n)^2
y(n+1) = b * x(n)
```

avec `a = 1.4`, `b = 0.3`, et prédire son évolution à l'aide d'un réseau de neurones
artificiel multicouche (perceptron multicouche).

## Structure du projet

```
.
├── README.md
├── requirements.txt        # dépendances Python
├── .gitignore
├── step1_generate.py       # génération des 500 valeurs (xn, yn)
├── step2_plot.py           # tracé de yn en fonction de xn (attracteur)
├── step3_architecture.py   # recherche de l'architecture optimale du réseau
├── step3_training.py       # apprentissage du réseau
├── step3_predict_1step.py  # prédictions à 1 pas
├── step3_predict_multistep.py  # prédictions à 3, 10 et 20 pas en avant
├── henon_data.csv          # données générées (500 points)
├── figures/                # graphiques générés
└── results/                # résultats numériques (erreurs, prédictions)
```

## Installation

python3 -m venv venv
source venv/bin/activate.fish   # sous fish
# ou : source venv/bin/activate   # sous bash/zsh
pip install -r requirements.txt


## Utilisation

Exécuter les scripts dans l'ordre :

```fish
python step1_generate.py            # étape 1 : génération des données
python step2_plot.py                # étape 2 : tracé yn = f(xn)
python step3_architecture.py        # étape 3a : choix de l'architecture
python step3_training.py            # étape 3b : apprentissage
python step3_predict_1step.py       # étape 3c : prédictions à 1 pas
python step3_predict_multistep.py   # étape 3d : prédictions à 3/10/20 pas
```

## Méthodologie

1. **Génération des données** : 500 valeurs de (xn, yn) générées à partir de
   x0 = 0, y0 = 0.
2. **Visualisation** : tracé de l'attracteur de Hénon (yn en fonction de xn).
3. **Prédiction par réseau de neurones** sur la série (xn) :
   - Détermination de l'architecture optimale (nombre de couches, de
     neurones, fenêtre temporelle d'entrée).
   - Apprentissage du réseau.
   - Prédictions à 1 pas (10 valeurs prédites vs 10 valeurs attendues).
   - Prédictions à 3, 10 et 20 pas en avant.
4. **Analyse** : comparaison de la dégradation des prédictions selon
   l'horizon de prédiction (1 pas vs pas multiples).

## Résultats

Les résultats numériques (8 chiffres significatifs) et les erreurs de
prédiction associées sont disponibles dans le dossier `results/`.
Les graphiques correspondants sont dans `figures/`.


