import numpy as np

a = 1.4
b = 0.3
N = 500  

x = np.zeros(N)
y = np.zeros(N)

x[0] = 0.0
y[0] = 0.0


for n in range(N - 1):
    x[n + 1] = y[n] + 1 - a * x[n] ** 2
    y[n + 1] = b * x[n]

# Sauvegarde des données pour les étapes suivantes
np.savetxt("henon_data.csv", np.column_stack([x, y]), delimiter=",",
           header="x,y", comments="")

print("8 premières valeurs générées :")
for n in range(8):
    print(f"n={n:3d}  x={x[n]: .8f}   y={y[n]: .8f}")

print("\n...\n")
print("Dernières valeurs (n=495 à 499) :")
for n in range(495, 500):
    print(f"n={n:3d}  x={x[n]: .8f}   y={y[n]: .8f}")

print(f"\nStatistiques x : min={x.min():.8f}, max={x.max():.8f}")
print(f"Statistiques y : min={y.min():.8f}, max={y.max():.8f}")