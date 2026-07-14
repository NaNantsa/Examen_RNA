import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

data = np.loadtxt("henon_data.csv", delimiter=",", skiprows=1)
x, y = data[:, 0], data[:, 1]

plt.figure(figsize=(7, 6))
plt.scatter(x, y, s=4, c="darkblue")
plt.title("Attracteur de Hénon : $y_n$ en fonction de $x_n$\n(a=1.4, b=0.3, 500 points)")
plt.xlabel("$x_n$")
plt.ylabel("$y_n$")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("henon_attractor.png", dpi=150)
print("Figure sauvegardée : henon_attractor.png")