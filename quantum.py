import numpy as np
import matplotlib.pyplot as plt


def simulate_wavefunction(x, E, V):

    if E > V:

        k = np.sqrt(2 * (E - V))

        return np.cos(k * x)

    elif E < V:

        kappa = np.sqrt(2 * (V - E))

        return np.exp(-kappa * x)

    else:

        return np.ones_like(x)


x = np.linspace(
    0,
    5,
    500
)


psi_a = simulate_wavefunction(
    x,
    E=12,
    V=2
)

psi_b = simulate_wavefunction(
    x,
    E=2,
    V=7
)

psi_c = simulate_wavefunction(
    x,
    E=5,
    V=5
)


plt.figure(figsize=(10, 6))

plt.plot(
    x,
    psi_a,
    label="E=12, V=2"
)

plt.plot(
    x,
    psi_b,
    label="E=2, V=7"
)

plt.plot(
    x,
    psi_c,
    label="E=5, V=5"
)

plt.xlabel("x")

plt.ylabel("ψ(x)")

plt.title(
    "Wavefunctions for Different Energy Regimes"
)

plt.legend()

plt.grid(True)

plt.show()
