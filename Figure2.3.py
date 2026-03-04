import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

"""
This is an exploratory file trying to empirically reproduce figure 2.3
in the textbook. Presently it uses SciPy's library as a baseline for solving
this problem. Our goal is to reproduce that algorithm ourselves, and then
work to improve on it by predicting the matrix A 
"""

def fig23():
    # Time interval
    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 1000)

    # Initial amplitudes
    amplitudes = [0, 0.3, 1, 2]

    # Linear system
    def linear(t, y):
        theta, omega = y
        return [omega, -theta]

    # Nonlinear system
    def nonlinear(t, y):
        theta, omega = y
        return [omega, -np.sin(theta)]

    # Create figure
    fig, axs = plt.subplots(2, 1, figsize=(8, 8))

    # --- (b) Linear ---
    for A in amplitudes:
        sol = solve_ivp(linear, t_span, [A, 0], t_eval=t_eval)
        axs[0].plot(sol.t, sol.y[0])

    axs[0].set_title("Linear: θ'' + θ = 0")
    axs[0].set_ylabel("θ(t)")
    axs[0].grid()

    # --- (c) Nonlinear ---
    for A in amplitudes:
        sol = solve_ivp(nonlinear, t_span, [A, 0], t_eval=t_eval)
        axs[1].plot(sol.t, sol.y[0])

    axs[1].set_title("Nonlinear: θ'' + sin(θ) = 0")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("θ(t)")
    axs[1].grid()

    plt.tight_layout()
    plt.savefig("figure2.3.png")

if __name__ == "__main__":
    fig23()