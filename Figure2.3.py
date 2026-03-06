import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from time import perf_counter

"""
This is an exploratory file trying to empirically reproduce figure 2.3
in the textbook. Presently it uses SciPy's library as a baseline for solving
this problem. Our goal is to reproduce that algorithm ourselves, and then
work to improve on it by predicting the matrix A 
"""

def fig23():
    # Time interval
    t_span = (0, 10)
    t_eval = np.linspace(0, 10, 10000000)

    # Initial amplitudes eyeballed from the textbook figure.
    amplitudes = [0, 0.3, 1, 2]
    tLin, tNLin = [], []
    # SciPy's solve_ivp expects the system to be in first-order form, so we need to rewrite our second-order ODE as 
    # a system of two first-order ODEs. We define theta as the angle and omega as the angular velocity (theta').

    # Linear system
    def linear(t, y):
        theta, omega = y
        return [omega, -theta]

    # Nonlinear system
    def nonlinear(t, y):
        theta, omega = y
        return [omega, -np.sin(theta)]

    # Create figure
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    # Linear approximation, where for small theta we can say sin(theta) is approximately equal to theta
    for A in amplitudes:
        start = perf_counter()
        sol = solve_ivp(linear, t_span, [A, 0], t_eval=t_eval)
        end = perf_counter()
        tLin.append(end - start)
        axs[0].plot(sol.t, sol.y[0])

    axs[0].set_title("Linear: θ'' + θ = 0")
    axs[0].set_ylabel("θ(t)")
    axs[0].set_xlim(0,10)
    axs[0].set_ylim(-2,2)
    axs[0].grid()

    # True nonlinear solutions, more expensive to compute
    for A in amplitudes:
        start = perf_counter()
        sol = solve_ivp(nonlinear, t_span, [A, 0], t_eval=t_eval)
        end = perf_counter()
        tNLin.append(end - start)
        axs[1].plot(sol.t, sol.y[0])

    axs[1].set_title("Nonlinear: θ'' + sin(θ) = 0")
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("θ(t)")
    axs[1].set_xlim(0,10)
    axs[1].set_ylim(-2,2)
    axs[1].grid()

    plt.tight_layout()
    plt.savefig("images/figure2.3.png")
    print(f"Mean Linear time: {np.mean(tLin):.4f} seconds")
    print(f"Mean Nonlinear time: {np.mean(tNLin):.4f} seconds")

    # We saw that in such a simple case, the two algorithms are actually pretty comparable in terms of runtime, 
    # which is interesting. We will see if we can do better by predicting the matrix A in the future.


if __name__ == "__main__":
    fig23()