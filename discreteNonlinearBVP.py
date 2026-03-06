import numpy as np
import matplotlib.pyplot as plt
# from scipy.integrate import solve_ivp
# from time import perf_counter

""" This file is for exploring the pendulum problem with a BVP.
Consider the situtaion in whcih we wish to set the pendulum swinging from 
some initial givien location theta(0) = alpha with some unknown angular
velocity theta'(0) in such a way that the pendulum will be at the desired
location theta(T) = beta at some specified later time T. 
Then we have a 2-point BVP"""

def prob2_16(T,m,alpha,beta,maxIter=1000,tol=1e-6):
    h = T/(m+1)
    thetaLst = np.zeros(m) # Initial guess for thetas at intermediary points
    A = np.zeros((m,m)) # Tridiagonal matrix for the second derivative approximation
    for i in range(m):
        A[i,i] = -2
        if i > 0:
            A[i,i-1] = 1
        if i < m-1:
            A[i,i+1] = 1
    A = A/h**2
    def G(theta):
        F = A @ theta + np.sin(theta) # System of equations to solve for the unknown thetas at intermediary points
        F[0] += alpha/h**2 # Incorporate the boundary condition at t=0
        F[-1] += beta/h**2 # Incorporate the boundary condition at t=T
        return F
    
    # Apply Newton's Method
    def Newton(thetaLst, maxIter, tol):
        """ Newton's method for solving G(theta) = 0 
        Accepts an initial guess thetaLst, a maximum number of iterations maxIter, and a tolerance tol for convergence.
        Returns every iterate of theta, which is a list of the unknown thetas at intermediary points. """
        theta = thetaLst.copy()
        thetaIterates = []
        for _ in range(maxIter):
            F = G(theta)
            if np.linalg.norm(F, ord=np.inf) < tol:
                # print(f"Converged in {iter} iterations.")
                return thetaIterates
            thetaIterates.append(theta.copy())
            J = A + np.diag(np.cos(theta)) # Jacobian matrix of G
            delta = np.linalg.solve(J, -F) # Solve for the update step
            theta += delta # Update the guess for theta
        raise RuntimeError("Did not converge within the maximum number of iterations.")
    
    # Plot the convergence of Newton iterates toward a solution of the pendulum problem
    # The iterates theta^[k] are denoted by the number k in the plots
    # (a) starting from theta^[0]_i = .7*cos(t_i)+.5*sin(t_i)
    # (b) starting from theta^[0]_i = .7

    _, ax = plt.subplots(1, 2, figsize=(12, 6))
    # Initial guess (a) 
    thetaLst = 0.7*np.cos(np.linspace(0,T,m)) + 0.5*np.sin(np.linspace(0,T,m))
    thetaIterates = Newton(thetaLst, maxIter, tol)
    for k, theta in enumerate(thetaIterates):
        ax[0].plot(np.linspace(0,T,m), theta, label=f"{k}")
    # Initial guess (b)
    thetaLst = np.full(m, 0.7)
    thetaIterates = Newton(thetaLst, maxIter, tol)
    for k, theta in enumerate(thetaIterates):
        ax[1].plot(np.linspace(0,T,m), theta, label=f"{k}")
    # Label everything appropriately
    ax[0].set_title(r"Convergence of Newton's Method from $\theta^{[0]}_i = 0.7\cos(t_i) + 0.5\sin(t_i)$")
    ax[1].set_title(r"Convergence of Newton's Method from $\theta^{[0]}_i = 0.7$")
    for axe in ax:
        axe.set_xlabel("Time")
        axe.set_ylabel("Theta")
        axe.legend()
        axe.set_xlim(0,T)
        axe.set_ylim(-1.2,1)
    plt.tight_layout()
    plt.savefig("images/figure2.4.png")
    return theta
if __name__ == "__main__":
    T = 2*np.pi
    alpha = beta = .7
    m = 100
    theta = prob2_16(T,m,alpha,beta)