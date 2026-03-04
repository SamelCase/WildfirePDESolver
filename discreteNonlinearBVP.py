import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from time import perf_counter

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
    theta = thetaLst.copy()
    for iter in range(maxIter):
        F = G(theta)
        if np.linalg.norm(F, ord=np.inf) < tol:
            print(f"Converged in {iter} iterations.")
            return theta
        J = A + np.diag(np.cos(theta)) # Jacobian matrix of G
        delta = np.linalg.solve(J, -F) # Solve for the update step
        theta += delta # Update the guess for theta