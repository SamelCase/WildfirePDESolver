import numpy as np
import matplotlib.pyplot as plt

def pendulumBVPSolver(alpha,beta,T,m=100,maxIter=1000,tol=1e-6):
    """ Solves the pendulum problem with a BVP using Newton's method. 
    Accepts the initial and final angles alpha and beta, the time T, the number of interior points m, the maximum number of iterations maxIter, and the tolerance tol for convergence. 
    Returns the solution theta at the interior points. """
    h = T/(m+1)
    thetaLst = np.zeros(m) # Initial guess for thetas at intermediary points
    def TriDiagA(m,h):
        A = np.zeros((m,m)) # Tridiagonal matrix for the second derivative approximation
        for i in range(m):
            A[i,i] = -2
            if i > 0:
                A[i,i-1] = 1
            if i < m-1:
                A[i,i+1] = 1
        return A/h**2
    
    A = TriDiagA(m,h)

    def G(theta):
        F = A @ theta + np.sin(theta) # System of equations to solve for the unknown thetas at intermediary points
        F[0] += alpha/h**2 # Incorporate the boundary condition at t=0
        F[-1] += beta/h**2 # Incorporate the boundary condition at t=T
        return F
    
    def Dg(theta):
        return A + np.diag(np.cos(theta)) # Jacobian matrix of G
    
    # Apply Newton's Method
    thetaSol, converged, numIter = Newton(G, thetaLst, Dg, maxIter, tol)


    dta = []
    t = np.linspace(h, T-h, m)

    dta.append([0, alpha, beta, T, alpha])
    for i in range(m): 
        dta.append([t[i], alpha, beta, T, thetaSol[i]])
    dta.append([T, alpha, beta, T, beta]) # Add the initial boundary condition to the data
    # Now thetaSol is the angle at every interior point
    dta = np.array(dta) # shape (m+2, 5)
    return dta, converged

def Newton(f, x0, Df, maxIter, tol):     
    """ 
    Newton's method for solving f(theta) = 0
    Adapted from ACME oneD_optimization.py 
    
    Parameters:
        f (function): a function from R^n to R^n (assume n=1 until Problem 5).
        x0 (float or ndarray): The initial guess for the zero of f.
        Df (function): The derivative of f, a function from R^n to R^(nxn).
        tol (float): Convergence tolerance. The function should returns when
            the difference between successive approximations is less than tol.
        maxiter (int): The maximum number of iterations to compute.

    Returns:
        (float or ndarray): The approximation for a zero of f.
        (bool): Whether or not Newton's method converged.
        (int): The number of iterations computed.
    """
    x = x0.copy()
    for iter in range(maxIter):
        Fx = f(x)
        J = Df(x) # Jacobian matrix of G
        delta = np.linalg.solve(J, -Fx) # Solve for the update step
        x += delta # Update the guess for theta
        if np.linalg.norm(delta, ord=np.inf) < tol:
            return (x, True, iter)
        # Alternative convergence condition: if the update step is small, we can also consider that as convergence
        # if np.linalg.norm(Fx, ord=np.inf) < tol:
        #     # print(f"Converged in {iter} iterations.")
        #     return x
    return (x, False, maxIter)
    