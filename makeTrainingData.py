import h5py
import numpy as np
import pendulumProblem as pp
from tqdm import tqdm

def createDataset():
    with h5py.File("data/pendulum_dataset.h5", "w") as f:
        f.create_dataset("inputs", shape=(0,4), maxshape=(None,4), chunks=True) # shape (N,4) where N is the number of data points, and each data point consists of (t, alpha, beta, T)
        f.create_dataset("targets", shape=(0,1), maxshape=(None,1), chunks=True) # shape (N,1) where N is the number of data points, and each data point consists of the corresponding theta(t) value

def addToDataset(alpha, beta, T, f):

    """ Solves the pendulum problem for the given parameters and adds the resulting data to the dataset. """
    dta, converged = pp.pendulumBVPSolver(alpha, beta, T)
    if not converged:
        return # If the solver did not converge, we skip adding this data point to the dataset. We could also choose to add it with a special label indicating non-convergence, but for now we'll just skip it.
    inputs = dta[:,:4]   # shape (N,4)
    targets = dta[:,4].reshape(-1,1)  # shape (N,)
    X = f["inputs"]
    Y = f["targets"]

    n = X.shape[0]
    new_n = n + inputs.shape[0]

    X.resize((new_n,4))
    Y.resize((new_n,1))

    X[n:new_n] = inputs
    Y[n:new_n] = targets

if __name__ == "__main__":
    createDataset()
    with h5py.File("data/pendulum_dataset.h5", "a") as f:
        for _ in tqdm(range(100000)):
            alpha = np.random.uniform(-np.pi/2, np.pi/2)
            beta = np.random.uniform(-np.pi/2, np.pi/2)
            T = np.random.uniform(0.5, 5)
            addToDataset(alpha, beta, T, f)