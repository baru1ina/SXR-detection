import numpy as np

def compute_derivative(signals, dt):
    deriv = np.gradient(signals, dt, axis=0)
    return deriv