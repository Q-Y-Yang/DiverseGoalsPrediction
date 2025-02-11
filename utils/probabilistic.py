# Imports
import numpy as np
from collections.abc import Iterable

# Functions
def fill_gaussian(size, mu, sigma):
    # determine final number of dimensions
    dim = len(size)

    # expand dims of mu, if one value
    if not isinstance(mu, Iterable):
        mu = np.full(dim, mu)
        
    # expand dims of sigma if one value is provided
    if not isinstance(sigma, Iterable):
        sigma = np.full(dim, sigma)
    
    # set of shapes per axis
    shapes = np.diag(size) + np.ones((dim, dim), dtype=int) - np.identity(dim, dtype=int)

    return np.exp(-4*np.log(2) * sum(
        (np.arange(shape[i], dtype=float).reshape(shape) - mu[i])**2 / sigma[i]**2
        for i, shape in enumerate(shapes)
    ))

def fill_multi_gaussian(size, mus, sigma) -> np.ndarray:
    # If no mus return zero matrix
    if len(mus) == 0:
        return np.zeros(size, dtype=float)

    # determine final number of dimensions
    dim = len(size)

    # expand dims of mu
    # expand dims of sigma if one value is provided
    if not isinstance(sigma, Iterable):
        sigma = np.full(dim, sigma)
    
    # set of shapes per axis
    shapes = np.diag(size) + np.ones((dim, dim), dtype=int) - np.identity(dim, dtype=int)

    idcs = [
        np.arange(shape[i], dtype=float).reshape(shape)
        for i, shape in enumerate(shapes)
    ]

    alpha = -4*np.log(2)
    return np.sum(
        [
            np.exp(alpha * sum(
                (x - mu[i])**2 / sigma[i]**2
                for i, x in enumerate(idcs)
            ))
            for mu in mus
        ], axis=0
    ).clip(0.0, 1.0)