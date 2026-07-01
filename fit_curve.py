"""
FlamApp AI - R&D/AI Assignment
Fit the parametric curve:
    x(t) = t*cos(theta) - e^(M*|t|) * sin(0.3t) * sin(theta) + X
    y(t) = 42 + t*sin(theta) + e^(M*|t|) * sin(0.3t) * cos(theta)
to the given (x, y) point cloud in xy_data.csv, for t in [6, 60].

Author: Samhitha Gorantla
"""

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.optimize import differential_evolution, minimize


def curve_points(theta_deg, M, X, t_dense):
    """Sample the parametric curve at given t values for candidate params."""
    theta = np.radians(theta_deg)
    exp_term = np.exp(M * np.abs(t_dense)) * np.sin(0.3 * t_dense)
    x = t_dense * np.cos(theta) - exp_term * np.sin(theta) + X
    y = 42 + t_dense * np.sin(theta) + exp_term * np.cos(theta)
    return np.column_stack([x, y])


def mean_l1_loss(params, data, t_dense):
    """
    Loss = mean L1 distance between each data point and its nearest
    point on the candidate curve.

    Note: the CSV gives (x, y) pairs with no 't' label and not in
    t-order, so this is a point-cloud-to-curve fit, not an ordered
    curve_fit. Each data point is matched to its nearest neighbor on
    the densely sampled candidate curve via a KD-tree.
    """
    theta_deg, M, X = params
    curve = curve_points(theta_deg, M, X, t_dense)
    tree = cKDTree(curve)
    _, idx = tree.query(data, k=1)
    matched = curve[idx]
    return np.abs(matched - data).sum(axis=1).mean()


def fit(data):
    bounds = [(0.001, 49.999), (-0.05, 0.05), (0.001, 99.999)]  # theta, M, X

    # Stage 1: global search (coarse grid, avoids local minima)
    t_coarse = np.linspace(6, 60, 6000)
    result = differential_evolution(
        mean_l1_loss, bounds, args=(data, t_coarse),
        maxiter=200, popsize=25, tol=1e-10, seed=42, polish=True
    )

    # Stage 2: local polish on a much finer curve grid for precision
    t_fine = np.linspace(6, 60, 20000)
    refined = minimize(
        mean_l1_loss, x0=result.x, args=(data, t_fine),
        method='Nelder-Mead',
        options={'xatol': 1e-8, 'fatol': 1e-12, 'maxiter': 5000}
    )
    return refined.x, refined.fun


if __name__ == "__main__":
    df = pd.read_csv("xy_data.csv")
    data = df[["x", "y"]].values

    (theta_deg, M, X), loss = fit(data)
    print(f"theta = {theta_deg:.6f} deg")
    print(f"M     = {M:.6f}")
    print(f"X     = {X:.6f}")
    print(f"mean L1 residual = {loss:.6f}")
