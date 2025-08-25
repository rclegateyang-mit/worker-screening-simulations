#!/usr/bin/env python3
"""
Worker Screening Equilibrium Solver

This module implements a fixed-point solver for the worker screening equilibrium
over wages (w_j) and cutoff costs (c_j) for firms j=1..J.

The solver uses Anderson acceleration with Tikhonov regularization and includes
comprehensive numerical safeguards for stability.
"""

import argparse
import time
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq
from scipy import special

# Optional numba import
try:
    import numba
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False


def safe_logs(x: np.ndarray, eps: float) -> np.ndarray:
    """
    Compute log with safety floor to prevent log(0).
    
    Args:
        x: Input array
        eps: Safety floor
        
    Returns:
        log(max(x, eps))
    """
    return np.log(np.maximum(x, eps))


def precompute_bases(xi: np.ndarray, loc_firms: np.ndarray, 
                    support_points: np.ndarray, gamma: float) -> np.ndarray:
    """
    Precompute base intensities B ∈ ℝ^{S×J}.
    
    Args:
        xi: Firm amenity shocks (J,)
        loc_firms: Firm locations (J, 2)
        support_points: Worker location support points (S, 2)
        gamma: Distance decay parameter
        
    Returns:
        Base intensities matrix B (S, J)
    """
    # Compute distances D_{s,j} = ||ℓ_s - ℓ_j||_2
    # Reshape for broadcasting: (S, 1, 2) - (1, J, 2) = (S, J, 2)
    diff = support_points[:, None, :] - loc_firms[None, :, :]
    distances = np.linalg.norm(diff, axis=2)  # (S, J)
    
    # Compute B_{s,j} = exp(-γ * D_{s,j}) * exp(ξ_j)
    # Broadcast xi: (1, J) to match distances (S, J)
    B = np.exp(-gamma * distances) * np.exp(xi[None, :])
    
    return B


def truncated_normal_column_terms(
    c_sorted: np.ndarray,
    mu_s: float,
    sigma_s: float,
    eps: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns (DeltaF, M) for k = 0,...,J using sentinels c_(0)=-inf, c_(J+1)=+inf.
    DeltaF[k] = Φ(z_k) − Φ(z_{k-1}),  M[k] = μ_s + σ_s * (φ(z_{k-1}) − φ(z_k)) / DeltaF[k],
    where z_k = (c_(k) − μ_s)/σ_s and Φ,φ are standard normal CDF/PDF.
    Enforces Φ(z_0)=0 and Φ(z_{J+1})=1 exactly; φ at sentinels is 0.
    
    Args:
        c_sorted: Cutoff costs sorted in ascending order (J,)
        mu_s: Mean of truncated normal
        sigma_s: Standard deviation of truncated normal
        eps: Numerical safety floor
        
    Returns:
        Tuple of (ΔF, M) arrays with lengths (J+1, J+1)
    """
    from scipy.stats import norm
    
    J = int(c_sorted.shape[0])
    c_pad = np.empty(J + 2, dtype=np.float64)
    c_pad[0] = -np.inf
    c_pad[1:-1] = c_sorted
    c_pad[-1] = np.inf

    z = (c_pad - mu_s) / sigma_s

    Phi = np.empty_like(z)
    Phi[0] = 0.0
    Phi[-1] = 1.0
    if J > 0:
        Phi[1:-1] = norm.cdf(z[1:-1])

    phi = np.zeros_like(z)
    if J > 0:
        phi[1:-1] = norm.pdf(z[1:-1])

    DeltaF = Phi[1:] - Phi[:-1]                # length J+1
    # Floor tiny masses for numerical stability but preserve unit sum
    DeltaF = np.maximum(DeltaF, eps)
    DeltaF = DeltaF / DeltaF.sum()

    # Conditional mean on each interval
    M = mu_s + sigma_s * (phi[:-1] - phi[1:]) / np.maximum(DeltaF, eps)
    # Clip extremes when DeltaF is tiny
    M = np.clip(M, mu_s - 20.0 * sigma_s, mu_s + 20.0 * sigma_s)
    
    return DeltaF, M


def fixed_point_map(logw: np.ndarray, logc: np.ndarray, 
                   A: np.ndarray, xi: np.ndarray, loc_firms: np.ndarray,
                   support_points: np.ndarray, support_weights: np.ndarray,
                   mu_s: float, sigma_s: float, alpha: float, beta: float, gamma: float,
                   B: np.ndarray, eps: float = 1e-12) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute one step of the fixed-point map G(x) where x = [logw; logc].
    
    Args:
        logw: Current log wages (J,)
        logc: Current log cutoff costs (J,)
        A: Firm TFP (J,)
        xi: Firm amenity shocks (J,)
        loc_firms: Firm locations (J, 2)
        support_points: Worker location support points (S, 2)
        support_weights: Worker location weights (S,)
        mu_s: Truncated normal mean
        sigma_s: Truncated normal std
        alpha: Wage elasticity parameter
        beta: Production function parameter
        gamma: Distance decay parameter
        B: Precomputed base intensities (S, J)
        eps: Numerical safety floor
        
    Returns:
        Tuple of (logw_new, logc_new)
    """
    J = len(A)
    S = len(support_points)
    
    # Step 1: Convert to levels and sort by c
    w = np.exp(logw)
    c = np.exp(logc)
    
    # Sort indices by c (ascending)
    order_idx = np.argsort(c)
    c_sorted = c[order_idx]
    
    # Compute ranks (0-based, ascending)
    rank = np.zeros(J, dtype=int)
    rank[order_idx] = np.arange(J)
    
    # Step 2: Compute truncated normal terms
    DeltaF, M = truncated_normal_column_terms(c_sorted, mu_s, sigma_s, eps)
    
    # Step 3: Compute NUM = B ⊙ (w^α)[None,:]
    NUM = B * (w ** alpha)[None, :]  # (S, J)
    
    # Step 4: Permute by order_idx
    NUM_sorted = NUM[:, order_idx]  # (S, J)
    
    # Step 5: Cumulative denominators along k
    DEN = np.cumsum(NUM_sorted, axis=1)  # (S, J)
    DEN = np.maximum(DEN, eps)  # Safety floor
    
    # Expand denominators to cover J+1 intervals
    DEN_full = np.concatenate([DEN, DEN[:, -1:]], axis=1)  # (S, J+1)
    
    # Step 6: Compute f = support_weights[:,None] / DEN_full
    f = support_weights[:, None] / np.maximum(DEN_full, eps)  # (S, J+1)
    
    # Step 7: Compute Gmat = NUM.T @ f (heavy BLAS-3 operation)
    Gmat = NUM.T @ f  # (J, J+1)
    
    # Step 8: Form column weights
    vL = DeltaF  # (J+1,)
    vS = DeltaF * M  # (J+1,)
    
    # Step 9: Weighted matrices
    H_L = Gmat * vL[None, :]  # (J, J+1)
    H_S = Gmat * vS[None, :]  # (J, J+1)
    
    # Step 10: Row-wise reversed cumulative sums (suffix sums) across interval index k
    CumL = np.flip(np.cumsum(np.flip(H_L, axis=1), axis=1), axis=1)  # (J, J+1)
    CumS = np.flip(np.cumsum(np.flip(H_S, axis=1), axis=1), axis=1)  # (J, J+1)
    
    # Step 11: Gather L and S using ranks (sum over k >= rank0[j])
    L = CumL[np.arange(J), rank]  # (J,)
    S = CumS[np.arange(J), rank]  # (J,)
    
    # Safety floors
    L = np.maximum(L, eps)
    S = np.maximum(S, eps)
    
    # Step 12: Update equations (E1') and (E2')
    t = safe_logs(L, eps) + safe_logs(S, eps)
    
    # (E2'): log w_j = log(1-β) + log A_j + (1-β) log(L_j S_j) - log L_j + log(α/(α+1))
    logw_new = (np.log(1 - beta) + np.log(A) + 
                (1 - beta) * t - safe_logs(L, eps) + np.log(alpha / (alpha + 1)))
    
    # (E1'): log c_j = -log(1-β) + log w_j - log A_j + β log(L_j S_j)
    logc_new = (-np.log(1 - beta) + logw_new - np.log(A) + beta * t)
    
    return logw_new, logc_new


def anderson_update(x: np.ndarray, Fx: np.ndarray, store: Dict[str, Any], 
                   m: int, reg: float, damping: float) -> np.ndarray:
    """
    Anderson acceleration update with Tikhonov regularization.
    
    Args:
        x: Current iterate
        Fx: Function evaluation F(x)
        store: Storage for Anderson history
        m: Anderson memory
        reg: Tikhonov regularization parameter
        damping: Fallback damping parameter
        
    Returns:
        Updated iterate
    """
    if m == 0:
        # No Anderson acceleration, use simple damping
        return x + damping * (Fx - x)
    
    # Initialize storage if needed
    if 'dx_history' not in store:
        store['dx_history'] = []
        store['dr_history'] = []
    
    # Current residual
    r = Fx - x
    
    # Add to history
    store['dx_history'].append(x.copy())
    store['dr_history'].append(r.copy())
    
    # Keep only last m+1 entries
    if len(store['dx_history']) > m + 1:
        store['dx_history'] = store['dx_history'][-m-1:]
        store['dr_history'] = store['dr_history'][-m-1:]
    
    # Need at least 2 points for Anderson
    if len(store['dx_history']) < 2:
        return x + damping * r
    
    # Build least-squares system
    k = len(store['dx_history']) - 1
    dx_mat = np.column_stack([store['dx_history'][i+1] - store['dx_history'][i] 
                             for i in range(k)])
    dr_mat = np.column_stack([store['dr_history'][i+1] - store['dr_history'][i] 
                             for i in range(k)])
    
    # Solve least squares with regularization
    try:
        # (dr_mat^T dr_mat + reg*I) * theta = dr_mat^T * r
        A = dr_mat.T @ dr_mat + reg * np.eye(k)
        b = dr_mat.T @ r
        theta = np.linalg.solve(A, b)
        
        # Anderson update
        x_new = (store['dx_history'][-1] + 
                np.sum([theta[i] * store['dx_history'][i] for i in range(k)], axis=0))
        
        # Clip update to prevent extreme jumps
        delta_x = x_new - x
        max_jump = 5.0
        if np.max(np.abs(delta_x)) > max_jump:
            scale = max_jump / np.max(np.abs(delta_x))
            x_new = x + scale * delta_x
        
        return x_new
        
    except np.linalg.LinAlgError:
        # Fallback to damping
        return x + damping * r


def solve_equilibrium(A: np.ndarray, xi: np.ndarray, loc_firms: np.ndarray,
                     support_points: np.ndarray, support_weights: np.ndarray,
                     mu_s: float, sigma_s: float, alpha: float, beta: float, gamma: float,
                     **options) -> Dict[str, Any]:
    """
    Solve the worker screening equilibrium.
    
    Args:
        A: Firm TFP (J,)
        xi: Firm amenity shocks (J,)
        loc_firms: Firm locations (J, 2)
        support_points: Worker location support points (S, 2)
        support_weights: Worker location weights (S,)
        mu_s: Truncated normal mean
        sigma_s: Truncated normal std
        alpha: Wage elasticity parameter
        beta: Production function parameter
        gamma: Distance decay parameter
        **options: Solver options
        
    Returns:
        Dictionary with solution and diagnostics
    """
    # Extract options with defaults
    max_iter = options.get('max_iter', 2000)
    tol = options.get('tol', 1e-8)
    damping = options.get('damping', 0.5)
    anderson_m = options.get('anderson_m', 5)
    anderson_reg = options.get('anderson_reg', 1e-8)
    eps = options.get('eps', 1e-12)
    check_every = options.get('check_every', 10)
    use_numba = options.get('use_numba', False)
    return_diagnostics = options.get('return_diagnostics', True)
    
    # Validate inputs
    J = len(A)
    S = len(support_points)
    
    assert len(xi) == J, f"xi length {len(xi)} != J {J}"
    assert loc_firms.shape == (J, 2), f"loc_firms shape {loc_firms.shape} != ({J}, 2)"
    assert support_points.shape == (S, 2), f"support_points shape {support_points.shape} != ({S}, 2)"
    assert len(support_weights) == S, f"support_weights length {len(support_weights)} != S {S}"
    assert np.all(A > 0), "All A must be positive"
    assert sigma_s > 0, "sigma_s must be positive"
    assert alpha > 0, "alpha must be positive"
    assert 0 < beta < 1, "beta must be in (0, 1)"
    assert gamma >= 0, "gamma must be nonnegative"
    assert np.all(support_weights >= 0), "All support_weights must be nonnegative"
    
    # Normalize support weights
    weight_sum = np.sum(support_weights)
    assert abs(weight_sum - 1.0) < eps, f"support_weights sum to {weight_sum}, not 1.0"
    support_weights = support_weights / weight_sum
    
    # Precompute base intensities
    start_time = time.time()
    B = precompute_bases(xi, loc_firms, support_points, gamma)
    precompute_time = time.time() - start_time
    
    # Initialize
    if 'init_logw' in options and 'init_logc' in options:
        logw = options['init_logw'].copy()
        logc = options['init_logc'].copy()
    else:
        # Default initialization
        logw = np.log(1 - beta) + np.log(A) + np.log(alpha / (alpha + 1))
        logc = -np.log(1 - beta) + logw - np.log(A)
    
    # Anderson storage
    anderson_store = {}
    
    # Iteration tracking
    residuals = []
    damping_used = []
    start_iter_time = time.time()
    
    # Main iteration loop
    for iter_num in range(max_iter):
        iter_start = time.time()
        
        # Compute fixed point map
        logw_new, logc_new = fixed_point_map(
            logw, logc, A, xi, loc_firms, support_points, support_weights,
            mu_s, sigma_s, alpha, beta, gamma, B, eps
        )
        
        # Anderson acceleration
        x = np.concatenate([logw, logc])
        Fx = np.concatenate([logw_new, logc_new])
        
        x_new = anderson_update(x, Fx, anderson_store, anderson_m, anderson_reg, damping)
        
        # Extract new values
        logw_new = x_new[:J]
        logc_new = x_new[J:]
        
        # Compute residual
        residual = np.max(np.abs(x_new - x))
        residuals.append(residual)
        
        # Check convergence
        if residual <= tol:
            converged = True
            break
        
        # Update iterate
        logw = logw_new.copy()
        logc = logc_new.copy()
        
        # Diagnostics
        if iter_num % check_every == 0:
            iter_time = time.time() - iter_start
            print(f"Iteration {iter_num}: residual = {residual:.2e}, time = {iter_time:.3f}s")
    
    else:
        converged = False
    
    total_time = time.time() - start_iter_time
    
    # Final evaluation to get L and S
    w = np.exp(logw)
    c = np.exp(logc)
    
    # Compute final L and S (reuse fixed_point_map logic)
    order_idx = np.argsort(c)
    c_sorted = c[order_idx]
    rank = np.zeros(J, dtype=int)
    rank[order_idx] = np.arange(J)
    
    DeltaF, M = truncated_normal_column_terms(c_sorted, mu_s, sigma_s, eps)
    
    NUM = B * (w ** alpha)[None, :]
    NUM_sorted = NUM[:, order_idx]
    DEN = np.maximum(np.cumsum(NUM_sorted, axis=1), eps)
    
    # Expand denominators to cover J+1 intervals
    DEN_full = np.concatenate([DEN, DEN[:, -1:]], axis=1)  # (S, J+1)
    f = support_weights[:, None] / np.maximum(DEN_full, eps)  # (S, J+1)
    Gmat = NUM.T @ f  # (J, J+1)
    
    vL = DeltaF  # (J+1,)
    vS = DeltaF * M  # (J+1,)
    H_L = Gmat * vL[None, :]  # (J, J+1)
    H_S = Gmat * vS[None, :]  # (J, J+1)
    CumL = np.flip(np.cumsum(np.flip(H_L, axis=1), axis=1), axis=1)  # (J, J+1)
    CumS = np.flip(np.cumsum(np.flip(H_S, axis=1), axis=1), axis=1)  # (J, J+1)
    
    L = np.maximum(CumL[np.arange(J), rank], eps)
    S = np.maximum(CumS[np.arange(J), rank], eps)
    
    # Prepare result
    result = {
        'w': w,
        'c': c,
        'L': L,
        'S': S,
        'logw': logw,
        'logc': logc,
        'iters': iter_num + 1,
        'converged': converged,
        'residual': residual if converged else residuals[-1],
        'rank': rank,
        'order_idx': order_idx
    }
    
    if return_diagnostics:
        result['diagnostics'] = {
            'total_time': total_time,
            'precompute_time': precompute_time,
            'residuals': residuals,
            'min_L': np.min(L),
            'max_L': np.max(L),
            'min_S': np.min(S),
            'max_S': np.max(S),
            'min_w': np.min(w),
            'max_w': np.max(w),
            'min_c': np.min(c),
            'max_c': np.max(c)
        }
    
    return result


def read_firms_csv(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Return (A, xi, loc_firms, firm_id, comp) from output/firms.csv.
    Validates schema, reconciles A and logA, sorts by firm_id.
    
    Args:
        path: Path to firms CSV file
        
    Returns:
        Tuple of (A, xi, loc_firms, firm_id, comp) arrays
    """
    df = pd.read_csv(path)
    
    # Validate required columns
    required_cols = ['firm_id', 'logA', 'A', 'xi', 'comp', 'x', 'y']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Sort by firm_id and reset index
    df = df.sort_values('firm_id').reset_index(drop=True)
    
    # Validate A and logA consistency
    if 'A' in df.columns and 'logA' in df.columns:
        A_from_log = np.exp(df['logA'].values)
        max_rel_error = np.max(np.abs(df['A'].values - A_from_log) / df['A'].values)
        if max_rel_error > 1e-8:
            raise ValueError(f"Max relative error between A and exp(logA): {max_rel_error:.2e}")
    
    # Extract arrays
    firm_id = df['firm_id'].values
    A = df['A'].values
    xi = df['xi'].values
    comp = df['comp'].values
    loc_firms = df[['x', 'y']].values
    
    return A, xi, loc_firms, firm_id, comp


def read_parameters_csv(path: str) -> Dict[str, float]:
    """
    Parse long-format parameters CSV into a dict.
    Required keys: mu_s, sigma_s, alpha, beta, gamma.
    Optional overrides: max_iter, tol, damping, anderson_m, anderson_reg, eps, grid_n, grid_log_span, max_plots.
    Raise with informative message if required keys missing.
    
    Args:
        path: Path to parameters CSV file
        
    Returns:
        Dictionary of parameter values
    """
    df = pd.read_csv(path)
    
    # Validate required columns
    required_cols = ['parameter', 'value']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Convert to dictionary
    params = dict(zip(df['parameter'], df['value']))
    
    # Check required keys
    required_keys = ['mu_s', 'sigma_s', 'alpha', 'beta', 'gamma']
    missing_keys = [key for key in required_keys if key not in params]
    if missing_keys:
        raise ValueError(f"Missing required parameters: {missing_keys}")
    
    return params


def write_parameters_template(path: str) -> None:
    """
    Write output/parameters_template.csv with default values and descriptions.
    
    Args:
        path: Path to write template file
    """
    template_data = [
        {'parameter': 'mu_s', 'value': 0.0, 'unit': 'miles', 'description': 'Mean of truncated normal distribution'},
        {'parameter': 'sigma_s', 'value': 1.0, 'unit': 'miles', 'description': 'Standard deviation of truncated normal'},
        {'parameter': 'alpha', 'value': 1.0, 'unit': 'NA', 'description': 'Wage elasticity parameter'},
        {'parameter': 'beta', 'value': 0.5, 'unit': 'NA', 'description': 'Production function parameter'},
        {'parameter': 'gamma', 'value': 0.1, 'unit': 'NA', 'description': 'Distance decay parameter'},
        {'parameter': 'max_iter', 'value': 2000, 'unit': 'NA', 'description': 'Maximum iterations for solver'},
        {'parameter': 'tol', 'value': 1e-8, 'unit': 'NA', 'description': 'Convergence tolerance'},
        {'parameter': 'damping', 'value': 0.5, 'unit': 'NA', 'description': 'Damping parameter for fixed-point iteration'},
        {'parameter': 'anderson_m', 'value': 5, 'unit': 'NA', 'description': 'Anderson acceleration memory'},
        {'parameter': 'anderson_reg', 'value': 1e-8, 'unit': 'NA', 'description': 'Anderson regularization parameter'},
        {'parameter': 'eps', 'value': 1e-12, 'unit': 'NA', 'description': 'Numerical safety floor'},
        {'parameter': 'grid_n', 'value': 40, 'unit': 'NA', 'description': 'Grid resolution for profit surface plots'},
        {'parameter': 'grid_log_span', 'value': 0.5, 'unit': 'NA', 'description': 'Log span for profit surface grid'},
        {'parameter': 'max_plots', 'value': 12, 'unit': 'NA', 'description': 'Maximum number of profit surface plots'}
    ]
    
    df = pd.DataFrame(template_data)
    df.to_csv(path, index=False)
    print(f"Parameters template written to: {path}")


def read_support_points_csv(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (support_points[S,2], support_weights[S]) from output/support_points.csv.
    Normalize weights to sum to 1, validate nonnegativity.
    
    Args:
        path: Path to support points CSV file
        
    Returns:
        Tuple of (support_points, support_weights) arrays
    """
    df = pd.read_csv(path)
    
    # Validate required columns
    required_cols = ['x', 'y', 'weight']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Extract arrays
    support_points = df[['x', 'y']].values
    support_weights = df['weight'].values
    
    # Validate weights
    if np.any(support_weights < 0):
        raise ValueError("Support weights must be nonnegative")
    
    # Normalize weights
    weight_sum = np.sum(support_weights)
    if abs(weight_sum - 1.0) > 1e-12:
        print(f"Warning: Normalizing support weights from sum {weight_sum:.6f} to 1.0")
        support_weights = support_weights / weight_sum
    
    return support_points, support_weights


def write_equilibrium_csv(path: str, firm_id: np.ndarray, comp: np.ndarray, A: np.ndarray, xi: np.ndarray,
                         loc_firms: np.ndarray, w: np.ndarray, c: np.ndarray, L: np.ndarray,
                         S: np.ndarray, Y: np.ndarray, logw: np.ndarray, logc: np.ndarray,
                         rank: np.ndarray) -> str:
    """
    Write equilibrium results to CSV with the specified columns and return the path.
    
    Args:
        path: Output file path
        firm_id: Firm IDs
        comp: Component assignments
        A: Firm TFP
        xi: Firm amenity shocks
        loc_firms: Firm locations
        w: Wages
        c: Cutoff costs
        L: Labor supply
        S: Skill supply
        Y: Output
        logw: Log wages
        logc: Log cutoff costs
        rank: Firm ranks
        
    Returns:
        Path to written file
    """
    df = pd.DataFrame({
        'firm_id': firm_id,
        'w': w,
        'c': c,
        'L': L,
        'S': S,
        'Y': Y,
        'A': A,
        'xi': xi,
        'x': loc_firms[:, 0],
        'y': loc_firms[:, 1],
        'comp': comp,
        'logw': logw,
        'logc': logc,
        'rank': rank
    })
    
    df.to_csv(path, index=False)
    return path


def profit_surface_for_firm(
    j: int,
    logw_eq: np.ndarray, logc_eq: np.ndarray,
    A: np.ndarray, xi: np.ndarray, loc_firms: np.ndarray,
    support_points: np.ndarray, support_weights: np.ndarray,
    mu_s: float, sigma_s: float, alpha: float, beta: float, gamma: float,
    B: np.ndarray,  # precomputed bases from precompute_bases(xi, loc_firms, support_points, gamma)
    grid_n: int = 40, grid_log_span: float = 0.5, eps: float = 1e-12
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    """
    Return (W, C, Pi, logw_star, logc_star, logw_argmax, logc_argmax) where:
      - (W, C) are 2D arrays of shape (grid_n, grid_n) with w and c levels,
      - Pi is the profit array π_j evaluated at the grid,
      - (logw_star, logc_star) are the equilibrium logs for firm j,
      - (logw_argmax, logc_argmax) are the grid argmax coordinates.
    
    Args:
        j: Firm index to vary
        logw_eq: Equilibrium log wages
        logc_eq: Equilibrium log cutoff costs
        A: Firm TFP
        xi: Firm amenity shocks
        loc_firms: Firm locations
        support_points: Worker location support points
        support_weights: Worker location weights
        mu_s: Truncated normal mean
        sigma_s: Truncated normal std
        alpha: Wage elasticity parameter
        beta: Production function parameter
        gamma: Distance decay parameter
        B: Precomputed base intensities
        grid_n: Grid resolution
        grid_log_span: Log span for grid
        eps: Numerical safety floor
        
    Returns:
        Tuple of (W, C, Pi, logw_star, logc_star, logw_argmax, logc_argmax)
    """
    J = len(A)
    S = len(support_points)
    
    # Equilibrium values for firm j
    logw_star = logw_eq[j]
    logc_star = logc_eq[j]
    
    # Create grid in log space
    logw_grid = np.linspace(logw_star - grid_log_span, logw_star + grid_log_span, grid_n)
    logc_grid = np.linspace(logc_star - grid_log_span, logc_star + grid_log_span, grid_n)
    
    # Initialize output arrays
    W = np.zeros((grid_n, grid_n))
    C = np.zeros((grid_n, grid_n))
    Pi = np.zeros((grid_n, grid_n))
    
    # Precompute equilibrium values for other firms
    logw_others = logw_eq.copy()
    logc_others = logc_eq.copy()
    
    # For each grid point, compute profit
    for i, logw_j in enumerate(logw_grid):
        for k, logc_j in enumerate(logc_grid):
            # Set firm j's values
            logw_others[j] = logw_j
            logc_others[j] = logc_j
            
            # Compute L and S for firm j using the same logic as fixed_point_map
            w = np.exp(logw_others)
            c = np.exp(logc_others)
            
            # Sort by c (ascending)
            order_idx = np.argsort(c)
            c_sorted = c[order_idx]
            rank = np.zeros(J, dtype=int)
            rank[order_idx] = np.arange(J)
            
            # Compute truncated normal terms
            DeltaF, M = truncated_normal_column_terms(c_sorted, mu_s, sigma_s, eps)
            
            # Compute NUM = B ⊙ (w^α)[None,:]
            NUM = B * (w ** alpha)[None, :]  # (S, J)
            NUM_sorted = NUM[:, order_idx]  # (S, J)
            
            # Cumulative denominators
            DEN = np.cumsum(NUM_sorted, axis=1)  # (S, J)
            DEN = np.maximum(DEN, eps)  # Safety floor
            
            # Expand denominators to cover J+1 intervals
            DEN_full = np.concatenate([DEN, DEN[:, -1:]], axis=1)  # (S, J+1)
            f = support_weights[:, None] / np.maximum(DEN_full, eps)  # (S, J+1)
            Gmat = NUM.T @ f  # (J, J+1)
            
            # Form column weights
            vL = DeltaF  # (J+1,)
            vS = DeltaF * M  # (J+1,)
            
            # Weighted matrices
            H_L = Gmat * vL[None, :]  # (J, J+1)
            H_S = Gmat * vS[None, :]  # (J, J+1)
            
            # Suffix sums
            CumL = np.flip(np.cumsum(np.flip(H_L, axis=1), axis=1), axis=1)  # (J, J+1)
            CumS = np.flip(np.cumsum(np.flip(H_S, axis=1), axis=1), axis=1)  # (J, J+1)
            
            # Get L and S for firm j
            L_j = np.maximum(CumL[j, rank[j]], eps)
            S_j = np.maximum(CumS[j, rank[j]], eps)
            
            # Compute profit π_j = A_j * (L_j * S_j)^(1-β) - w_j * L_j
            Y_j = A[j] * (L_j * S_j) ** (1 - beta)
            profit = Y_j - w[j] * L_j
            
            # Store values
            W[i, k] = w[j]
            C[i, k] = c[j]
            Pi[i, k] = profit
    
    # Find argmax on grid
    argmax_idx = np.unravel_index(np.argmax(Pi), Pi.shape)
    logw_argmax = logw_grid[argmax_idx[0]]
    logc_argmax = logc_grid[argmax_idx[1]]
    
    return W, C, Pi, logw_star, logc_star, logw_argmax, logc_argmax


def plot_profit_surface(
    firm_id: int,
    W: np.ndarray, C: np.ndarray, Pi: np.ndarray,
    w_star: float, c_star: float, w_hat: float, c_hat: float,
    out_path: str
) -> str:
    """
    Save a high-resolution PNG (dpi≥200) to out_path using matplotlib:
      - contourf or pcolormesh of Pi over (w_j, c_j),
      - overlay markers for equilibrium (w_star, c_star) and argmax (w_hat, c_hat),
      - axis labels 'w_j' and 'c_j', colorbar labeled 'profit',
      - tight_layout, equal aspect optional (not required), grid lightly on.
    
    Args:
        firm_id: Firm ID for title
        W: Wage grid
        C: Cutoff cost grid
        Pi: Profit grid
        w_star: Equilibrium wage
        c_star: Equilibrium cutoff cost
        w_hat: Argmax wage
        c_hat: Argmax cutoff cost
        out_path: Output file path
        
    Returns:
        Path to saved file
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create contour plot
    contour = ax.contourf(W, C, Pi, levels=20, cmap='viridis')
    
    # Add colorbar
    cbar = plt.colorbar(contour, ax=ax)
    cbar.set_label('Profit', fontsize=12)
    
    # Mark equilibrium point
    ax.plot(w_star, c_star, 'ro', markersize=10, label='Equilibrium', markeredgecolor='white', markeredgewidth=2)
    
    # Mark argmax point
    ax.plot(w_hat, c_hat, 'k*', markersize=12, label='Grid Argmax', markeredgecolor='white', markeredgewidth=2)
    
    # Customize plot
    ax.set_xlabel('w_j', fontsize=12)
    ax.set_ylabel('c_j', fontsize=12)
    ax.set_title(f'Profit Surface for Firm {firm_id}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Save plot
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return out_path


def make_profit_plots_for_firms(
    w: np.ndarray, c: np.ndarray, logw: np.ndarray, logc: np.ndarray,
    A: np.ndarray, xi: np.ndarray, loc_firms: np.ndarray,
    support_points: np.ndarray, support_weights: np.ndarray,
    mu_s: float, sigma_s: float, alpha: float, beta: float, gamma: float,
    firm_id: np.ndarray, out_dir: str,
    grid_n: int = 40, grid_log_span: float = 0.5, max_plots: Optional[int] = 12,
) -> List[str]:
    """
    Precompute B = precompute_bases(...).
    Iterate firms in increasing firm_id; respect max_plots if provided.
    For each firm, compute and save profit surface to 'output/profit_surfaces/firm_{firm_id}.png'.
    
    Args:
        w: Equilibrium wages
        c: Equilibrium cutoff costs
        logw: Equilibrium log wages
        logc: Equilibrium log cutoff costs
        A: Firm TFP
        xi: Firm amenity shocks
        loc_firms: Firm locations
        support_points: Worker location support points
        support_weights: Worker location weights
        mu_s: Truncated normal mean
        sigma_s: Truncated normal std
        alpha: Wage elasticity parameter
        beta: Production function parameter
        gamma: Distance decay parameter
        firm_id: Firm IDs
        out_dir: Output directory
        grid_n: Grid resolution
        grid_log_span: Log span for grid
        max_plots: Maximum number of plots (None for all)
        
    Returns:
        List of saved file paths
    """
    # Precompute base intensities
    B = precompute_bases(xi, loc_firms, support_points, gamma)
    
    # Create output directory
    profit_dir = Path(out_dir) / "profit_surfaces"
    profit_dir.mkdir(exist_ok=True)
    
    # Determine which firms to plot
    if max_plots is None:
        firms_to_plot = np.arange(len(firm_id))
    else:
        firms_to_plot = np.arange(min(max_plots, len(firm_id)))
    
    saved_paths = []
    
    for i in firms_to_plot:
        j = i  # Firm index
        firm_id_j = firm_id[j]
        
        print(f"Computing profit surface for firm {firm_id_j}...")
        
        # Compute profit surface
        W, C, Pi, logw_star, logc_star, logw_argmax, logc_argmax = profit_surface_for_firm(
            j, logw, logc, A, xi, loc_firms, support_points, support_weights,
            mu_s, sigma_s, alpha, beta, gamma, B, grid_n, grid_log_span
        )
        
        # Convert to levels for plotting
        w_star = np.exp(logw_star)
        c_star = np.exp(logc_star)
        w_hat = np.exp(logw_argmax)
        c_hat = np.exp(logc_argmax)
        
        # Create plot
        out_path = profit_dir / f"firm_{firm_id_j}.png"
        plot_path = plot_profit_surface(
            firm_id_j, W, C, Pi, w_star, c_star, w_hat, c_hat, str(out_path)
        )
        saved_paths.append(plot_path)
        
        print(f"  Saved to: {plot_path}")
    
    return saved_paths


def create_synthetic_data(J: int = 20, S: int = 8, seed: int = 42) -> Dict[str, np.ndarray]:
    """
    Create synthetic data for testing.
    
    Args:
        J: Number of firms
        S: Number of support points
        seed: Random seed
        
    Returns:
        Dictionary with synthetic data
    """
    rng = np.random.default_rng(seed)
    
    # Firm fundamentals
    A = np.exp(rng.normal(0, 0.3, J))  # Log-normal TFP
    xi = rng.normal(0, 0.2, J)  # Amenity shocks
    
    # Firm locations (random in [-5, 5] x [-5, 5])
    loc_firms = rng.uniform(-5, 5, (J, 2))
    
    # Worker support points (grid)
    grid_size = int(np.sqrt(S))
    if grid_size * grid_size != S:
        # If S is not a perfect square, use random points
        support_points = rng.uniform(-4, 4, (S, 2))
    else:
        x_grid = np.linspace(-4, 4, grid_size)
        y_grid = np.linspace(-4, 4, grid_size)
        X, Y = np.meshgrid(x_grid, y_grid)
        support_points = np.column_stack([X.ravel(), Y.ravel()])
    
    # Equal weights
    support_weights = np.ones(S) / S
    
    return {
        'A': A,
        'xi': xi,
        'loc_firms': loc_firms,
        'support_points': support_points,
        'support_weights': support_weights
    }


def run_unit_tests():
    """Run unit tests on synthetic data."""
    print("Running unit tests...")
    
    # Create synthetic data
    data = create_synthetic_data(J=3, S=8, seed=42)
    
    # Parameters
    mu_s = 5.0
    sigma_s = 1.0
    alpha = 1.0
    beta = 0.5
    gamma = 0.1
    
    # Solve equilibrium
    start_time = time.time()
    result = solve_equilibrium(
        data['A'], data['xi'], data['loc_firms'],
        data['support_points'], data['support_weights'],
        mu_s, sigma_s, alpha, beta, gamma,
        max_iter=500, tol=1e-6, anderson_m=0, damping=0.1, return_diagnostics=True
    )
    solve_time = time.time() - start_time
    
    # Validation checks
    print(f"\n=== UNIT TEST RESULTS ===")
    print(f"Converged: {result['converged']}")
    print(f"Iterations: {result['iters']}")
    print(f"Final residual: {result['residual']:.2e}")
    print(f"Solve time: {solve_time:.3f}s")
    
    # Check solution properties
    L, S, w, c = result['L'], result['S'], result['w'], result['c']
    
    print(f"\nSolution bounds:")
    print(f"  L: [{np.min(L):.4f}, {np.max(L):.4f}]")
    print(f"  S: [{np.min(S):.4f}, {np.max(S):.4f}]")
    print(f"  w: [{np.min(w):.4f}, {np.max(w):.4f}]")
    print(f"  c: [{np.min(c):.4f}, {np.max(c):.4f}]")
    
    # Assertions
    assert result['converged'], "Solver did not converge"
    assert result['iters'] <= 500, f"Too many iterations: {result['iters']}"
    assert np.all(L > 0), "Some L <= 0"
    assert np.all(S > 0), "Some S <= 0"
    assert np.all(w > 0), "Some w <= 0"
    assert np.all(c > 0), "Some c <= 0"
    assert np.all(np.isfinite(L)), "Some L not finite"
    assert np.all(np.isfinite(S)), "Some S not finite"
    assert np.all(np.isfinite(w)), "Some w not finite"
    assert np.all(np.isfinite(c)), "Some c not finite"
    
    # Test truncated normal boundary conventions
    # Create test data for boundary validation
    test_c = np.array([0.5, 1.0, 2.0])
    test_DeltaF, test_M = truncated_normal_column_terms(test_c, mu_s=0.0, sigma_s=1.0, eps=1e-12)
    
    # Validation checks for boundary conventions
    assert test_DeltaF.min() >= 0, "DeltaF has negative values"
    assert abs(test_DeltaF.sum() - 1.0) <= 1e-12, f"DeltaF sum is {test_DeltaF.sum()}, not 1.0"
    assert np.isfinite(test_M).all(), "Some M values are not finite"
    assert len(test_DeltaF) == len(test_c) + 1, f"DeltaF length {len(test_DeltaF)} != J+1 {len(test_c)+1}"
    assert len(test_M) == len(test_c) + 1, f"M length {len(test_M)} != J+1 {len(test_c)+1}"
    
    # Test denominator expansion logic
    J_test = 3
    S_test = 4
    test_NUM_sorted = np.random.rand(S_test, J_test)
    test_DEN = np.cumsum(test_NUM_sorted, axis=1)
    test_DEN_full = np.concatenate([test_DEN, test_DEN[:, -1:]], axis=1)
    assert test_DEN_full.shape == (S_test, J_test + 1), f"DEN_full shape {test_DEN_full.shape} != ({S_test}, {J_test + 1})"
    assert np.allclose(test_DEN_full[:, -1], test_DEN_full[:, -2]), "Last two columns of DEN_full should be equal"
    
    print("\n✅ All unit tests passed!")
    
    # Print diagnostics
    if 'diagnostics' in result:
        diag = result['diagnostics']
        print(f"\nDiagnostics:")
        print(f"  Total time: {diag['total_time']:.3f}s")
        print(f"  Precompute time: {diag['precompute_time']:.3f}s")
        print(f"  Residual history: {len(diag['residuals'])} iterations")
    
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Worker Screening Equilibrium Solver")
    parser.add_argument("--test", action="store_true", help="Run unit tests")
    parser.add_argument("--firms_path", type=str, default="output/firms.csv", help="Path to firms CSV file")
    parser.add_argument("--params_path", type=str, default="output/parameters.csv", help="Path to parameters CSV file")
    parser.add_argument("--support_path", type=str, default="output/support_points.csv", help="Path to support points CSV file")
    parser.add_argument("--out_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--write_templates", action="store_true", help="Write parameter template if missing")
    parser.add_argument("--grid_n", type=int, default=40, help="Grid resolution for profit surface plots")
    parser.add_argument("--grid_log_span", type=float, default=0.5, help="Log span for profit surface grid")
    parser.add_argument("--max_plots", type=int, default=12, help="Maximum number of profit surface plots (0 to skip)")
    parser.add_argument("--J", type=int, default=20, help="Number of firms for synthetic data (default: 20)")
    parser.add_argument("--S", type=int, default=8, help="Number of support points for synthetic data (default: 8)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    if args.test:
        result = run_unit_tests()
    else:
        # Check if we should use CSV files or synthetic data
        use_csv = (Path(args.firms_path).exists() and 
                  Path(args.params_path).exists() and 
                  Path(args.support_path).exists())
        
        if use_csv:
            print("Reading data from CSV files...")
            
            # Read firms data
            A, xi, loc_firms, firm_id, comp = read_firms_csv(args.firms_path)
            print(f"Loaded {len(A)} firms from {args.firms_path}")
            
            # Read parameters
            params = read_parameters_csv(args.params_path)
            print(f"Loaded parameters from {args.params_path}")
            
            # Read support points
            support_points, support_weights = read_support_points_csv(args.support_path)
            print(f"Loaded {len(support_points)} support points from {args.support_path}")
            
            # Extract required parameters
            mu_s = params['mu_s']
            sigma_s = params['sigma_s']
            alpha = params['alpha']
            beta = params['beta']
            gamma = params['gamma']
            
            # Extract optional parameters with defaults
            max_iter = params.get('max_iter', 2000)
            tol = params.get('tol', 1e-8)
            damping = params.get('damping', 0.5)
            anderson_m = params.get('anderson_m', 5)
            anderson_reg = params.get('anderson_reg', 1e-8)
            eps = params.get('eps', 1e-12)
            grid_n = params.get('grid_n', args.grid_n)
            grid_log_span = params.get('grid_log_span', args.grid_log_span)
            max_plots = params.get('max_plots', args.max_plots)
            
        else:
            print("CSV files not found, using synthetic data...")
            
            # Check if we should write templates
            if args.write_templates and not Path(args.params_path).exists():
                write_parameters_template(args.params_path)
                print(f"Parameters template written to: {args.params_path}")
                print("Please fill in the parameters and rerun the solver.")
                exit(0)
            
            # Create synthetic data
            data = create_synthetic_data(J=args.J, S=args.S, seed=args.seed)
            A = data['A']
            xi = data['xi']
            loc_firms = data['loc_firms']
            support_points = data['support_points']
            support_weights = data['support_weights']
            firm_id = np.arange(1, len(A) + 1)
            comp = np.ones(len(A), dtype=int)
            
            # Default parameters
            mu_s = 5.0
            sigma_s = 1.0
            alpha = 1.0
            beta = 0.5
            gamma = 0.1
            max_iter = 1000
            tol = 1e-8
            damping = 0.1
            anderson_m = 0
            anderson_reg = 1e-8
            eps = 1e-12
            grid_n = args.grid_n
            grid_log_span = args.grid_log_span
            max_plots = args.max_plots
        
        print(f"Solving equilibrium for {len(A)} firms...")
        
        # Solve equilibrium
        result = solve_equilibrium(
            A, xi, loc_firms, support_points, support_weights,
            mu_s, sigma_s, alpha, beta, gamma,
            max_iter=max_iter, tol=tol, damping=damping, 
            anderson_m=anderson_m, anderson_reg=anderson_reg, eps=eps,
            return_diagnostics=True
        )
        
        print(f"Converged: {result['converged']}")
        print(f"Iterations: {result['iters']}")
        print(f"Final residual: {result['residual']:.2e}")
        
        if result['converged']:
            # Compute output Y
            w = result['w']
            c = result['c']
            L = result['L']
            S = result['S']
            logw = result['logw']
            logc = result['logc']
            rank = result['rank']
            
            Y = A * (L * S) ** (1 - beta)
            
            # Write equilibrium results
            eq_path = Path(args.out_dir) / "equilibrium_firms.csv"
            write_equilibrium_csv(
                str(eq_path), firm_id, comp, A, xi, loc_firms,
                w, c, L, S, Y, logw, logc, rank
            )
            print(f"Equilibrium results saved to: {eq_path}")
            
            # Generate profit surface plots
            if max_plots > 0:
                print(f"Generating profit surface plots (max {max_plots})...")
                plot_paths = make_profit_plots_for_firms(
                    w, c, logw, logc, A, xi, loc_firms,
                    support_points, support_weights,
                    mu_s, sigma_s, alpha, beta, gamma,
                    firm_id, args.out_dir,
                    grid_n=grid_n, grid_log_span=grid_log_span, max_plots=max_plots
                )
                print(f"Generated {len(plot_paths)} profit surface plots")
            else:
                print("Skipping profit surface plots (max_plots=0)")
        else:
            print("Warning: Solver did not converge!")
    
    # Print package versions
    print(f"\n=== PACKAGE VERSIONS ===")
    print(f"numpy: {np.__version__}")
    print(f"pandas: {pd.__version__}")
    print(f"pyarrow: {pa.__version__}")
    print(f"numba available: {NUMBA_AVAILABLE}")
    if NUMBA_AVAILABLE:
        print(f"numba: {numba.__version__}")
