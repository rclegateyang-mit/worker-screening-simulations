#!/usr/bin/env python3
"""
Worker Screening Spatial Simulation

This module simulates firms and workers in 2D space with specified distributions
for firm fundamentals and spatial locations.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pyarrow as pa
import pyarrow.parquet as pq


def build_covariance(sA: float, sXi: float, rho: float) -> np.ndarray:
    """
    Build 2x2 covariance matrix from standard deviations and correlation.
    
    Args:
        sA: Standard deviation of logA
        sXi: Standard deviation of xi
        rho: Correlation between logA and xi
        
    Returns:
        2x2 covariance matrix
        
    Raises:
        AssertionError: If covariance matrix is not positive definite
    """
    cov = np.array([
        [sA**2, rho * sA * sXi],
        [rho * sA * sXi, sXi**2]
    ])
    
    # Validate positive definiteness
    eigenvals = np.linalg.eigvals(cov)
    assert np.all(eigenvals > 0), f"Covariance matrix not positive definite. Eigenvalues: {eigenvals}"
    
    return cov


def draw_firm_fundamentals(J: int, cov: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """
    Draw firm fundamentals from bivariate normal distribution.
    
    Args:
        J: Number of firms
        cov: 2x2 covariance matrix
        rng: Random number generator
        
    Returns:
        DataFrame with firm fundamentals
    """
    # Draw from bivariate normal
    logA, xi = rng.multivariate_normal(mean=[0, 0], cov=cov, size=J).T
    
    # Compute A = exp(logA)
    A = np.exp(logA)
    
    # Create DataFrame
    df = pd.DataFrame({
        'firm_id': np.arange(1, J + 1),
        'logA': logA,
        'A': A,
        'xi': xi
    })
    
    return df


def draw_firm_locations(J: int, centers: np.ndarray, sds: np.ndarray, 
                       weights: np.ndarray, rng: np.random.Generator) -> pd.DataFrame:
    """
    Draw firm locations from Gaussian mixture model.
    
    Args:
        J: Number of firms
        centers: Mixture centers (K x 2)
        sds: Component standard deviations (K x 2)
        weights: Mixture weights (K,)
        rng: Random number generator
        
    Returns:
        DataFrame with firm locations and component assignments
    """
    # Validate weights
    assert np.abs(np.sum(weights) - 1.0) < 1e-12, f"Weights sum to {np.sum(weights)}, not 1.0"
    assert np.all((weights >= 0) & (weights <= 1)), f"Weights not in [0,1]: {weights}"
    
    # Draw component assignments
    comp = rng.choice(len(centers), size=J, p=weights)
    
    # Draw locations conditional on component
    x_coords = np.zeros(J)
    y_coords = np.zeros(J)
    
    for k in range(len(centers)):
        mask = comp == k
        if np.any(mask):
            x_coords[mask] = rng.normal(centers[k, 0], sds[k, 0], size=np.sum(mask))
            y_coords[mask] = rng.normal(centers[k, 1], sds[k, 1], size=np.sum(mask))
    
    # Create DataFrame
    df = pd.DataFrame({
        'firm_id': np.arange(1, J + 1),
        'comp': comp + 1,  # 1-indexed components
        'x': x_coords,
        'y': y_coords
    })
    
    return df


def draw_workers(I: int, mean: np.ndarray, std: np.ndarray, 
                rng: np.random.Generator) -> pd.DataFrame:
    """
    Draw worker locations from single anisotropic Gaussian.
    
    Args:
        I: Number of workers
        mean: Mean location (2,)
        std: Standard deviations (2,)
        rng: Random number generator
        
    Returns:
        DataFrame with worker locations
    """
    # Draw from bivariate normal
    x_coords, y_coords = rng.multivariate_normal(
        mean=mean, 
        cov=np.diag(std**2), 
        size=I
    ).T
    
    # Create DataFrame
    df = pd.DataFrame({
        'worker_id': np.arange(1, I + 1),
        'x': x_coords,
        'y': y_coords
    })
    
    return df


def plot_spatial(firms: pd.DataFrame, workers: pd.DataFrame, out_path: str) -> None:
    """
    Create spatial distribution plot.
    
    Args:
        firms: Firms DataFrame
        workers: Workers DataFrame
        out_path: Output path for plot
    """
    # Set up the plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot workers (background)
    ax.scatter(workers['x'], workers['y'], 
              c='gray', alpha=0.25, s=10, zorder=1, label='Workers')
    
    # Plot firms by component
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, orange, green
    for comp in [1, 2, 3]:
        mask = firms['comp'] == comp
        if np.any(mask):
            ax.scatter(firms.loc[mask, 'x'], firms.loc[mask, 'y'],
                      c=colors[comp-1], alpha=0.8, s=30, zorder=2,
                      label=f'Firms (Component {comp})')
    
    # Customize plot
    ax.set_xlabel('x (miles)')
    ax.set_ylabel('y (miles)')
    ax.set_title('Spatial distribution of firms (by component) and workers')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    # Save plot
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


def main(J: int = 3000, I: int = 20000, seed: int = 123, out_dir: str = "output") -> None:
    """
    Main simulation function.
    
    Args:
        J: Number of firms
        I: Number of workers
        seed: Random seed
        out_dir: Output directory
    """
    # Set random seed
    rng = np.random.default_rng(seed)
    
    # Create output directory
    out_path = Path(out_dir)
    out_path.mkdir(exist_ok=True)
    
    # Check if directory is writable
    if not os.access(out_path, os.W_OK):
        raise PermissionError(f"Cannot write to output directory: {out_path}")
    
    # Define parameters
    # Firm fundamentals parameters
    sA, sXi, rho = 0.37, 0.2, 0.44
    cov = build_covariance(sA, sXi, rho)
    
    # Firm location parameters (Gaussian mixture)
    centers = np.array([
        [0, 0],      # c1
        [10, 4],     # c2
        [-8, 6]      # c3
    ])
    
    weights = np.array([0.55, 0.25, 0.20])
    
    sds = np.array([
        [2.2, 1.0],  # Component 1
        [3.0, 1.4],  # Component 2
        [2.6, 1.2]   # Component 3
    ])
    
    # Worker location parameters
    worker_mean = np.array([0, 0])
    worker_std = np.array([2, np.sqrt(2)])  # std = (2, √2)
    
    print(f"Simulating {J} firms and {I} workers...")
    
    # Generate data
    firm_fundamentals = draw_firm_fundamentals(J, cov, rng)
    firm_locations = draw_firm_locations(J, centers, sds, weights, rng)
    workers = draw_workers(I, worker_mean, worker_std, rng)
    
    # Merge firm fundamentals and locations
    firms = pd.merge(firm_fundamentals, firm_locations, on='firm_id', how='inner')
    firms = firms.sort_values('firm_id').reset_index(drop=True)
    
    # Validation checks
    assert len(firms) == J, f"Expected {J} firms, got {len(firms)}"
    assert len(workers) == I, f"Expected {I} workers, got {len(workers)}"
    
    # Check for NaNs and infs
    numeric_cols = firms.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        assert not firms[col].isna().any(), f"NaNs found in firms.{col}"
        assert not np.isinf(firms[col]).any(), f"Infs found in firms.{col}"
    
    worker_numeric_cols = workers.select_dtypes(include=[np.number]).columns
    for col in worker_numeric_cols:
        assert not workers[col].isna().any(), f"NaNs found in workers.{col}"
        assert not np.isinf(workers[col]).any(), f"Infs found in workers.{col}"
    
    # Component count validation
    comp_counts = firms['comp'].value_counts().sort_index()
    for comp in [1, 2, 3]:
        if comp_counts.get(comp, 0) == 0:
            print(f"Warning: Component {comp} has zero firms")
    
    # Save data
    print("Saving data...")
    
    # CSV files
    firms.to_csv(out_path / "firms.csv", index=False)
    workers.to_csv(out_path / "workers.csv", index=False)
    
    # Parquet files
    firms.to_parquet(out_path / "firms.parquet", index=False)
    workers.to_parquet(out_path / "workers.parquet", index=False)
    
    # Create plot
    print("Creating plot...")
    plot_spatial(firms, workers, out_path / "sim_scatter.png")
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Firm component counts:")
    for comp in [1, 2, 3]:
        count = len(firms[firms['comp'] == comp])
        print(f"  Component {comp}: {count} firms")
    
    print(f"\nFirm fundamentals:")
    print(f"  A: mean={firms['A'].mean():.4f}, std={firms['A'].std():.4f}")
    print(f"  xi: mean={firms['xi'].mean():.4f}, std={firms['xi'].std():.4f}")
    
    print(f"\nSpatial bounds:")
    print(f"  Firms: x∈[{firms['x'].min():.2f}, {firms['x'].max():.2f}], "
          f"y∈[{firms['y'].min():.2f}, {firms['y'].max():.2f}]")
    print(f"  Workers: x∈[{workers['x'].min():.2f}, {workers['x'].max():.2f}], "
          f"y∈[{workers['y'].min():.2f}, {workers['y'].max():.2f}]")
    
    print(f"\nOutput saved to: {out_path}")
    
    # Print package versions
    print(f"\n=== PACKAGE VERSIONS ===")
    print(f"numpy: {np.__version__}")
    print(f"pandas: {pd.__version__}")
    print(f"matplotlib: {plt.matplotlib.__version__}")
    print(f"pyarrow: {pa.__version__}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Worker Screening Spatial Simulation")
    parser.add_argument("--J", type=int, default=3000, help="Number of firms (default: 3000)")
    parser.add_argument("--I", type=int, default=20000, help="Number of workers (default: 20000)")
    parser.add_argument("--seed", type=int, default=123, help="Random seed (default: 123)")
    parser.add_argument("--out_dir", type=str, default="output", help="Output directory (default: output)")
    
    args = parser.parse_args()
    
    main(J=args.J, I=args.I, seed=args.seed, out_dir=args.out_dir)
