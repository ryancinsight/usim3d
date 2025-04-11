"""
Grid Optimizer for k-wave Simulations

This module provides functions to optimize grid sizes for k-wave simulations
by finding dimensions with small prime factors, which improves FFT performance.
"""

import numpy as np
import math

def prime_factors(n):
    """
    Find all prime factors of a number

    Parameters:
    -----------
    n : int
        The number to factorize

    Returns:
    --------
    list
        List of prime factors
    """
    factors = []
    # Check for 2 as a factor
    while n % 2 == 0:
        factors.append(2)
        n = n // 2

    # Check for odd factors
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        while n % i == 0:
            factors.append(i)
            n = n // i

    # If n is a prime number greater than 2
    if n > 2:
        factors.append(n)

    return factors

def get_optimal_grid_size(min_size, max_size=None, max_prime_factor=13):
    """
    Find the optimal grid size with small prime factors

    Parameters:
    -----------
    min_size : int
        Minimum required grid size
    max_size : int or None
        Maximum allowed grid size (if None, will use min_size * 1.2)
    max_prime_factor : int
        Maximum allowed prime factor

    Returns:
    --------
    int
        Optimal grid size with small prime factors
    """
    if max_size is None:
        max_size = int(min_size * 1.2)  # Allow up to 20% increase

    best_size = min_size
    best_max_prime = float('inf')

    for size in range(min_size, max_size + 1):
        factors = prime_factors(size)
        current_max_prime = max(factors) if factors else 0

        # If we find a size with all prime factors <= max_prime_factor, return it immediately
        if current_max_prime <= max_prime_factor:
            return size

        # Otherwise, keep track of the size with the smallest maximum prime factor
        if current_max_prime < best_max_prime:
            best_max_prime = current_max_prime
            best_size = size

    return best_size

def optimize_grid_dimensions(Nx, Ny, Nz, max_increase_percent=20, pml_size=None):
    """
    Optimize all three grid dimensions for FFT efficiency

    Parameters:
    -----------
    Nx, Ny, Nz : int
        Original grid dimensions
    max_increase_percent : int
        Maximum allowed percentage increase in each dimension
    pml_size : int or None
        Size of PML layer to account for in optimization. If None, no adjustment is made.

    Returns:
    --------
    tuple
        Optimized (Nx, Ny, Nz) dimensions
    """
    max_Nx = int(Nx * (1 + max_increase_percent/100))
    max_Ny = int(Ny * (1 + max_increase_percent/100))
    max_Nz = int(Nz * (1 + max_increase_percent/100))

    # If PML size is provided, adjust the grid dimensions to account for it
    # This is because k-wave may add PML layers internally, changing the effective grid size
    if pml_size is not None:
        # For each dimension, try to find a size that will result in optimal grid dimensions
        # after k-wave adds the PML layers
        Nx_with_pml = Nx + 2 * pml_size
        Ny_with_pml = Ny + 2 * pml_size
        Nz_with_pml = Nz + 2 * pml_size

        # Find optimal sizes for the grid with PML
        opt_Nx_with_pml = get_optimal_grid_size(Nx_with_pml, int(Nx_with_pml * (1 + max_increase_percent/100)))
        opt_Ny_with_pml = get_optimal_grid_size(Ny_with_pml, int(Ny_with_pml * (1 + max_increase_percent/100)))
        opt_Nz_with_pml = get_optimal_grid_size(Nz_with_pml, int(Nz_with_pml * (1 + max_increase_percent/100)))

        # Calculate the corresponding grid sizes without PML
        opt_Nx = opt_Nx_with_pml - 2 * pml_size
        opt_Ny = opt_Ny_with_pml - 2 * pml_size
        opt_Nz = opt_Nz_with_pml - 2 * pml_size

        # Make sure the optimized sizes are at least as large as the original sizes
        opt_Nx = max(Nx, opt_Nx)
        opt_Ny = max(Ny, opt_Ny)
        opt_Nz = max(Nz, opt_Nz)

        # Make sure the optimized sizes don't exceed the maximum allowed increase
        opt_Nx = min(opt_Nx, max_Nx)
        opt_Ny = min(opt_Ny, max_Ny)
        opt_Nz = min(opt_Nz, max_Nz)

        # Print information about the optimization with PML consideration
        print(f"Grid optimization (accounting for PML size {pml_size}):")
        print(f"  Original dimensions: {Nx} x {Ny} x {Nz}")
        print(f"  Original with PML: {Nx_with_pml} x {Ny_with_pml} x {Nz_with_pml}")
        print(f"  Original prime factors with PML: {prime_factors(Nx_with_pml)} x {prime_factors(Ny_with_pml)} x {prime_factors(Nz_with_pml)}")
        print(f"  Optimized with PML: {opt_Nx_with_pml} x {opt_Ny_with_pml} x {opt_Nz_with_pml}")
        print(f"  Optimized prime factors with PML: {prime_factors(opt_Nx_with_pml)} x {prime_factors(opt_Ny_with_pml)} x {prime_factors(opt_Nz_with_pml)}")
        print(f"  Final optimized dimensions: {opt_Nx} x {opt_Ny} x {opt_Nz}")
        print(f"  Size increase: {(opt_Nx/Nx - 1)*100:.1f}% x {(opt_Ny/Ny - 1)*100:.1f}% x {(opt_Nz/Nz - 1)*100:.1f}%")
    else:
        # Standard optimization without PML consideration
        opt_Nx = get_optimal_grid_size(Nx, max_Nx)
        opt_Ny = get_optimal_grid_size(Ny, max_Ny)
        opt_Nz = get_optimal_grid_size(Nz, max_Nz)

        # Print information about the optimization
        print(f"Grid optimization:")
        print(f"  Original dimensions: {Nx} x {Ny} x {Nz}")
        print(f"  Original prime factors: {prime_factors(Nx)} x {prime_factors(Ny)} x {prime_factors(Nz)}")
        print(f"  Optimized dimensions: {opt_Nx} x {opt_Ny} x {opt_Nz}")
        print(f"  Optimized prime factors: {prime_factors(opt_Nx)} x {prime_factors(opt_Ny)} x {prime_factors(opt_Nz)}")
        print(f"  Size increase: {(opt_Nx/Nx - 1)*100:.1f}% x {(opt_Ny/Ny - 1)*100:.1f}% x {(opt_Nz/Nz - 1)*100:.1f}%")

    return opt_Nx, opt_Ny, opt_Nz

def get_highest_prime_factors(Nx, Ny, Nz):
    """
    Get the highest prime factor for each dimension

    Parameters:
    -----------
    Nx, Ny, Nz : int
        Grid dimensions

    Returns:
    --------
    list
        Highest prime factor for each dimension
    """
    return [max(prime_factors(Nx)), max(prime_factors(Ny)), max(prime_factors(Nz))]

if __name__ == "__main__":
    # Test the functions
    test_dims = [128, 129, 130, 131, 132]
    for dim in test_dims:
        factors = prime_factors(dim)
        print(f"Prime factors of {dim}: {factors}")

    # Test optimization
    original = (130, 130, 166)
    optimized = optimize_grid_dimensions(*original)
    print(f"Optimized {original} to {optimized}")
