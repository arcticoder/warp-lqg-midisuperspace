#!/usr/bin/env python3
"""
Solve Constraint (Task 4)

Numerically solves the quantum constraint Ĥ|Ψ⟩ = 0:
- Constructs master constraint M̂ = Ĥ†Ĥ
- Finds zero eigenvalues using sparse linear algebra
- Identifies semiclassical coherent states
- Validates physical state properties

Author: Loop Quantum Gravity Implementation
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import argparse
import json
import os
from typing import Dict, Tuple, List, Optional

from kinematical_hilbert import MidisuperspaceHilbert, load_lattice_from_reduced_variables
from hamiltonian_constraint import HamiltonianConstraint


class ConstraintSolver:
    """
    Solves quantum constraint equation for physical states
    
    Methods:
    1. Exact diagonalization for small Hilbert spaces
    2. Iterative eigensolvers for larger spaces  
    3. Semiclassical coherent state analysis
    """
    
    def __init__(self, hilbert: MidisuperspaceHilbert, 
                 constraint: HamiltonianConstraint):
        self.hilbert = hilbert
        self.constraint = constraint
        self.physical_states = []
        self.eigenvalues = []
        
    def solve_master_constraint(self, n_states: int = 5, 
                              tolerance: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve master constraint M̂|Ψₙ⟩ = λₙ|Ψₙ⟩
        
        Physical states correspond to λₙ ≈ 0
        
        Args:
            n_states: Number of lowest eigenvalues to compute
            tolerance: Convergence tolerance for eigenvalues
            
        Returns:
            (eigenvalues, eigenvectors) with shape (n_states,) and (dim, n_states)
        """
        print("Computing master constraint operator...")
        M = self.constraint.master_constraint_operator()
        
        print(f"Master constraint matrix: {M.shape}")
        print(f"Non-zero elements: {M.nnz}")
        print(f"Sparsity: {M.nnz / (M.shape[0]**2):.6f}")
        
        if M.shape[0] <= 100:
            # Small matrices: use dense eigendecomposition
            print("Using dense eigendecomposition...")
            M_dense = M.toarray()
            eigenvals, eigenvecs = np.linalg.eigh(M_dense)
            
            # Sort by eigenvalue magnitude
            idx = np.argsort(eigenvals)
            eigenvals = eigenvals[idx][:n_states]
            eigenvecs = eigenvecs[:, idx][:, :n_states]
            
        else:
            # Large matrices: use sparse iterative solver
            print("Using sparse eigendecomposition...")
            
            try:
                eigenvals, eigenvecs = spla.eigsh(
                    M, k=n_states, which='SM',  # Smallest magnitude
                    tol=tolerance, maxiter=1000
                )
                
                # Sort by eigenvalue
                idx = np.argsort(eigenvals)
                eigenvals = eigenvals[idx]
                eigenvecs = eigenvecs[:, idx]
                
            except spla.ArpackNoConvergence as e:
                print(f"ARPACK convergence warning: {e}")
                eigenvals = e.eigenvalues
                eigenvecs = e.eigenvectors
                
                if eigenvals is not None:
                    idx = np.argsort(eigenvals)
                    eigenvals = eigenvals[idx]
                    eigenvecs = eigenvecs[:, idx]
        
        self.eigenvalues = eigenvals
        self.physical_states = eigenvecs
        
        print(f"\nEigenvalue spectrum:")
        for i, lam in enumerate(eigenvals):
            print(f"  lambda_{i} = {lam:.2e}")
        
        # Identify physical states (near-zero eigenvalues)
        physical_threshold = 1e-6
        n_physical = np.sum(eigenvals < physical_threshold)
        print(f"\nPhysical states (lambda < {physical_threshold}): {n_physical}")
        
        return eigenvals, eigenvecs
    
    def validate_physical_state(self, state_index: int = 0) -> Dict:
        """
        Validate properties of physical state
        
        Checks:
        1. Normalization: ⟨Ψ|Ψ⟩ = 1
        2. Constraint satisfaction: ⟨Ψ|Ĥ†Ĥ|Ψ⟩ ≈ 0
        3. Expectation values of observables
        """
        if len(self.physical_states) == 0:
            raise ValueError("No physical states computed yet")
        
        psi = self.physical_states[:, state_index]
        
        # Check normalization
        norm = np.linalg.norm(psi)
        print(f"\nValidating physical state {state_index}:")
        print(f"  Normalization: ||Psi|| = {norm:.6f}")
        
        # Check constraint violation
        M = self.constraint.master_constraint_operator()
        constraint_violation = np.real(np.conj(psi) @ M @ psi)
        print(f"  Constraint violation: <Psi|M_hat|Psi> = {constraint_violation:.2e}")
        
        # Compute flux expectation values
        flux_expectations = {}
        for site in range(self.hilbert.n_sites):
            E_x_op = self.hilbert.flux_E_x_operator(site)
            E_phi_op = self.hilbert.flux_E_phi_operator(site)
            
            exp_E_x = np.real(np.conj(psi) @ E_x_op @ psi)
            exp_E_phi = np.real(np.conj(psi) @ E_phi_op @ psi)
            
            flux_expectations[f"E_x_{site}"] = exp_E_x
            flux_expectations[f"E_phi_{site}"] = exp_E_phi
          # Compare with classical values (ensure JSON serializable)
        classical_comparison = {}
        for site in range(self.hilbert.n_sites):
            E_x_cl = float(self.constraint.E_x_classical[site])
            E_phi_cl = float(self.constraint.E_phi_classical[site])
            
            E_x_quantum = float(flux_expectations[f"E_x_{site}"].real)
            E_phi_quantum = float(flux_expectations[f"E_phi_{site}"].real)
            
            classical_comparison[f"site_{site}"] = {
                "E_x_classical": E_x_cl,
                "E_x_quantum": E_x_quantum,
                "E_x_ratio": float(E_x_quantum / E_x_cl) if E_x_cl != 0 else float('inf'),
                "E_phi_classical": E_phi_cl,
                "E_phi_quantum": E_phi_quantum,
                "E_phi_ratio": float(E_phi_quantum / E_phi_cl) if E_phi_cl != 0 else float('inf')
            }
        
        # Convert all flux expectations to floats for JSON
        flux_json = {key: float(val.real) for key, val in flux_expectations.items()}
        
        validation_result = {
            "normalization": float(norm),
            "constraint_violation": float(constraint_violation),
            "eigenvalue": float(self.eigenvalues[state_index]),
            "flux_expectations": flux_json,
            "classical_comparison": classical_comparison
        }
        
        return validation_result
    
    def construct_coherent_state_guess(self) -> np.ndarray:
        """
        Construct initial guess based on classical data
        
        Creates semiclassical coherent state peaked on classical values
        """
        E_x_classical = self.constraint.E_x_classical
        E_phi_classical = self.constraint.E_phi_classical
        
        print("Constructing semiclassical coherent state...")
        coherent_state = self.hilbert.create_coherent_state(
            E_x_classical, E_phi_classical, width=2.0
        )
        
        # Check how well it satisfies constraint
        M = self.constraint.master_constraint_operator()
        violation = np.real(np.conj(coherent_state) @ M @ coherent_state)
        
        print(f"Coherent state constraint violation: {violation:.2e}")
        
        return coherent_state
    
    def save_physical_states(self, output_dir: str):
        """Save physical states and eigenvalues to files"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save eigenvalues
        eigenvals_file = os.path.join(output_dir, "eigenvalues.npy")
        np.save(eigenvals_file, self.eigenvalues)
        
        # Save eigenvectors (physical states)
        states_file = os.path.join(output_dir, "physical_states.npy") 
        np.save(states_file, self.physical_states)
        
        # Save validation info for ground state
        if len(self.physical_states) > 0:
            validation = self.validate_physical_state(0)
            validation_file = os.path.join(output_dir, "validation.json")
            
            with open(validation_file, 'w') as f:
                json.dump(validation, f, indent=2, default=str)
          # Create summary (ensure all values are JSON serializable)
        summary = {
            "solver_info": {
                "hilbert_dimension": int(self.hilbert.hilbert_dim),
                "n_eigenvalues": int(len(self.eigenvalues)),
                "n_physical_states": int(np.sum(self.eigenvalues < 1e-6)),
                "lowest_eigenvalue": float(self.eigenvalues[0]) if len(self.eigenvalues) > 0 else None
            },
            "files": {
                "eigenvalues": "eigenvalues.npy",
                "physical_states": "physical_states.npy", 
                "validation": "validation.json"
            }
        }
        
        summary_file = os.path.join(output_dir, "solver_summary.json")
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nPhysical states saved to {output_dir}/")
        print(f"  Eigenvalues: {eigenvals_file}")
        print(f"  States: {states_file}")
        print(f"  Summary: {summary_file}")


def load_reduced_variables(filename: str) -> Dict:
    """Load reduced variables from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)


def main():
    """Command line interface for constraint solver"""
    parser = argparse.ArgumentParser(description="Solve LQG constraint equation")
    parser.add_argument("--lattice", type=str, required=True,
                       help="JSON file with reduced variables")
    parser.add_argument("--out", type=str, default="quantum_outputs",
                       help="Output directory")
    parser.add_argument("--n-states", type=int, default=5,
                       help="Number of eigenvalues to compute")
    parser.add_argument("--mu-scheme", type=str, default="constant",
                       choices=["constant", "improved"],
                       help="mu-bar-scheme for holonomy regularization")
    parser.add_argument("--mu-range", type=int, nargs=2, default=[-3, 3],
                       help="Range for mu quantum numbers")
    parser.add_argument("--nu-range", type=int, nargs=2, default=[-3, 3],
                       help="Range for nu quantum numbers")
    parser.add_argument("--tolerance", type=float, default=1e-10,
                       help="Eigenvalue convergence tolerance")
    
    args = parser.parse_args()
    
    # Load reduced variables
    print(f"Loading reduced variables from {args.lattice}")
    reduced_data = load_reduced_variables(args.lattice)
    
    # Create Hilbert space
    config = load_lattice_from_reduced_variables(args.lattice)
    config.mu_range = tuple(args.mu_range)
    config.nu_range = tuple(args.nu_range)
    
    print(f"Building Hilbert space with quantum number ranges:")
    print(f"  mu range: {config.mu_range}")
    print(f"  nu range: {config.nu_range}")
    
    hilbert = MidisuperspaceHilbert(config)
    
    if hilbert.hilbert_dim > 10000:
        print(f"Warning: Large Hilbert space dimension {hilbert.hilbert_dim}")
        print("Consider reducing quantum number ranges for faster computation")
    
    # Build Hamiltonian constraint
    print(f"\nBuilding Hamiltonian constraint with {args.mu_scheme} mu-bar-scheme")
    constraint = HamiltonianConstraint(hilbert, reduced_data, args.mu_scheme)
    
    # Solve constraint
    print(f"\nSolving master constraint equation...")
    solver = ConstraintSolver(hilbert, constraint)
    
    eigenvals, eigenvecs = solver.solve_master_constraint(
        n_states=args.n_states, tolerance=args.tolerance
    )
    
    # Save results
    print(f"\nSaving results to {args.out}/")
    solver.save_physical_states(args.out)
    
    # Additional analysis for ground state
    if len(eigenvals) > 0:
        print(f"\nGround state analysis:")
        validation = solver.validate_physical_state(0)
        
        print(f"  Eigenvalue: lambda_0 = {validation['eigenvalue']:.2e}")
        print(f"  Constraint violation: {validation['constraint_violation']:.2e}")
        
        # Show classical vs quantum comparison        print(f"\n  Classical vs Quantum flux comparison:")
        for site in range(min(3, hilbert.n_sites)):  # Show first 3 sites
            comp = validation['classical_comparison'][f'site_{site}']
            print(f"    Site {site}: E^x = {comp['E_x_classical']:.3f} -> {comp['E_x_quantum']:.3f}")
            print(f"             E^phi = {comp['E_phi_classical']:.3f} -> {comp['E_phi_quantum']:.3f}")


if __name__ == "__main__":
    main()
