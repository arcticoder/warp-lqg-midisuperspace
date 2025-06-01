#!/usr/bin/env python3
"""
Quantum Stability Analysis (Task 6)

Analyzes stability of quantum warp bubble states:
- Builds discrete Sturm-Liouville operator from quantum metric
- Computes fluctuation spectrum {ω²ₙ} around background
- Identifies unstable modes (ω² < 0) and tachyonic instabilities
- Exports spectrum for comparison with classical analysis

Author: Loop Quantum Gravity Implementation
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import json
import os
from typing import Dict, List, Tuple, Optional

from expectation_values import ExpectationValueCalculator


class QuantumStabilityAnalyzer:
    """
    Analyzes quantum stability via linearized fluctuations
    
    Method:
    1. Extract quantum-corrected metric from ⟨E^a⟩ 
    2. Build discrete Sturm-Liouville operator for perturbations
    3. Solve eigenvalue problem: L̂ψₙ = ω²ₙ Ŵψₙ
    4. Identify stable (ω²>0) vs unstable (ω²<0) modes
    """
    
    def __init__(self, lattice_r: List[float], observables: Dict):
        self.lattice_r = lattice_r
        self.observables = observables
        self.n_points = len(lattice_r)
        
        # Extract quantum metric from flux expectation values
        self.extract_quantum_metric()
        
        print(f"Quantum stability analyzer initialized:")
        print(f"  Lattice points: {self.n_points}")
        print(f"  Radial range: {lattice_r[0]:.2e} -> {lattice_r[-1]:.2e}")
    
    def extract_quantum_metric(self):
        """
        Extract quantum-corrected metric functions from flux expectations
        
        E^x ↔ radial metric component g_rr = β²(r)
        E^φ ↔ angular metric components g_θθ = g_φφ = r²
        """
        E_x = self.observables["flux_expectations"]["E_x"]
        E_phi = self.observables["flux_expectations"]["E_phi"]
        
        self.alpha = []  # Lapse function α(r)
        self.beta = []   # Radial metric β(r)
        self.r_eff = []  # Effective radius from quantum geometry
        
        for i in range(self.n_points):
            r = self.lattice_r[i]
            
            # Extract effective radius from φ-flux
            if abs(E_phi[i]) > 1e-12:
                r_quantum = abs(E_phi[i]) / (4 * np.pi)
            else:
                r_quantum = r  # Fallback to coordinate radius
            
            # Extract radial metric from x-flux
            if abs(E_x[i]) > 1e-12 and r_quantum > 1e-12:
                beta_sq = abs(E_x[i]) / (4 * np.pi * r_quantum**2)
                beta_val = np.sqrt(max(beta_sq, 1e-12))
            else:
                beta_val = 1.0
            
            # Lapse function (assume static for now)
            alpha_val = 1.0
            
            self.alpha.append(alpha_val)
            self.beta.append(beta_val)
            self.r_eff.append(r_quantum)
        
        print(f"Quantum metric extracted:")
        print(f"  beta range: {min(self.beta):.3f} -> {max(self.beta):.3f}")
        print(f"  r_eff range: {min(self.r_eff):.2e} -> {max(self.r_eff):.2e}")
    
    def compute_potential_function(self) -> List[float]:
        """
        Compute effective potential V(r) for perturbation equation
        
        For spherical warp bubble perturbations:
        V(r) = curvature terms + matter contributions
        
        Sturm-Liouville form: -d/dr[p(r)dψ/dr] + q(r)ψ = ω²w(r)ψ
        """
        potential = []
        
        # Use stress-energy for potential terms
        T00 = self.observables["stress_energy_tensor"]["T00"]
        Trr = self.observables["stress_energy_tensor"]["Trr"]
        
        for i in range(self.n_points):
            r = self.r_eff[i]
            alpha = self.alpha[i]
            beta = self.beta[i]
            
            # Gravitational potential (curvature)
            if r > 1e-12:
                V_grav = 2.0 / (alpha**2 * r**2)  # l(l+1)/r² term
            else:
                V_grav = 0.0
            
            # Matter potential (from stress-energy)
            rho = T00[i]  # Energy density
            p_r = Trr[i]  # Radial pressure
            
            # Effective potential including matter
            V_matter = 4 * np.pi * (rho + 3*p_r) / alpha**2
            
            V_total = V_grav + V_matter
            potential.append(V_total)
        
        return potential
    
    def build_sturm_liouville_operator(self) -> Tuple[sp.csr_matrix, sp.csr_matrix]:
        """
        Build discrete Sturm-Liouville operator matrices
        
        Equation: -d/dr[p(r)dψ/dr] + q(r)ψ = ω²w(r)ψ
        
        Discretized: L ψ = ω² W ψ (generalized eigenvalue problem)
        
        Returns: (L_matrix, W_matrix)
        """
        # Coefficient functions
        p_coeff = []  # p(r) = α(r)/β(r) × r²
        q_coeff = self.compute_potential_function()  # q(r) = V(r) 
        w_coeff = []  # w(r) = r² (weight function)
        
        for i in range(self.n_points):
            r = self.r_eff[i]
            alpha = self.alpha[i]
            beta = self.beta[i]
            
            p_val = alpha * r**2 / beta if beta > 1e-12 else 0.0
            w_val = r**2
            
            p_coeff.append(p_val)
            w_coeff.append(w_val)
        
        # Build finite difference matrices
        L_matrix = sp.lil_matrix((self.n_points, self.n_points))
        W_matrix = sp.lil_matrix((self.n_points, self.n_points))
        
        # Interior points: second-order finite differences
        for i in range(1, self.n_points - 1):
            dr_minus = self.lattice_r[i] - self.lattice_r[i-1]
            dr_plus = self.lattice_r[i+1] - self.lattice_r[i]
            dr_avg = 0.5 * (dr_minus + dr_plus)
            
            # p-coefficient at interfaces (harmonic mean)
            p_left = 2 * p_coeff[i-1] * p_coeff[i] / (p_coeff[i-1] + p_coeff[i] + 1e-12)
            p_right = 2 * p_coeff[i] * p_coeff[i+1] / (p_coeff[i] + p_coeff[i+1] + 1e-12)
            
            # Finite difference coefficients
            coeff_left = p_left / (dr_minus * dr_avg)
            coeff_center = -(p_left / (dr_minus * dr_avg) + p_right / (dr_plus * dr_avg))
            coeff_right = p_right / (dr_plus * dr_avg)
            
            # L matrix: -d/dr[p dψ/dr] + qψ
            L_matrix[i, i-1] = -coeff_left
            L_matrix[i, i] = -coeff_center + q_coeff[i]
            L_matrix[i, i+1] = -coeff_right
            
            # W matrix: weight function
            W_matrix[i, i] = w_coeff[i]
        
        # Boundary conditions (Dirichlet: ψ=0 at boundaries)
        L_matrix[0, 0] = 1.0
        L_matrix[-1, -1] = 1.0
        W_matrix[0, 0] = 1.0
        W_matrix[-1, -1] = 1.0
        
        return L_matrix.tocsr(), W_matrix.tocsr()
    
    def solve_eigenvalue_problem(self, n_modes: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve generalized eigenvalue problem: L ψₙ = ω²ₙ W ψₙ
        
        Returns: (eigenvalues ω²ₙ, eigenvectors ψₙ)
        """
        print("Building Sturm-Liouville operator...")
        L, W = self.build_sturm_liouville_operator()
        
        print(f"L matrix: {L.shape}, nnz = {L.nnz}")
        print(f"W matrix: {W.shape}, nnz = {W.nnz}")
        
        # Solve generalized eigenvalue problem
        print(f"Solving for {n_modes} eigenvalues...")
        
        try:
            if self.n_points <= 50:
                # Small matrices: use dense solver
                L_dense = L.toarray()
                W_dense = W.toarray()
                
                eigenvals, eigenvecs = np.linalg.eigh(L_dense, W_dense)
                
                # Sort by eigenvalue
                idx = np.argsort(eigenvals)
                eigenvals = eigenvals[idx][:n_modes]
                eigenvecs = eigenvecs[:, idx][:, :n_modes]
                
            else:
                # Large matrices: use sparse solver
                eigenvals, eigenvecs = spla.eigsh(
                    L, M=W, k=min(n_modes, self.n_points-2),
                    which='SM',  # Smallest magnitude
                    tol=1e-10
                )
                
                # Sort by eigenvalue
                idx = np.argsort(eigenvals)
                eigenvals = eigenvals[idx]
                eigenvecs = eigenvecs[:, idx]
        
        except Exception as e:
            print(f"Eigenvalue solver error: {e}")
            print("Trying alternative approach...")
            
            # Fallback: regularized problem
            L_reg = L + 1e-12 * sp.eye(L.shape[0])
            eigenvals, eigenvecs = spla.eigsh(
                L_reg, M=W, k=min(n_modes, self.n_points-2),
                which='SM', tol=1e-8
            )
            
            idx = np.argsort(eigenvals)
            eigenvals = eigenvals[idx]
            eigenvecs = eigenvecs[:, idx]
        
        return eigenvals, eigenvecs
    
    def analyze_stability(self, eigenvalues: np.ndarray) -> Dict:
        """
        Analyze stability based on eigenvalue spectrum
        
        ω²ₙ > 0: stable mode
        ω²ₙ < 0: unstable (tachyonic) mode
        ω²ₙ ≈ 0: marginally stable/gauge mode
        """
        n_total = len(eigenvalues)
        n_negative = np.sum(eigenvalues < -1e-10)
        n_zero = np.sum(np.abs(eigenvalues) < 1e-10)
        n_positive = np.sum(eigenvalues > 1e-10)
        
        # Find most unstable mode
        most_unstable_idx = np.argmin(eigenvalues)
        most_unstable_omega2 = eigenvalues[most_unstable_idx]
          # Growth rate for unstable modes
        if most_unstable_omega2 < 0:
            growth_rate = np.sqrt(-most_unstable_omega2)
        else:
            growth_rate = 0.0
        
        stability_analysis = {
            "total_modes": int(n_total),
            "stable_modes": int(n_positive),
            "unstable_modes": int(n_negative),
            "marginal_modes": int(n_zero),
            "most_unstable_eigenvalue": float(most_unstable_omega2),
            "growth_rate": float(growth_rate),
            "stability_verdict": "STABLE" if n_negative == 0 else "UNSTABLE"
        }
        
        print(f"\nStability analysis:")
        print(f"  Total modes: {n_total}")
        print(f"  Stable (omega^2 > 0): {n_positive}")
        print(f"  Unstable (omega^2 < 0): {n_negative}")
        print(f"  Marginal (omega^2 ~ 0): {n_zero}")
        print(f"  Most unstable: omega^2 = {most_unstable_omega2:.3e}")
        print(f"  Verdict: {stability_analysis['stability_verdict']}")
        
        return stability_analysis
    
    def save_spectrum(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray,
                     output_dir: str) -> str:
        """Save quantum fluctuation spectrum to JSON file"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Stability analysis
        stability = self.analyze_stability(eigenvalues)
          # Spectrum data (ensure JSON serializable)
        spectrum_data = {
            "quantum_spectrum": {
                "eigenvalues_omega_squared": [float(x) for x in eigenvalues],
                "frequencies": [float(x) for x in np.sqrt(np.abs(eigenvalues))],
                "lattice_r": [float(x) for x in self.lattice_r],
                "n_modes": int(len(eigenvalues))
            },
            "stability_analysis": stability,
            "quantum_metric": {
                "alpha_lapse": [float(x) for x in self.alpha],
                "beta_radial": [float(x) for x in self.beta],
                "r_effective": [float(x) for x in self.r_eff]
            },
            "metadata": {
                "description": "Quantum fluctuation spectrum from LQG warp bubble",
                "method": "discrete_sturm_liouville",
                "units": "Planck_units"
            }
        }
        
        # Save main spectrum file
        spectrum_file = os.path.join(output_dir, "quantum_spectrum.json")
        with open(spectrum_file, 'w') as f:
            json.dump(spectrum_data, f, indent=2)
        
        # Save eigenvectors separately (can be large)
        eigenvecs_file = os.path.join(output_dir, "quantum_eigenvectors.npy")
        np.save(eigenvecs_file, eigenvectors)
        
        print(f"\nQuantum spectrum saved:")
        print(f"  Spectrum: {spectrum_file}")
        print(f"  Eigenvectors: {eigenvecs_file}")
        
        return spectrum_file


def main():
    """Command line interface for quantum stability analysis"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze quantum warp bubble stability")
    parser.add_argument("--observables", type=str, required=True,
                       help="JSON file with expectation values")
    parser.add_argument("--out", type=str, default="quantum_outputs",
                       help="Output directory")
    parser.add_argument("--n-modes", type=int, default=20,
                       help="Number of eigenmodes to compute")
    
    args = parser.parse_args()
    
    # Load observables
    print(f"Loading observables from {args.observables}")
    with open(args.observables, 'r') as f:
        observables = json.load(f)
    
    lattice_r = observables["lattice_r"]
    
    # Analyze stability
    analyzer = QuantumStabilityAnalyzer(lattice_r, observables)
    
    print(f"\nSolving for quantum fluctuation spectrum...")
    eigenvals, eigenvecs = analyzer.solve_eigenvalue_problem(args.n_modes)
    
    # Save results
    spectrum_file = analyzer.save_spectrum(eigenvals, eigenvecs, args.out)
    
    # Show sample eigenvalues
    print(f"\nSample eigenvalues omega^2_n:")
    n_show = min(10, len(eigenvals))
    for i in range(n_show):
        omega2 = eigenvals[i]
        omega = np.sqrt(abs(omega2))
        stability = "stable" if omega2 > 0 else "unstable"
        print(f"  Mode {i}: omega^2 = {omega2:.3e}, omega = {omega:.3e} ({stability})")


if __name__ == "__main__":
    main()
