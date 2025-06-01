#!/usr/bin/env python3
"""
Hamiltonian Constraint (Task 3)

Implements the quantum Hamiltonian constraint for midisuperspace LQG:
- Gravitational constraint Ĥ_grav with holonomy corrections
- Matter constraint Ĥ_matter for phantom scalar field  
- Regularization of inverse triad operators
- μ̄-scheme for loop size determination

Author: Loop Quantum Gravity Implementation
"""

import numpy as np
import scipy.sparse as sp
from typing import List, Dict, Tuple, Optional
import json
from kinematical_hilbert import MidisuperspaceHilbert, BasisState


class HamiltonianConstraint:
    """
    Quantum Hamiltonian constraint: Ĥ_grav + Ĥ_matter
    
    Physical content:
    - Einstein's equations → constraint Ĥ|Ψ⟩ = 0
    - Holonomy corrections regularize curvature singularities
    - Discrete loop area ≈ μ̄² ℓ_Pl² (Immirzi parameter dependence)
    """
    
    def __init__(self, hilbert: MidisuperspaceHilbert, 
                 reduced_data: Dict, mu_bar_scheme: str = "constant"):
        self.hilbert = hilbert
        self.reduced_data = reduced_data
        self.mu_bar_scheme = mu_bar_scheme
        
        # Physical parameters
        self.gamma = hilbert.gamma
        self.l_planck = 1.0  # Planck length in Planck units
        self.G = 1.0         # Newton constant in Planck units
        
        # Extract lattice and classical data
        self.lattice_r = reduced_data["lattice_r"]
        self.n_sites = len(self.lattice_r)
        
        # Classical backgrounds for regularization
        self.E_x_classical = reduced_data["E_classical"]["E^x"]
        self.E_phi_classical = reduced_data["E_classical"]["E^phi"]
        self.scalar_classical = reduced_data["exotic_matter_profile"]["scalar_field"]
        
        print(f"Hamiltonian constraint initialized:")
        print(f"  Lattice sites: {self.n_sites}")
        print(f"  mu-bar-scheme: {mu_bar_scheme}")
        print(f"  Hilbert dimension: {hilbert.hilbert_dim}")
    
    def compute_mu_bar(self, site: int) -> float:
        """
        Compute μ̄ parameter determining loop size
        
        Physical requirement: loop area ≈ Planck area
        Δ = μ̄² γ ℓ_Pl² ≈ ℓ_Pl²  →  μ̄ ≈ 1/√γ
        """
        if self.mu_bar_scheme == "constant":
            return 1.0 / np.sqrt(self.gamma)
        elif self.mu_bar_scheme == "improved":
            # Area-dependent: μ̄ = min(μ_site, μ_0) for improved dynamics
            mu_site = abs(self.hilbert.basis_states[0].mu[site]) + 1  # Avoid zero
            mu_0 = 1.0 / np.sqrt(self.gamma)
            return min(mu_site, mu_0)
        else:
            raise ValueError(f"Unknown mu-bar-scheme: {self.mu_bar_scheme}")
    
    def inverse_triad_operator(self, site: int, power: float = 0.5) -> sp.csr_matrix:
        """
        Regularized inverse triad operator |E|^(-p)
        
        Uses Thiemann's regularization to avoid singularities:
        |E|^(-p) → sign(E) |E|^(-p) when E ≠ 0, else 0
        
        Matrix elements: ⟨μ',ν'||E^x|^(-p)|μ,ν⟩
        """
        matrix = sp.lil_matrix((self.hilbert.hilbert_dim, self.hilbert.hilbert_dim))
        
        for i, state in enumerate(self.hilbert.basis_states):
            mu_i = state.mu[site]
            
            if mu_i != 0:  # Regularization: avoid 1/0
                E_eigenval = self.hilbert.flux_E_x_eigenvalue(mu_i, site)
                inv_E = np.sign(E_eigenval) * abs(E_eigenval)**(-power)
                matrix[i, i] = inv_E
            # else: matrix[i,i] = 0 (regularized)
        
        return matrix.tocsr()
    
    def sqrt_inverse_volume_operator(self, site: int) -> sp.csr_matrix:
        """
        Operator 1/√|E^x E^φ| with Thiemann regularization
        
        Physical role: appears in Hamiltonian constraint density
        """
        matrix = sp.lil_matrix((self.hilbert.hilbert_dim, self.hilbert.hilbert_dim))
        
        for i, state in enumerate(self.hilbert.basis_states):
            mu_i = state.mu[site]
            nu_i = state.nu[site]
            
            if mu_i != 0 and nu_i != 0:
                E_x = self.hilbert.flux_E_x_eigenvalue(mu_i, site)
                E_phi = self.hilbert.flux_E_phi_eigenvalue(nu_i, site)
                
                volume = abs(E_x * E_phi)
                if volume > 0:
                    inv_sqrt_vol = 1.0 / np.sqrt(volume)
                    matrix[i, i] = inv_sqrt_vol
        
        return matrix.tocsr()
    
    def holonomy_curvature_operator(self, site_i: int, site_j: int) -> sp.csr_matrix:
        """
        Discretized curvature via holonomy loops
        
        Classical: F_{ab} = ∂_a A_b - ∂_b A_a + [A_a, A_b]
        Quantum: h_□ = h_x(i) h_φ(j) h_x^†(i) h_φ^†(j) ≈ 1 + i Area·F + O(Area²)
        
        For midisuperspace: approximate as finite difference
        """
        if site_j != site_i + 1:
            # Only nearest-neighbor interactions in 1D lattice
            return sp.csr_matrix((self.hilbert.hilbert_dim, self.hilbert.hilbert_dim))
        
        mu_bar_i = self.compute_mu_bar(site_i)
        mu_bar_j = self.compute_mu_bar(site_j)
        
        # Holonomy operators
        h_x_i = self.hilbert.holonomy_operator_x(site_i, mu_bar_i)
        h_phi_j = self.hilbert.holonomy_operator_phi(site_j, mu_bar_j)
        
        # Commutator approximating curvature: [h_x, h_φ] ≈ i·Area·F
        commutator = h_x_i @ h_phi_j - h_phi_j @ h_x_i
        
        # Scale by loop area
        loop_area = mu_bar_i * mu_bar_j * self.gamma * self.l_planck**2
        
        return commutator / loop_area  # Remove area factor to get curvature
    
    def gravitational_hamiltonian(self) -> sp.csr_matrix:
        """
        Gravitational part of Hamiltonian constraint
        
        H_grav = ∫ d³x N/√|E| [E^a E^b F_ab + constraint terms]
        
        In spherical symmetry with holonomy regularization:
        H_grav ≈ Σ_i [regularized curvature terms]
        """
        H_grav = sp.csr_matrix((self.hilbert.hilbert_dim, self.hilbert.hilbert_dim))
        
        for i in range(self.n_sites - 1):
            # Inverse volume factor
            inv_sqrt_vol = self.sqrt_inverse_volume_operator(i)
            
            # Holonomy curvature between sites i and i+1
            curvature = self.holonomy_curvature_operator(i, i + 1)
            
            # Combine: (1/√V) × Curvature
            term = inv_sqrt_vol @ curvature
            
            # Grid spacing factor
            dr = self.lattice_r[i + 1] - self.lattice_r[i]
            
            H_grav += dr * term
        
        return H_grav
    
    def scalar_kinetic_operator(self) -> sp.csr_matrix:
        """
        Kinetic term for phantom scalar field
        
        For phantom field: T_kinetic = -(1/2) g^μν ∂_μφ ∂_νφ (wrong sign!)
        
        Discretized: π_φ²/(2√|E|) where π_φ is conjugate momentum
        """
        H_kinetic = sp.csr_matrix((self.hilbert.hilbert_dim, self.hilbert.hilbert_dim))
        
        for i in range(self.n_sites):
            # Classical momentum (for semiclassical approximation)
            pi_phi_cl = self.reduced_data["exotic_matter_profile"]["scalar_momentum"][i]
            
            if abs(pi_phi_cl) > 1e-12:  # Non-zero momentum
                inv_sqrt_vol = self.sqrt_inverse_volume_operator(i)
                
                # Phantom field: negative kinetic energy
                kinetic_term = -0.5 * pi_phi_cl**2 * inv_sqrt_vol
                
                H_kinetic += kinetic_term
        
        return H_kinetic
    
    def scalar_potential_operator(self) -> sp.csr_matrix:
        """
        Potential term for scalar field: V(φ) = (1/2)m²φ²
        """
        H_potential = sp.csr_matrix((self.hilbert.hilbert_dim, self.hilbert.hilbert_dim))
        
        for i in range(self.n_sites):
            phi_cl = self.scalar_classical[i]
            mass_sq = self.reduced_data.get("scalar_mass_squared", 1e-20)
            
            if abs(phi_cl) > 1e-12:
                inv_sqrt_vol = self.sqrt_inverse_volume_operator(i)
                
                # Potential contribution: (m²φ²/2) / √|E|
                potential_term = 0.5 * mass_sq * phi_cl**2 * inv_sqrt_vol
                
                H_potential += potential_term
        
        return H_potential
    
    def matter_hamiltonian(self) -> sp.csr_matrix:
        """
        Complete matter Hamiltonian: H_matter = H_kinetic + H_potential
        """
        H_kinetic = self.scalar_kinetic_operator()
        H_potential = self.scalar_potential_operator()
        
        return H_kinetic + H_potential
    
    def total_hamiltonian(self) -> sp.csr_matrix:
        """
        Total Hamiltonian constraint: Ĥ = Ĥ_grav + Ĥ_matter
        
        Physical states satisfy: Ĥ|Ψ⟩ = 0
        """
        print("Building gravitational Hamiltonian...")
        H_grav = self.gravitational_hamiltonian()
        
        print("Building matter Hamiltonian...")  
        H_matter = self.matter_hamiltonian()
        
        print("Combining total Hamiltonian...")
        H_total = H_grav + H_matter
        
        print(f"Total Hamiltonian: {H_total.shape}, nnz = {H_total.nnz}")
        return H_total
    
    def master_constraint_operator(self) -> sp.csr_matrix:
        """
        Master constraint: M̂ = Ĥ†Ĥ
        
        Physical states are zero modes: M̂|Ψ⟩ = 0
        Advantages: Hermitian, positive semi-definite
        """
        H = self.total_hamiltonian()
        H_dagger = H.getH()  # Hermitian conjugate
        
        print("Computing master constraint H†H...")
        M = H_dagger @ H
        
        print(f"Master constraint: {M.shape}, nnz = {M.nnz}")
        return M
    
    def save_hamiltonian_info(self, filename: str):
        """Save Hamiltonian constraint information"""
        H = self.total_hamiltonian()
        
        info = {
            "hamiltonian_info": {
                "matrix_dimension": H.shape[0],
                "non_zero_elements": int(H.nnz),
                "sparsity": float(H.nnz) / (H.shape[0]**2),
                "mu_bar_scheme": self.mu_bar_scheme,
                "immirzi_parameter": self.gamma
            },
            "lattice_info": {
                "n_sites": self.n_sites,
                "r_range": [min(self.lattice_r), max(self.lattice_r)],
                "lattice_spacing": np.diff(self.lattice_r).tolist()
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(info, f, indent=2)
        
        print(f"Hamiltonian info saved to {filename}")


def load_reduced_variables(filename: str) -> Dict:
    """Load reduced variables from JSON file"""
    with open(filename, 'r') as f:
        return json.load(f)


def main():
    """Test Hamiltonian constraint construction"""
    from kinematical_hilbert import load_lattice_from_reduced_variables
    
    # Create test reduced variables
    test_data = {
        "lattice_r": [1e-15, 2e-15, 5e-15],
        "E_classical": {
            "E^x": [1.0, 1.5, 2.0],
            "E^phi": [0.5, 0.7, 1.0]
        },
        "K_classical": {
            "K_x": [0.1, 0.15, 0.2],
            "K_phi": [0.05, 0.08, 0.1]
        },
        "exotic_matter_profile": {
            "scalar_field": [1.0, 0.8, 0.5],
            "scalar_momentum": [0.1, 0.05, 0.02]
        }
    }
    
    # Save test data
    with open("test_reduced.json", 'w') as f:
        json.dump(test_data, f)
    
    # Load Hilbert space
    config = load_lattice_from_reduced_variables("test_reduced.json")
    config.mu_range = (-1, 1)  # Small for testing
    config.nu_range = (-1, 1)
    
    hilbert = MidisuperspaceHilbert(config)
    
    # Build Hamiltonian
    constraint = HamiltonianConstraint(hilbert, test_data, "constant")
    
    # Test individual components
    print("\nTesting Hamiltonian components:")
    
    H_grav = constraint.gravitational_hamiltonian()
    print(f"H_grav: {H_grav.shape}, nnz = {H_grav.nnz}")
    
    H_matter = constraint.matter_hamiltonian()
    print(f"H_matter: {H_matter.shape}, nnz = {H_matter.nnz}")
    
    H_total = constraint.total_hamiltonian()
    print(f"H_total: {H_total.shape}, nnz = {H_total.nnz}")
    
    M = constraint.master_constraint_operator()
    print(f"Master constraint: {M.shape}, nnz = {M.nnz}")
    
    # Save info
    constraint.save_hamiltonian_info("hamiltonian_info.json")


if __name__ == "__main__":
    main()
