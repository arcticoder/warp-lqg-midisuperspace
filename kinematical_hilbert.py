#!/usr/bin/env python3
"""
Kinematical Hilbert Space (Task 2)

Constructs the LQG Hilbert space for midisuperspace models:
- Discrete lattice with flux operators E^x(r_i), E^φ(r_i)  
- Holonomy operators h_x(r_i), h_φ(r_i)
- Basis states |μ_i, ν_i⟩ with quantum numbers
- Spin network truncation and gauge invariance

Author: Loop Quantum Gravity Implementation
"""

import numpy as np
import scipy.sparse as sp
from typing import List, Tuple, Dict, Optional, Union
from dataclasses import dataclass
import json


@dataclass
class LatticeConfig:
    """Configuration for discrete radial lattice"""
    r_points: List[float]
    mu_range: Tuple[int, int]  # Range for radial quantum numbers
    nu_range: Tuple[int, int]  # Range for angular quantum numbers
    

class BasisState:
    """Individual basis state |μ₁,ν₁; μ₂,ν₂; ...; μₙ,νₙ⟩"""
    
    def __init__(self, mu_values: List[int], nu_values: List[int]):
        if len(mu_values) != len(nu_values):
            raise ValueError("mu and nu lists must have same length")
        
        self.mu = tuple(mu_values)  # Radial quantum numbers
        self.nu = tuple(nu_values)  # Angular quantum numbers
        self.n_sites = len(mu_values)
        
    def __repr__(self):
        pairs = [f"mu{i}={self.mu[i]},nu{i}={self.nu[i]}" for i in range(self.n_sites)]
        return f"|{'; '.join(pairs)}⟩"
    
    def __eq__(self, other):
        return isinstance(other, BasisState) and self.mu == other.mu and self.nu == other.nu
    
    def __hash__(self):
        return hash((self.mu, self.nu))


class MidisuperspaceHilbert:
    """
    Kinematical Hilbert space for spherically symmetric LQG
    
    Physical interpretation:
    - Each lattice site r_i has quantum numbers (μ_i, ν_i) 
    - μ_i: radial flux quantum number → E^x eigenvalue
    - ν_i: angular flux quantum number → E^φ eigenvalue
    - Planck-scale discretization of area operators
    """
    
    def __init__(self, config: LatticeConfig):
        self.config = config
        self.n_sites = len(config.r_points)
        
        # Physical constants (Planck units)
        self.gamma = 0.2375  # Immirzi parameter
        self.l_planck_sq = 1.0  # ℓ_Pl² in Planck units
        
        # Build complete basis
        self.basis_states = self._generate_basis()
        self.basis_dict = {state: i for i, state in enumerate(self.basis_states)}
        self.hilbert_dim = len(self.basis_states)        
        print(f"Kinematical Hilbert space constructed:")
        print(f"  Lattice sites: {self.n_sites}")
        print(f"  mu range: {config.mu_range}")  
        print(f"  nu range: {config.nu_range}")
        print(f"  Total dimension: {self.hilbert_dim}")
        
    def _generate_basis(self) -> List[BasisState]:
        """Generate all basis states within quantum number ranges"""
        mu_min, mu_max = self.config.mu_range
        nu_min, nu_max = self.config.nu_range
        
        # All possible mu and nu values
        mu_vals = list(range(mu_min, mu_max + 1))
        nu_vals = list(range(nu_min, nu_max + 1))
        
        basis = []
        
        # Cartesian product over all lattice sites
        def recursive_build(site_idx, current_mu, current_nu):
            if site_idx == self.n_sites:
                basis.append(BasisState(current_mu.copy(), current_nu.copy()))
                return
                
            for mu in mu_vals:
                for nu in nu_vals:
                    current_mu.append(mu)
                    current_nu.append(nu)
                    recursive_build(site_idx + 1, current_mu, current_nu)
                    current_mu.pop()
                    current_nu.pop()
        
        recursive_build(0, [], [])
        return basis
    
    def state_index(self, state: BasisState) -> int:
        """Get index of basis state in Hilbert space"""
        return self.basis_dict[state]
    
    def index_to_state(self, index: int) -> BasisState:
        """Get basis state from index"""
        return self.basis_states[index]
    
    def flux_E_x_eigenvalue(self, mu: int, site: int) -> float:
        """
        Eigenvalue of radial flux operator E^x(r_i)
        
        E^x |μ,ν⟩ = γ ℓ_Pl² μ |μ,ν⟩
        """
        return self.gamma * self.l_planck_sq * mu
    
    def flux_E_phi_eigenvalue(self, nu: int, site: int) -> float:
        """
        Eigenvalue of angular flux operator E^φ(r_i)
        
        E^φ |μ,ν⟩ = γ ℓ_Pl² ν |μ,ν⟩  
        """
        return self.gamma * self.l_planck_sq * nu
    
    def flux_E_x_operator(self, site: int) -> sp.csr_matrix:
        """
        Matrix representation of flux operator E^x(r_i)
        
        Diagonal operator: ⟨μ',ν'|E^x(r_i)|μ,ν⟩ = γℓ_Pl² μ_i δ_{μ',μ} δ_{ν',ν}
        """
        diagonal = np.zeros(self.hilbert_dim)
        
        for i, state in enumerate(self.basis_states):
            mu_i = state.mu[site]
            diagonal[i] = self.flux_E_x_eigenvalue(mu_i, site)
            
        return sp.diags(diagonal, format='csr')
    
    def flux_E_phi_operator(self, site: int) -> sp.csr_matrix:
        """Matrix representation of flux operator E^φ(r_i)"""
        diagonal = np.zeros(self.hilbert_dim)
        
        for i, state in enumerate(self.basis_states):
            nu_i = state.nu[site]
            diagonal[i] = self.flux_E_phi_eigenvalue(nu_i, site)
            
        return sp.diags(diagonal, format='csr')
    
    def holonomy_operator_x(self, site: int, mu_bar: float) -> sp.csr_matrix:
        """
        Radial holonomy operator h_x = exp(i μ̄ K_x)
        
        Acts as: h_x |μ,ν⟩ = |μ+1,ν⟩ (raising operator)
        Physical: parallel transport around elementary loop
        """
        row_indices = []
        col_indices = []
        data = []
        
        for i, state in enumerate(self.basis_states):
            # Create new state with μ_site → μ_site + 1
            new_mu = list(state.mu)
            if new_mu[site] < self.config.mu_range[1]:  # Check bounds
                new_mu[site] += 1
                new_state = BasisState(new_mu, list(state.nu))
                
                if new_state in self.basis_dict:
                    j = self.basis_dict[new_state]
                    row_indices.append(j)
                    col_indices.append(i)
                    
                    # Matrix element: holonomy amplitude
                    amplitude = 1.0  # Simplified: could include μ̄ dependence
                    data.append(amplitude)
        
        return sp.csr_matrix((data, (row_indices, col_indices)), 
                           shape=(self.hilbert_dim, self.hilbert_dim))
    
    def holonomy_operator_phi(self, site: int, nu_bar: float) -> sp.csr_matrix:
        """Angular holonomy operator h_φ = exp(i ν̄ K_φ)"""
        row_indices = []
        col_indices = []
        data = []
        
        for i, state in enumerate(self.basis_states):
            # Create new state with ν_site → ν_site + 1
            new_nu = list(state.nu)
            if new_nu[site] < self.config.nu_range[1]:  # Check bounds
                new_nu[site] += 1
                new_state = BasisState(list(state.mu), new_nu)
                
                if new_state in self.basis_dict:
                    j = self.basis_dict[new_state]
                    row_indices.append(j)
                    col_indices.append(i)
                    data.append(1.0)
        
        return sp.csr_matrix((data, (row_indices, col_indices)),
                           shape=(self.hilbert_dim, self.hilbert_dim))
    
    def volume_operator(self, site: int) -> sp.csr_matrix:
        """
        Volume operator V̂(r_i) = √|E^x E^φ|(r_i)
        
        Fundamental area operator in LQG: quantized with discrete spectrum
        """
        V_matrix = sp.lil_matrix((self.hilbert_dim, self.hilbert_dim))
        
        for i, state in enumerate(self.basis_states):
            mu_i = state.mu[site]
            nu_i = state.nu[site]
            
            # Volume eigenvalue: √|μ ν| in Planck units
            if mu_i != 0 and nu_i != 0:
                volume_eigenval = np.sqrt(abs(mu_i * nu_i)) * self.l_planck_sq**(3/2)
                V_matrix[i, i] = volume_eigenval
        
        return V_matrix.tocsr()
    
    def create_coherent_state(self, classical_E_x: List[float], 
                            classical_E_phi: List[float],
                            width: float = 1.0) -> np.ndarray:
        """
        Construct semiclassical coherent state peaked on classical values
        
        |Ψ_coherent⟩ = Σ_μ,ν exp(-|μ-μ_cl|²/2σ² - |ν-ν_cl|²/2σ²) |μ,ν⟩
        """
        if len(classical_E_x) != self.n_sites or len(classical_E_phi) != self.n_sites:
            raise ValueError("Classical arrays must match lattice size")
        
        # Convert classical values to expected quantum numbers
        classical_mu = [E_x / (self.gamma * self.l_planck_sq) for E_x in classical_E_x]
        classical_nu = [E_phi / (self.gamma * self.l_planck_sq) for E_phi in classical_E_phi]
        
        # Build coherent state amplitudes
        amplitudes = np.zeros(self.hilbert_dim, dtype=complex)
        
        for i, state in enumerate(self.basis_states):
            # Gaussian weight centered on classical values
            weight = 1.0
            
            for site in range(self.n_sites):
                mu_diff = state.mu[site] - classical_mu[site]
                nu_diff = state.nu[site] - classical_nu[site]
                
                weight *= np.exp(-(mu_diff**2 + nu_diff**2) / (2 * width**2))
            
            amplitudes[i] = weight
        
        # Normalize
        norm = np.linalg.norm(amplitudes)
        if norm > 0:
            amplitudes /= norm
            
        return amplitudes
    
    def expectation_value(self, state: np.ndarray, operator: sp.csr_matrix) -> complex:
        """Compute ⟨Ψ|Ô|Ψ⟩ for given state and operator"""
        return np.conj(state) @ operator @ state
    
    def save_hilbert_info(self, filename: str):
        """Save Hilbert space configuration to JSON"""
        info = {
            "lattice_sites": self.n_sites,
            "r_points": self.config.r_points,
            "mu_range": self.config.mu_range,
            "nu_range": self.config.nu_range,
            "hilbert_dimension": self.hilbert_dim,
            "immirzi_parameter": self.gamma,
            "sample_states": [str(self.basis_states[i]) for i in range(min(5, len(self.basis_states)))]
        }
        
        with open(filename, 'w') as f:
            json.dump(info, f, indent=2)


def load_lattice_from_reduced_variables(filename: str) -> LatticeConfig:
    """Load lattice configuration from reduced variables JSON"""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    r_points = data["lattice_r"]
    
    # Default quantum number ranges (can be tuned)
    mu_range = (-5, 5)  # Radial flux quantum numbers
    nu_range = (-5, 5)  # Angular flux quantum numbers
    
    return LatticeConfig(r_points, mu_range, nu_range)


def main():
    """Test kinematical Hilbert space construction"""
    # Example lattice
    r_points = [1e-15, 2e-15, 5e-15, 1e-14, 2e-14]
    config = LatticeConfig(r_points, (-2, 2), (-2, 2))
    
    # Build Hilbert space
    hilbert = MidisuperspaceHilbert(config)
    
    # Test operators
    E_x_0 = hilbert.flux_E_x_operator(0)
    E_phi_1 = hilbert.flux_E_phi_operator(1)
    
    print(f"\nOperator E^x(r_0) has shape: {E_x_0.shape}")
    print(f"Operator E^φ(r_1) has shape: {E_phi_1.shape}")
    
    # Create coherent state
    classical_E_x = [1.0, 1.5, 2.0, 2.5, 3.0]
    classical_E_phi = [0.5, 0.7, 1.0, 1.2, 1.5]
    
    coherent_state = hilbert.create_coherent_state(classical_E_x, classical_E_phi)
    
    # Compute expectation values
    exp_E_x_0 = hilbert.expectation_value(coherent_state, E_x_0)
    exp_E_phi_1 = hilbert.expectation_value(coherent_state, E_phi_1)
    
    print(f"\n⟨E^x(r_0)⟩ = {exp_E_x_0:.3f}")
    print(f"⟨E^φ(r_1)⟩ = {exp_E_phi_1:.3f}")
    print(f"Classical target E^x(r_0) = {classical_E_x[0]}")
    print(f"Classical target E^φ(r_1) = {classical_E_phi[1]}")
    
    # Save info
    hilbert.save_hilbert_info("hilbert_space_info.json")


if __name__ == "__main__":
    main()
