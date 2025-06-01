#!/usr/bin/env python3
"""
Expectation Values (Task 5)

Computes quantum expectation values from physical states:
- Flux operators ⟨E^x⟩, ⟨E^φ⟩ → quantum-corrected metric
- Stress-energy tensor ⟨T^μν⟩ for matter fields
- Curvature observables ⟨R⟩, ⟨R_μν⟩
- Exports results in JSON format for warp-framework

Author: Loop Quantum Gravity Implementation  
"""

import numpy as np
import scipy.sparse as sp
import json
import os
from typing import Dict, List, Tuple, Optional

from kinematical_hilbert import MidisuperspaceHilbert
from hamiltonian_constraint import HamiltonianConstraint


class ExpectationValueCalculator:
    """
    Computes physical observables from quantum states
    
    Key outputs:
    - ⟨E^a(r_i)⟩: quantum-corrected triad fluxes
    - ⟨T^μν(r_i)⟩: stress-energy tensor components
    - ⟨R(r_i)⟩: curvature scalar
    """
    
    def __init__(self, hilbert: MidisuperspaceHilbert, 
                 constraint: HamiltonianConstraint,
                 reduced_data: Dict):
        self.hilbert = hilbert
        self.constraint = constraint
        self.reduced_data = reduced_data
        
        self.lattice_r = reduced_data["lattice_r"]
        self.n_sites = len(self.lattice_r)
        
        # Physical constants (Planck units)
        self.c = 1.0
        self.G = 1.0
        self.hbar = 1.0
        
    def compute_flux_expectations(self, state: np.ndarray) -> Dict[str, List[float]]:
        """
        Compute expectation values of flux operators
        
        ⟨E^x(r_i)⟩ = ⟨Ψ|Ê^x(r_i)|Ψ⟩
        ⟨E^φ(r_i)⟩ = ⟨Ψ|Ê^φ(r_i)|Ψ⟩
        
        These determine the quantum-corrected metric components
        """
        E_x_expectations = []
        E_phi_expectations = []
        
        for site in range(self.n_sites):
            # Flux operators
            E_x_op = self.hilbert.flux_E_x_operator(site)
            E_phi_op = self.hilbert.flux_E_phi_operator(site)
            
            # Expectation values
            exp_E_x = np.real(np.conj(state) @ E_x_op @ state)
            exp_E_phi = np.real(np.conj(state) @ E_phi_op @ state)
            
            E_x_expectations.append(exp_E_x)
            E_phi_expectations.append(exp_E_phi)
        
        return {
            "E_x": E_x_expectations,
            "E_phi": E_phi_expectations
        }
    
    def compute_volume_expectations(self, state: np.ndarray) -> List[float]:
        """
        Compute volume operator expectation values
        
        ⟨V(r_i)⟩ = ⟨Ψ|√|E^x E^φ|(r_i)|Ψ⟩
        
        Gives quantum-corrected proper volume elements
        """
        volume_expectations = []
        
        for site in range(self.n_sites):
            V_op = self.hilbert.volume_operator(site)
            exp_V = np.real(np.conj(state) @ V_op @ state)
            volume_expectations.append(exp_V)
        
        return volume_expectations
    
    def compute_metric_components(self, flux_expectations: Dict) -> Dict[str, List[float]]:
        """
        Convert flux expectation values to metric components
        
        For spherical symmetry: ds² = -α²dt² + β²dr² + r²dΩ²
        
        Relation: E^x ∝ β r², E^φ ∝ r
        """
        E_x = flux_expectations["E_x"]
        E_phi = flux_expectations["E_phi"]
        
        alpha_components = []  # Lapse function
        beta_components = []   # Radial metric
        
        for i in range(self.n_sites):
            r = self.lattice_r[i]
            
            # Extract metric from flux eigenvalues
            # E^x = 4π β r² (spherical symmetry)
            # E^φ = 4π r
            
            if abs(E_phi[i]) > 1e-12 and r > 1e-12:
                # φ-flux gives effective radius
                r_eff = abs(E_phi[i]) / (4 * np.pi)
                
                # x-flux gives radial metric
                if abs(r_eff) > 1e-12:
                    beta_sq = abs(E_x[i]) / (4 * np.pi * r_eff**2)
                    beta = np.sqrt(beta_sq) if beta_sq > 0 else 1.0
                else:
                    beta = 1.0
            else:
                beta = 1.0
            
            # For static spacetime: α ≈ 1 (can be modified)
            alpha = 1.0
            
            alpha_components.append(alpha)
            beta_components.append(beta)
        
        return {
            "lapse_alpha": alpha_components,
            "radial_beta": beta_components
        }
    
    def compute_stress_energy_expectations(self, state: np.ndarray) -> Dict[str, List[float]]:
        """
        Compute stress-energy tensor expectation values
        
        For phantom scalar field:
        T^00 = (1/2)[π_φ²/|E| - |E|∇_r φ ∇^r φ - |E|m²φ²]
        T^rr = (1/2)[π_φ²/|E| + |E|∇_r φ ∇^r φ - |E|m²φ²]
        T^θθ = T^φφ = -(1/2)|E|m²φ²
        """
        T00_expectations = []  # Energy density
        Trr_expectations = []  # Radial pressure  
        Ttheta_expectations = []  # Angular pressure
        
        # Classical field profiles for semiclassical approximation
        scalar_field = self.reduced_data["exotic_matter_profile"]["scalar_field"]
        scalar_momentum = self.reduced_data["exotic_matter_profile"]["scalar_momentum"]
        mass_sq = self.reduced_data.get("scalar_mass_squared", 1e-20)
        
        # Volume expectations
        volume_exp = self.compute_volume_expectations(state)
        
        for site in range(self.n_sites):
            phi = scalar_field[site]
            pi_phi = scalar_momentum[site]
            V = volume_exp[site]
            
            # Kinetic energy density (phantom field has wrong sign)
            if V > 1e-12:
                kinetic_term = -0.5 * pi_phi**2 / V  # Negative for phantom
            else:
                kinetic_term = 0.0
            
            # Potential energy density
            potential_term = -0.5 * mass_sq * phi**2 * V
            
            # Gradient term (approximate with finite differences)
            if site > 0 and site < self.n_sites - 1:
                dr_minus = self.lattice_r[site] - self.lattice_r[site-1]
                dr_plus = self.lattice_r[site+1] - self.lattice_r[site]
                dphi_dr = (scalar_field[site+1] - scalar_field[site-1]) / (dr_plus + dr_minus)
                
                gradient_term = 0.5 * V * dphi_dr**2  # Normal sign for spatial gradient
            else:
                gradient_term = 0.0
            
            # Stress-energy components
            T00 = kinetic_term - gradient_term + potential_term  # Energy density
            Trr = kinetic_term + gradient_term + potential_term  # Radial pressure
            Ttheta = potential_term  # Angular pressure
            
            T00_expectations.append(T00)
            Trr_expectations.append(Trr)
            Ttheta_expectations.append(Ttheta)
        
        return {
            "T00": T00_expectations,  # Energy density
            "Trr": Trr_expectations,  # Radial pressure
            "Ttheta": Ttheta_expectations,  # Angular pressure
            "Tphi": Ttheta_expectations   # T^φφ = T^θθ for spherical symmetry
        }
    
    def compute_curvature_expectations(self, state: np.ndarray,
                                     flux_expectations: Dict) -> Dict[str, List[float]]:
        """
        Compute curvature expectation values from quantum geometry
        
        Uses quantum-corrected metric to compute:
        - Ricci scalar R
        - Einstein tensor G_μν
        """
        E_x = flux_expectations["E_x"]
        E_phi = flux_expectations["E_phi"]
        
        ricci_scalar = []
        einstein_00 = []  # G_00 component
        
        for i in range(self.n_sites):
            r = self.lattice_r[i]
            
            # Quantum-corrected metric from flux eigenvalues
            if abs(E_phi[i]) > 1e-12 and r > 1e-12:
                r_eff = abs(E_phi[i]) / (4 * np.pi)
                
                if abs(r_eff) > 1e-12 and abs(E_x[i]) > 1e-12:
                    beta_sq = abs(E_x[i]) / (4 * np.pi * r_eff**2)
                    
                    # Approximate curvature (spherical case)
                    # R ≈ 2/r² + derivatives of metric functions
                    R = 2.0 / (r_eff**2 * beta_sq)
                    
                    # Einstein tensor G_00 (related to energy density)
                    G00 = 0.5 * R  # Simplified
                    
                else:
                    R = 0.0
                    G00 = 0.0
            else:
                R = 0.0
                G00 = 0.0
            
            ricci_scalar.append(R)
            einstein_00.append(G00)
        
        return {
            "ricci_scalar": ricci_scalar,
            "einstein_00": einstein_00
        }
    
    def compute_all_observables(self, state: np.ndarray) -> Dict:
        """
        Compute all physical observables from quantum state
        
        Returns complete set of expectation values for export
        """
        print("Computing flux expectation values...")
        flux_exp = self.compute_flux_expectations(state)
        
        print("Computing metric components...")
        metric_exp = self.compute_metric_components(flux_exp)
        
        print("Computing stress-energy tensor...")
        stress_energy_exp = self.compute_stress_energy_expectations(state)
        
        print("Computing curvature observables...")
        curvature_exp = self.compute_curvature_expectations(state, flux_exp)
        
        print("Computing volume expectation values...")
        volume_exp = self.compute_volume_expectations(state)
        
        # Combine all results
        observables = {
            "lattice_r": self.lattice_r,
            "flux_expectations": flux_exp,
            "metric_components": metric_exp,
            "stress_energy_tensor": stress_energy_exp,
            "curvature": curvature_exp,
            "volume": volume_exp,
            "metadata": {
                "n_lattice_points": self.n_sites,
                "physical_units": "Planck_units",
                "state_normalization": float(np.linalg.norm(state))
            }
        }
        
        return observables
    
    def save_observables(self, observables: Dict, output_dir: str):
        """Save observables to JSON files for warp-framework integration"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save complete observables
        full_file = os.path.join(output_dir, "expectation_values.json")
        with open(full_file, 'w') as f:
            json.dump(observables, f, indent=2)
        
        # Save T^00 separately for warp-framework
        T00_data = {
            "r": observables["lattice_r"],
            "T00": observables["stress_energy_tensor"]["T00"],
            "metadata": {
                "description": "Energy density T^00 from LQG quantum state",
                "units": "Planck_units",
                "source": "loop_quantum_gravity_midisuperspace"
            }
        }
        
        T00_file = os.path.join(output_dir, "expectation_T00.json")
        with open(T00_file, 'w') as f:
            json.dump(T00_data, f, indent=2)
        
        # Save flux expectations
        flux_data = {
            "r": observables["lattice_r"],
            "E_x": observables["flux_expectations"]["E_x"],
            "E_phi": observables["flux_expectations"]["E_phi"],
            "metadata": {
                "description": "Quantum flux expectation values",
                "units": "Planck_units"
            }
        }
        
        flux_file = os.path.join(output_dir, "expectation_E.json")
        with open(flux_file, 'w') as f:
            json.dump(flux_data, f, indent=2)
        
        print(f"Observables saved to {output_dir}/:")
        print(f"  Complete: {full_file}")
        print(f"  T^00: {T00_file}")
        print(f"  Flux: {flux_file}")
        
        return {
            "expectation_values": full_file,
            "expectation_T00": T00_file,
            "expectation_E": flux_file
        }


def load_physical_state(state_file: str, state_index: int = 0) -> np.ndarray:
    """Load physical state from solve_constraint.py output"""
    states = np.load(state_file)
    
    if states.ndim == 1:
        return states
    elif states.ndim == 2:
        if state_index >= states.shape[1]:
            raise ValueError(f"State index {state_index} >= {states.shape[1]}")
        return states[:, state_index]
    else:
        raise ValueError(f"Unexpected state array shape: {states.shape}")


def main():
    """Command line interface for expectation value calculation"""
    import argparse
    from kinematical_hilbert import load_lattice_from_reduced_variables
    
    parser = argparse.ArgumentParser(description="Compute LQG expectation values")
    parser.add_argument("--lattice", type=str, required=True,
                       help="JSON file with reduced variables")
    parser.add_argument("--states", type=str, required=True,
                       help="NPY file with physical states")
    parser.add_argument("--state-index", type=int, default=0,
                       help="Index of state to analyze")
    parser.add_argument("--out", type=str, default="quantum_outputs",
                       help="Output directory")
    parser.add_argument("--mu-range", type=int, nargs=2, default=[-3, 3],
                       help="Range for μ quantum numbers")
    parser.add_argument("--nu-range", type=int, nargs=2, default=[-3, 3],
                       help="Range for ν quantum numbers")
    
    args = parser.parse_args()
    
    # Load reduced variables
    with open(args.lattice, 'r') as f:
        reduced_data = json.load(f)
    
    # Create Hilbert space
    config = load_lattice_from_reduced_variables(args.lattice)
    config.mu_range = tuple(args.mu_range)
    config.nu_range = tuple(args.nu_range)
    
    hilbert = MidisuperspaceHilbert(config)
    
    # Create constraint (needed for stress-energy calculation)
    constraint = HamiltonianConstraint(hilbert, reduced_data)
    
    # Load physical state
    print(f"Loading physical state from {args.states}")
    state = load_physical_state(args.states, args.state_index)
    
    print(f"State normalization: {np.linalg.norm(state):.6f}")
    
    # Compute observables
    calculator = ExpectationValueCalculator(hilbert, constraint, reduced_data)
    observables = calculator.compute_all_observables(state)
    
    # Save results
    files = calculator.save_observables(observables, args.out)
    
    # Print summary
    print(f"\nObservable summary:")
    print(f"  Lattice points: {len(observables['lattice_r'])}")
    
    # Show sample values
    n_show = min(3, len(observables['lattice_r']))
    for i in range(n_show):
        r = observables['lattice_r'][i]
        T00 = observables['stress_energy_tensor']['T00'][i]
        E_x = observables['flux_expectations']['E_x'][i]
        
        print(f"  r[{i}] = {r:.2e}: T^00 = {T00:.3e}, E^x = {E_x:.3f}")


if __name__ == "__main__":
    main()
