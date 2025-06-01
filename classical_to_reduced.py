#!/usr/bin/env python3
"""
Classical to Reduced Variables (Task 1)

Converts classical warp bubble metric to midisuperspace variables:
- Spherical symmetry reduction: ds² → (Kₓ, Kφ; Eˣ, Eφ)(r)
- Handles exotic matter fields (phantom scalar)
- Outputs discrete lattice representation for LQG quantization

Author: Loop Quantum Gravity Implementation
"""

import numpy as np
import sympy as sp
import json
import argparse
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class WarpGeometry:
    """Classical warp bubble geometry configuration"""
    throat_radius: float
    asymptotic_radius: float 
    wall_thickness: float
    exotic_energy_density: float
    scalar_field_mass: float
    
    
@dataclass
class MidisuperspaceVariables:
    """Reduced gravitational and matter variables on lattice"""
    lattice_r: List[float]
    E_x: List[float]        # Radial triad flux E^x(r_i)
    E_phi: List[float]      # Angular triad flux E^φ(r_i)  
    K_x: List[float]        # Radial connection K_x(r_i)
    K_phi: List[float]      # Angular connection K_φ(r_i)
    scalar_field: List[float]  # Phantom scalar φ(r_i)
    scalar_momentum: List[float]  # Conjugate momentum π_φ(r_i)
    

class ClassicalToReduced:
    """Converts classical warp metrics to LQG midisuperspace variables"""
    
    def __init__(self, geometry: WarpGeometry, n_points: int = 20):
        self.geometry = geometry
        self.n_points = n_points
        self.planck_length = 1.616e-35  # meters
        self.planck_mass = 2.176e-8     # kg
        self.c = 299792458              # m/s
        self.hbar = 1.055e-34          # J⋅s
        self.G = 6.674e-11             # m³/kg⋅s²
        
        # Immirzi parameter (commonly γ ≈ 0.2375)
        self.gamma = 0.2375
        
    def create_radial_lattice(self) -> List[float]:
        """Create discrete radial lattice from throat to asymptotic region"""
        r_min = self.geometry.throat_radius
        r_max = self.geometry.asymptotic_radius
        
        # Use logarithmic spacing to capture throat physics
        log_min = np.log(r_min)
        log_max = np.log(r_max)
        log_points = np.linspace(log_min, log_max, self.n_points)
        
        return np.exp(log_points).tolist()
    
    def classical_warp_metric(self, r: float) -> Tuple[float, float]:
        """
        Classical warp bubble metric functions
        ds² = -α(r)²dt² + β(r)²dr² + r²(dθ² + sin²θ dφ²)
        
        Returns: (α(r), β(r)) - lapse and radial metric functions
        """
        r0 = self.geometry.throat_radius
        sigma = self.geometry.wall_thickness
        rho = self.geometry.exotic_energy_density
        
        # Smooth warp profile with Gaussian wall
        wall_factor = np.exp(-(r - r0)**2 / (2*sigma**2))
        
        # Lapse function (redshift factor)
        alpha = 1.0 - 0.5 * rho * wall_factor
        
        # Radial metric (proper distance factor)  
        beta = 1.0 + 0.3 * rho * wall_factor
        
        return alpha, beta
    
    def compute_triad_variables(self, lattice: List[float]) -> Tuple[List[float], List[float]]:
        """
        Compute densitized triads E^x, E^φ from metric
        
        For spherical symmetry:
        E^x = √|det(q)| e^x_a ∂/∂x^a = r² sin(θ) ∂/∂r  
        E^φ = √|det(q)| e^φ_a ∂/∂x^a = r sin(θ) ∂/∂θ
        """
        E_x = []
        E_phi = []
        
        for r in lattice:
            alpha, beta = self.classical_warp_metric(r)
            
            # Metric determinant: |det(q)| = β²(r) × r⁴ sin²(θ)
            # For unit sphere surface: integrate out angles → 4π
            sqrt_det_q = beta * r**2
            
            # Densitized triads (integrate over sphere angles)
            e_x = 4 * np.pi * sqrt_det_q  # Radial direction
            e_phi = 4 * np.pi * sqrt_det_q / r  # Angular directions
            
            E_x.append(e_x)
            E_phi.append(e_phi)
            
        return E_x, E_phi
    
    def compute_extrinsic_curvature(self, lattice: List[float]) -> Tuple[List[float], List[float]]:
        """
        Compute extrinsic curvature K_x, K_φ from metric evolution
        
        Using ADM decomposition: K_ab = (1/2α)[∂_t q_ab - ∇_a β_b - ∇_b β_a]
        For static spherical metrics, this reduces to intrinsic curvature
        """
        K_x = []
        K_phi = []
        
        for i, r in enumerate(lattice):
            alpha, beta = self.classical_warp_metric(r)
            
            # Approximate derivatives using finite differences
            dr = 0.01 * r
            alpha_plus, beta_plus = self.classical_warp_metric(r + dr)
            alpha_minus, beta_minus = self.classical_warp_metric(r - dr)
            
            dalpha_dr = (alpha_plus - alpha_minus) / (2 * dr)
            dbeta_dr = (beta_plus - beta_minus) / (2 * dr)
            
            # Connection components (Ashtekar-Barbero variables)
            # K_x ∝ derivative of radial metric
            k_x = -0.5 * dbeta_dr / (alpha * beta)
            
            # K_φ ∝ curvature of 2-sphere embedded in 3-space
            k_phi = (beta - 1.0) / (alpha * r)
            
            K_x.append(k_x)
            K_phi.append(k_phi)
            
        return K_x, K_phi
    
    def compute_scalar_field_profile(self, lattice: List[float]) -> Tuple[List[float], List[float]]:
        """
        Compute phantom scalar field and momentum from exotic matter
        
        Stress-energy: T_μν = ∂_μφ ∂_νφ - (1/2)g_μν[(∂φ)² + m²φ²]
        For phantom field: kinetic term has wrong sign → negative energy
        """
        scalar_field = []
        scalar_momentum = []
        
        r0 = self.geometry.throat_radius
        sigma = self.geometry.wall_thickness
        phi0 = np.sqrt(abs(self.geometry.exotic_energy_density))
        m = self.geometry.scalar_field_mass
        
        for r in lattice:
            # Phantom field profile (localized at throat)
            wall_profile = np.exp(-(r - r0)**2 / (2*sigma**2))
            
            # Field value (complex for phantom field)
            phi = phi0 * wall_profile
            
            # Conjugate momentum π_φ = √|det(q)| n^μ ∂_μφ
            # For static config: π_φ ≈ 0, but include small fluctuation
            pi_phi = 0.1 * phi0 * (r - r0) / sigma**2 * wall_profile
            
            scalar_field.append(phi)
            scalar_momentum.append(pi_phi)
            
        return scalar_field, scalar_momentum
    
    def reduce_to_midisuperspace(self) -> MidisuperspaceVariables:
        """Main reduction: classical warp metric → discrete LQG variables"""
        
        # Create radial lattice
        lattice = self.create_radial_lattice()
        
        # Compute gravitational variables
        E_x, E_phi = self.compute_triad_variables(lattice)
        K_x, K_phi = self.compute_extrinsic_curvature(lattice)
        
        # Compute matter variables  
        scalar_field, scalar_momentum = self.compute_scalar_field_profile(lattice)
        
        return MidisuperspaceVariables(
            lattice_r=lattice,
            E_x=E_x,
            E_phi=E_phi,
            K_x=K_x,
            K_phi=K_phi,
            scalar_field=scalar_field,
            scalar_momentum=scalar_momentum
        )
    
    def save_to_json(self, variables: MidisuperspaceVariables, filename: str):
        """Save reduced variables to JSON format"""
        data = {
            "metadata": {
                "throat_radius": self.geometry.throat_radius,
                "asymptotic_radius": self.geometry.asymptotic_radius,
                "n_lattice_points": len(variables.lattice_r),
                "immirzi_parameter": self.gamma,
                "planck_length": self.planck_length
            },
            "lattice_r": variables.lattice_r,
            "E_classical": {
                "E^x": variables.E_x,
                "E^phi": variables.E_phi
            },
            "K_classical": {
                "K_x": variables.K_x,
                "K_phi": variables.K_phi  
            },
            "exotic_matter_profile": {
                "scalar_field": variables.scalar_field,
                "scalar_momentum": variables.scalar_momentum
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Reduced variables saved to {filename}")
        print(f"Lattice points: {len(variables.lattice_r)}")
        print(f"Radial range: {variables.lattice_r[0]:.3e} → {variables.lattice_r[-1]:.3e}")


def main():
    """Command line interface"""
    parser = argparse.ArgumentParser(description="Convert classical warp metric to LQG variables")
    parser.add_argument("--throat-radius", type=float, default=1e-15, 
                       help="Warp throat radius (meters)")
    parser.add_argument("--asymptotic-radius", type=float, default=1e-10,
                       help="Asymptotic radius (meters)")
    parser.add_argument("--wall-thickness", type=float, default=1e-16,
                       help="Exotic matter wall thickness (meters)")
    parser.add_argument("--exotic-density", type=float, default=-1e30,
                       help="Exotic energy density (kg/m³)")
    parser.add_argument("--scalar-mass", type=float, default=1e-10,
                       help="Phantom scalar field mass (kg)")
    parser.add_argument("--n-points", type=int, default=20,
                       help="Number of lattice points")
    parser.add_argument("--output", type=str, default="reduced_variables.json",
                       help="Output JSON filename")
    
    args = parser.parse_args()
    
    # Create geometry configuration
    geometry = WarpGeometry(
        throat_radius=args.throat_radius,
        asymptotic_radius=args.asymptotic_radius,
        wall_thickness=args.wall_thickness,
        exotic_energy_density=args.exotic_density,
        scalar_field_mass=args.scalar_mass
    )
    
    # Perform reduction
    reducer = ClassicalToReduced(geometry, args.n_points)
    variables = reducer.reduce_to_midisuperspace()
    
    # Save results
    reducer.save_to_json(variables, args.output)


if __name__ == "__main__":
    main()
