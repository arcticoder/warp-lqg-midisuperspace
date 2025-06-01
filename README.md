# Warp LQG Midisuperspace

Loop Quantum Gravity quantization of warp bubble geometries using midisuperspace reduction.

## Overview

This repository implements the Loop Quantum Gravity (LQG) quantization of warp bubble spacetimes through a midisuperspace approach. It reduces the infinite-dimensional field theory to a finite-dimensional quantum mechanical system with spherical symmetry, making the problem tractable while preserving the essential physics of exotic matter and negative energy regions.

## Key Components

1. **Symmetry Reduction**: Reduces warp bubble geometry to spherically symmetric variables
2. **Kinematical Hilbert Space**: Constructs discrete quantum states on a radial lattice
3. **Hamiltonian Constraint**: Implements the quantum Einstein constraint with holonomy corrections
4. **Physical States**: Solves for states satisfying the Wheeler-DeWitt equation
5. **Expectation Values**: Computes quantum-corrected stress-energy tensor ⟨T⁰⁰⟩
6. **Stability Analysis**: Extracts quantum fluctuation spectrum {ω²ₙ}

## Structure

```
├── classical_to_reduced.py      # Task 1: warp metric → (Kₓ, Kφ; Eˣ, Eφ)
├── kinematical_hilbert.py       # Task 2: defines lattice, basis states, flux ops
├── hamiltonian_constraint.py    # Task 3: builds Ĥ_grav + Ĥ_matter on lattice
├── solve_constraint.py          # Task 4: numerically solve Ĥ |Ψ⟩ = 0
├── expectation_values.py        # Task 5: compute ⟨E⟩, ⟨T⁰⁰⟩ from solved state
├── quantum_stability.py         # Task 6: discrete SL operator for ω²ₙ
├── feed_to_warp_framework.py    # Exports ⟨T⁰⁰(rᵢ)⟩ as JSON/NDJSON
├── examples/                    # Example inputs & outputs
└── tests/                      # Unit tests
```

## Usage

1. **Generate reduced variables from classical warp metric:**
   ```pwsh
   python classical_to_reduced.py --config examples/warp_config.json --out examples/example_reduced_variables.json
   ```

2. **Solve quantum constraints:**
   ```pwsh
   python solve_constraint.py --lattice examples/example_reduced_variables.json --out quantum_outputs
   ```

3. **Export to warp-framework:**
   ```pwsh
   python feed_to_warp_framework.py --input quantum_outputs --framework-path ../warp-framework
   ```

## Physical Motivation

The midisuperspace approach captures the essential quantum gravity effects while remaining computationally feasible:

- **Loop quantization** introduces discrete area spectra and resolves curvature singularities
- **Holonomy corrections** modify the classical Einstein equations at Planck scales
- **Quantum bounce** replaces classical singularities with smooth quantum transitions
- **Exotic matter** is consistently quantized alongside the gravitational degrees of freedom

## Dependencies

- Python 3.8+
- NumPy, SciPy (numerical computations)
- SymPy (symbolic mathematics)
- python-ndjson (data export)

Install with:
```pwsh
pip install -r requirements.txt
```

## Theory Background

This implementation follows the canonical LQG quantization program:

1. **Phase space variables**: (Kₐ, Eᵃ) where K is the extrinsic curvature and E is the densitized triad
2. **Holonomy-flux algebra**: Quantum operators satisfy [Ĥᵢ, Êʲ] = i ħ γ κ δᵢⱼ
3. **Regularization**: Curvature → holonomies around finite loops of Planck area
4. **Physical states**: Solutions to Ĥ|Ψ⟩ = 0 (Wheeler-DeWitt equation)

## References

- Ashtekar, A. & Bojowald, M. "Loop quantum cosmology" (2006)
- Bojowald, M. "Spherically symmetric quantum geometry" (2004)
- Thiemann, T. "Modern canonical quantum general relativity" (2007)
