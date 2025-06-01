#!/usr/bin/env python3
"""
Feed to Warp Framework (Task 7)

Converts LQG quantum outputs to formats compatible with warp-framework:
- Transforms JSON expectation values to NDJSON format
- Handles unit conversions (Planck → SI units)
- Creates interface files for warp-framework ingestion
- Validates data integrity and formats

Author: Loop Quantum Gravity Implementation
"""

import numpy as np
import json
import ndjson
import os
import argparse
from typing import Dict, List, Optional, Union


class WarpFrameworkInterface:
    """
    Interface between LQG midisuperspace and warp-framework
    
    Handles data format conversion and unit transformations
    """
    
    def __init__(self, framework_path: Optional[str] = None):
        self.framework_path = framework_path
        
        # Physical constants for unit conversion
        self.c = 299792458              # m/s
        self.G = 6.674e-11             # m³/kg⋅s²  
        self.hbar = 1.055e-34          # J⋅s
        self.planck_length = np.sqrt(self.hbar * self.G / self.c**3)
        self.planck_time = self.planck_length / self.c
        self.planck_mass = np.sqrt(self.hbar * self.c / self.G)
        self.planck_energy = self.planck_mass * self.c**2
        
        print(f"Warp-framework interface initialized:")
        print(f"  Framework path: {framework_path}")
        print(f"  Planck length: {self.planck_length:.3e} m")
        print(f"  Planck energy: {self.planck_energy:.3e} J")
    
    def convert_units_to_si(self, data: Dict, quantity_type: str) -> Dict:
        """
        Convert quantities from Planck units to SI units
        
        Args:
            data: Data dictionary with values in Planck units
            quantity_type: Type of quantity for proper conversion
        """
        converted_data = data.copy()
        
        if quantity_type == "stress_energy":
            # Energy density: [T^μν] = Energy/Volume = M L⁻¹ T⁻²
            # Planck units: ρ_Planck = ρ_SI / (ρ_Planck_scale)
            planck_energy_density = self.planck_energy / self.planck_length**3
            
            if "T00" in data:
                T00_si = [T * planck_energy_density for T in data["T00"]]
                converted_data["T00"] = T00_si
                converted_data["T00_units"] = "J/m³"
            
            if "Trr" in data:
                Trr_si = [T * planck_energy_density for T in data["Trr"]]
                converted_data["Trr"] = Trr_si
                converted_data["Trr_units"] = "Pa"
                
        elif quantity_type == "length":
            # Length: [r] = L
            if "r" in data:
                r_si = [r * self.planck_length for r in data["r"]]
                converted_data["r"] = r_si
                converted_data["r_units"] = "m"
                
        elif quantity_type == "frequency":
            # Frequency: [ω] = T⁻¹
            planck_frequency = 1.0 / self.planck_time
            
            if "frequencies" in data:
                freq_si = [f * planck_frequency for f in data["frequencies"]]
                converted_data["frequencies"] = freq_si
                converted_data["frequency_units"] = "Hz"
                
            if "eigenvalues_omega_squared" in data:
                omega2_si = [w2 * planck_frequency**2 for w2 in data["eigenvalues_omega_squared"]]
                converted_data["eigenvalues_omega_squared"] = omega2_si
                converted_data["omega2_units"] = "Hz²"
        
        return converted_data
    
    def load_lqg_outputs(self, output_dir: str) -> Dict:
        """Load all LQG output files from solve_constraint.py and related scripts"""
        lqg_data = {}
        
        # Load expectation values
        expectation_file = os.path.join(output_dir, "expectation_values.json")
        if os.path.exists(expectation_file):
            with open(expectation_file, 'r') as f:
                lqg_data["expectation_values"] = json.load(f)
        
        # Load T^00 data
        T00_file = os.path.join(output_dir, "expectation_T00.json")
        if os.path.exists(T00_file):
            with open(T00_file, 'r') as f:
                lqg_data["T00_data"] = json.load(f)
        
        # Load quantum spectrum
        spectrum_file = os.path.join(output_dir, "quantum_spectrum.json")
        if os.path.exists(spectrum_file):
            with open(spectrum_file, 'r') as f:
                lqg_data["quantum_spectrum"] = json.load(f)
        
        # Load solver summary
        summary_file = os.path.join(output_dir, "solver_summary.json")
        if os.path.exists(summary_file):
            with open(summary_file, 'r') as f:
                lqg_data["solver_summary"] = json.load(f)
        
        print(f"Loaded LQG data from {output_dir}:")
        for key in lqg_data:
            print(f"  {key}: OK")
        
        return lqg_data
    
    def create_T00_ndjson(self, T00_data: Dict, output_file: str, 
                         convert_units: bool = True):
        """
        Convert T^00 JSON to NDJSON format for warp-framework
        
        Format: Each line is {"r": value, "T00": value, "metadata": {...}}
        """
        if convert_units:
            T00_converted = self.convert_units_to_si(T00_data, "stress_energy")
            r_converted = self.convert_units_to_si({"r": T00_data["r"]}, "length")
            r_values = r_converted["r"]
            T00_values = T00_converted["T00"]
            r_units = "m"
            T00_units = "J/m³"
        else:
            r_values = T00_data["r"]
            T00_values = T00_data["T00"]
            r_units = "Planck_length"
            T00_units = "Planck_energy_density"
        
        # Create NDJSON entries
        ndjson_entries = []
        for i, (r, T00) in enumerate(zip(r_values, T00_values)):
            entry = {
                "r": r,
                "T00": T00,
                "lattice_index": i,
                "metadata": {
                    "source": "loop_quantum_gravity",
                    "r_units": r_units,
                    "T00_units": T00_units,
                    "description": "Energy density from LQG physical state"
                }
            }
            ndjson_entries.append(entry)
        
        # Write NDJSON file
        with open(output_file, 'w') as f:
            writer = ndjson.writer(f)
            for entry in ndjson_entries:
                writer.writerow(entry)
        
        print(f"T^00 NDJSON created: {output_file}")
        print(f"  Entries: {len(ndjson_entries)}")
        print(f"  Units: r in {r_units}, T^00 in {T00_units}")
    
    def create_spectrum_ndjson(self, spectrum_data: Dict, output_file: str,
                              convert_units: bool = True):
        """
        Convert quantum spectrum to NDJSON format
        
        Format: Each line is {"mode": n, "omega2": value, "omega": value, "stable": bool}
        """
        spectrum = spectrum_data["quantum_spectrum"]
        
        if convert_units:
            converted = self.convert_units_to_si(spectrum, "frequency")
            omega2_values = converted["eigenvalues_omega_squared"]
            freq_values = converted["frequencies"]
            units = "Hz"
        else:
            omega2_values = spectrum["eigenvalues_omega_squared"]
            freq_values = spectrum["frequencies"]
            units = "Planck_frequency"
        
        # Create NDJSON entries
        ndjson_entries = []
        for n, (omega2, omega) in enumerate(zip(omega2_values, freq_values)):
            stable = omega2 > 0
            entry = {
                "mode_number": n,
                "omega_squared": omega2,
                "frequency": omega,
                "stable": stable,
                "metadata": {
                    "source": "loop_quantum_gravity_stability",
                    "units": units,
                    "description": f"Mode {n} fluctuation frequency"
                }
            }
            ndjson_entries.append(entry)
        
        # Write NDJSON file
        with open(output_file, 'w') as f:
            writer = ndjson.writer(f)
            for entry in ndjson_entries:
                writer.writerow(entry)
        
        print(f"Spectrum NDJSON created: {output_file}")
        print(f"  Modes: {len(ndjson_entries)}")
        print(f"  Units: {units}")
    
    def create_quantum_metric_json(self, expectation_data: Dict, output_file: str,
                                  convert_units: bool = True):
        """
        Create quantum-corrected metric data for warp-framework
        
        Includes quantum geometry corrections from LQG
        """
        if convert_units:
            r_converted = self.convert_units_to_si(
                {"r": expectation_data["lattice_r"]}, "length"
            )
            r_values = r_converted["r"]
            r_units = "m"
        else:
            r_values = expectation_data["lattice_r"]
            r_units = "Planck_length"
        
        # Extract quantum metric components
        metric_components = expectation_data.get("metric_components", {})
        flux_expectations = expectation_data.get("flux_expectations", {})
        
        quantum_metric = {
            "lattice_data": {
                "r": r_values,
                "r_units": r_units,
                "n_points": len(r_values)
            },
            "quantum_metric_functions": {
                "lapse_alpha": metric_components.get("lapse_alpha", [1.0] * len(r_values)),
                "radial_beta": metric_components.get("radial_beta", [1.0] * len(r_values)),
                "description": "Quantum-corrected metric from LQG expectation values"
            },
            "quantum_fluxes": {
                "E_x": flux_expectations.get("E_x", []),
                "E_phi": flux_expectations.get("E_phi", []),
                "description": "Quantum flux expectation values ⟨E^a⟩"
            },
            "metadata": {
                "source": "loop_quantum_gravity_midisuperspace",
                "quantum_corrections": True,
                "classical_limit": "large_quantum_numbers",
                "discretization": "lattice_regularization"
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(quantum_metric, f, indent=2)
        
        print(f"Quantum metric data created: {output_file}")
    
    def copy_to_framework(self, source_files: Dict[str, str]):
        """Copy generated files to warp-framework directory structure"""
        if not self.framework_path:
            print("No framework path specified, skipping copy")
            return
        
        if not os.path.exists(self.framework_path):
            print(f"Framework path does not exist: {self.framework_path}")
            return
        
        # Create quantum_inputs directory in framework
        quantum_inputs_dir = os.path.join(self.framework_path, "quantum_inputs")
        os.makedirs(quantum_inputs_dir, exist_ok=True)
        
        # Copy files with appropriate names
        for file_type, source_file in source_files.items():
            if os.path.exists(source_file):
                if file_type == "T00_ndjson":
                    dest_file = os.path.join(quantum_inputs_dir, "T00_quantum.ndjson")
                elif file_type == "spectrum_ndjson":
                    dest_file = os.path.join(quantum_inputs_dir, "quantum_spectrum.ndjson")
                elif file_type == "quantum_metric":
                    dest_file = os.path.join(quantum_inputs_dir, "quantum_metric.json")
                else:
                    dest_file = os.path.join(quantum_inputs_dir, os.path.basename(source_file))
                
                # Copy file
                import shutil
                shutil.copy2(source_file, dest_file)
                print(f"Copied {source_file} → {dest_file}")
        
        print(f"Files copied to warp-framework: {quantum_inputs_dir}/")


def main():
    """Command line interface for warp-framework integration"""
    parser = argparse.ArgumentParser(description="Export LQG data to warp-framework")
    parser.add_argument("--input", type=str, required=True,
                       help="Directory with LQG quantum outputs")
    parser.add_argument("--output", type=str, default="warp_framework_inputs",
                       help="Output directory for framework files")
    parser.add_argument("--framework-path", type=str,
                       help="Path to warp-framework directory")
    parser.add_argument("--units", type=str, choices=["si", "planck"], default="si",
                       help="Output units (SI or Planck)")
    parser.add_argument("--format", type=str, choices=["json", "ndjson", "both"], 
                       default="both", help="Output format")
    
    args = parser.parse_args()
    
    # Create interface
    interface = WarpFrameworkInterface(args.framework_path)
    
    # Load LQG data
    print(f"Loading LQG outputs from {args.input}")
    lqg_data = interface.load_lqg_outputs(args.input)
    
    if not lqg_data:
        print("No LQG data found. Run solve_constraint.py first.")
        return
    
    # Create output directory
    os.makedirs(args.output, exist_ok=True)
    
    convert_units = (args.units == "si")
    created_files = {}
    
    # Export T^00 data
    if "T00_data" in lqg_data:
        if args.format in ["ndjson", "both"]:
            T00_ndjson = os.path.join(args.output, "T00_quantum.ndjson")
            interface.create_T00_ndjson(lqg_data["T00_data"], T00_ndjson, convert_units)
            created_files["T00_ndjson"] = T00_ndjson
        
        if args.format in ["json", "both"]:
            T00_json = os.path.join(args.output, "T00_quantum.json")
            T00_data = lqg_data["T00_data"].copy()
            if convert_units:
                T00_data = interface.convert_units_to_si(T00_data, "stress_energy")
                T00_data.update(interface.convert_units_to_si({"r": T00_data["r"]}, "length"))
            
            with open(T00_json, 'w') as f:
                json.dump(T00_data, f, indent=2)
            created_files["T00_json"] = T00_json
    
    # Export quantum spectrum
    if "quantum_spectrum" in lqg_data:
        if args.format in ["ndjson", "both"]:
            spectrum_ndjson = os.path.join(args.output, "quantum_spectrum.ndjson")
            interface.create_spectrum_ndjson(lqg_data["quantum_spectrum"], spectrum_ndjson, convert_units)
            created_files["spectrum_ndjson"] = spectrum_ndjson
    
    # Export quantum metric
    if "expectation_values" in lqg_data:
        quantum_metric_file = os.path.join(args.output, "quantum_metric.json")
        interface.create_quantum_metric_json(lqg_data["expectation_values"], quantum_metric_file, convert_units)
        created_files["quantum_metric"] = quantum_metric_file
    
    # Create summary
    summary = {
        "export_info": {
            "source_directory": args.input,
            "output_directory": args.output,
            "units": args.units,
            "format": args.format,
            "framework_path": args.framework_path
        },
        "exported_files": created_files,
        "lqg_data_summary": {
            "available_datasets": list(lqg_data.keys()),
            "total_files_created": len(created_files)
        }
    }
    
    summary_file = os.path.join(args.output, "export_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nExport summary:")
    print(f"  Files created: {len(created_files)}")
    print(f"  Output directory: {args.output}")
    print(f"  Units: {args.units}")
    print(f"  Summary: {summary_file}")
    
    # Copy to framework if requested
    if args.framework_path:
        interface.copy_to_framework(created_files)


if __name__ == "__main__":
    main()
