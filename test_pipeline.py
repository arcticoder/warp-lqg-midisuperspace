#!/usr/bin/env python3
"""
Test the complete LQG pipeline from start to finish
"""

import os
import subprocess
import json
import sys

def run_command(cmd, description):
    """Run a command and check for success"""
    print(f"\n{'='*60}")
    print(f"Step: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"❌ FAILED: {description}")
        print(f"Error: {result.stderr}")
        return False
    else:
        print(f"✅ SUCCESS: {description}")
        if result.stdout.strip():
            print("Output:")
            print(result.stdout)
        return True

def check_file_exists(filepath, description):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✅ {description}: {filepath}")
        return True
    else:
        print(f"❌ Missing {description}: {filepath}")
        return False

def main():
    print("LQG Warp Bubble Pipeline Test")
    print("="*60)
    
    # Ensure we're in the correct directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    # Clean up any previous outputs
    if os.path.exists("quantum_outputs"):
        import shutil
        shutil.rmtree("quantum_outputs")
    if os.path.exists("warp_inputs"):
        import shutil
        shutil.rmtree("warp_inputs")
    
    success = True
    
    # Step 1: Solve quantum constraints
    success &= run_command(
        "python solve_constraint.py --lattice examples/example_reduced_variables.json --out quantum_outputs --n-states 3 --mu-range 0 1 --nu-range 0 1 --tolerance 1e-6",
        "Solving quantum constraints"
    )
    
    # Step 2: Compute expectation values
    success &= run_command(
        "python expectation_values.py --lattice examples/example_reduced_variables.json --states quantum_outputs/physical_states.npy --out quantum_outputs --mu-range 0 1 --nu-range 0 1",
        "Computing expectation values"
    )
    
    # Step 3: Quantum stability analysis
    success &= run_command(
        "python quantum_stability.py --observables quantum_outputs/expectation_values.json --out quantum_outputs --n-modes 5",
        "Quantum stability analysis"
    )
    
    # Step 4: Export to warp framework
    success &= run_command(
        "python feed_to_warp_framework.py --input quantum_outputs --output warp_inputs --format json",
        "Export to warp framework"
    )
    
    # Verify outputs
    print(f"\n{'='*60}")
    print("VERIFYING OUTPUTS")
    print(f"{'='*60}")
    
    # Check key output files
    files_to_check = [
        ("quantum_outputs/eigenvalues.npy", "Physical state eigenvalues"),
        ("quantum_outputs/expectation_T00.json", "T^00 expectation values"),
        ("quantum_outputs/quantum_spectrum.json", "Quantum fluctuation spectrum"),
        ("warp_inputs/T00_quantum.json", "T^00 for warp framework"),
        ("warp_inputs/quantum_metric.json", "Quantum metric for warp framework"),
    ]
    
    for filepath, description in files_to_check:
        success &= check_file_exists(filepath, description)
    
    # Check data integrity
    try:
        with open("quantum_outputs/expectation_T00.json", 'r') as f:
            t00_data = json.load(f)
        print(f"✅ T^00 data has {len(t00_data['T00'])} lattice points")
        
        with open("quantum_outputs/quantum_spectrum.json", 'r') as f:
            spectrum_data = json.load(f)
        n_modes = spectrum_data["quantum_spectrum"]["n_modes"]
        stability = spectrum_data["stability_analysis"]["stability_verdict"]
        print(f"✅ Quantum spectrum: {n_modes} modes, stability: {stability}")
        
    except Exception as e:
        print(f"❌ Error checking data integrity: {e}")
        success = False
    
    # Final summary
    print(f"\n{'='*60}")
    if success:
        print("🎉 PIPELINE TEST SUCCESSFUL!")
        print("All components working correctly.")
        print("\nGenerated outputs:")
        print("- quantum_outputs/: LQG physical states and observables")
        print("- warp_inputs/: Data formatted for warp-framework")
    else:
        print("💥 PIPELINE TEST FAILED!")
        print("Some components have issues.")
        sys.exit(1)
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
