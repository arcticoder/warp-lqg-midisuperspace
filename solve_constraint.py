#!/usr/bin/env python3
"""
solve_constraint.py

Unified LQG midisuperspace constraint solver (CPU or GPU).
Replaces the old:
  - pytorch_gpu_solver.py
  - solve_constraint_gpu.py
  - solve_constraint_pytorch.py
  - solve_constraint.py (original CPU-only)

Usage:
    python3 solve_constraint.py \
      --lattice examples/example_reduced_variables.json \
      --outdir quantum_inputs \
      [--use-gpu] [--device cuda:0] [--num-eigs 1]
"""

import argparse, json, os, sys, time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# Try to import PyTorch—if it’s not installed or CUDA isn’t available, GPU mode will be disabled
try:
    import torch
    torch.backends.cudnn.benchmark = True
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        torch_device = torch.device("cuda")
    else:
        torch_device = torch.device("cpu")
except ImportError:
    torch = None
    GPU_AVAILABLE = False
    torch_device = None


# -----------------------------------------------------------------------------
# 1) Load midisuperspace “reduced variables” (same as before)
# -----------------------------------------------------------------------------
def load_reduced_variables(lattice_json):
    """
    Reads a JSON file containing:
      {
        "r_grid": [...],
        "E_classical": { "E_x": [...], "E_phi": [...] },
        "K_classical": { "K_x": [...], "K_phi": [...] },
        "exotic_profile": { "scalar_field": [...] }
      }
    Returns: (r_grid, E_x, E_phi, K_x, K_phi, exotic_array)
    """
    data = json.load(open(lattice_json))
    r_grid  = np.array(data["r_grid"])
    E_x     = np.array(data["E_classical"]["E_x"])
    E_phi   = np.array(data["E_classical"]["E_phi"])
    K_x     = np.array(data["K_classical"]["K_x"])
    K_phi   = np.array(data["K_classical"]["K_phi"])
    exotic  = np.array(data["exotic_profile"]["scalar_field"])
    return r_grid, E_x, E_phi, K_x, K_phi, exotic


# -----------------------------------------------------------------------------
# 2) Build (toy) Hamiltonian constraint matrix on a very small truncated basis
# -----------------------------------------------------------------------------
def build_hamiltonian_matrix(r_grid, E_x, E_phi, K_x, K_phi, exotic,
                             mu_vals, nu_vals, mu_bar_scheme="auto"):
    """
    Assemble a sparse Hamiltonian Ĥ (gravity + matter) in a tiny midisuperspace basis.

    For demonstration, we just build a diagonal matrix of size basis_dim,
    where basis_dim = (len(mu_vals)*len(nu_vals))^Nsites (but truncated if too large).

    In a real implementation, you’d fill in holonomy loops, inverse‐triad regularizations, etc.
    """
    Nsites = len(r_grid)
    dim_per_site = len(mu_vals) * len(nu_vals)
    basis_dim = dim_per_site**Nsites
    if basis_dim > 10000:
        print(f"⚠ Truncating basis from {basis_dim} → 10000")
        basis_dim = 10000

    # Toy diagonal entries that depend on E_x[0], E_phi[-1], sum(exotic), etc.
    diag = np.zeros(basis_dim, dtype=np.complex128)
    for i in range(basis_dim):
        diag[i] = (E_x[0] + E_phi[-1] + exotic.sum()) * (1 + 0.01 * i)

    return sp.csr_matrix(np.diag(diag))


# -----------------------------------------------------------------------------
# 3) CPU solver path (SciPy)
# -----------------------------------------------------------------------------
def solve_constraint_cpu(H, num_eigs=1):
    """
    Solve H†H |ψ⟩ = 0 for the num_eigs lowest modes via ARPACK.
    Returns (eigenvalues, eigenvectors) of H†H near zero.
    """
    HdagH = H.getH().dot(H)
    vals, vecs = spla.eigs(HdagH, k=num_eigs, sigma=0.0)
    return vals, vecs


# -----------------------------------------------------------------------------
# 4) GPU solver path (PyTorch)
# -----------------------------------------------------------------------------
def solve_constraint_gpu_torch(H_cpu, num_eigs=1, device="cuda:0"):
    """
    Convert scipy.sparse matrix H_cpu → PyTorch sparse on GPU/device.
    Then compute HdagH = H^† H on GPU and do a dense eigh (for small dims)
    or a simple power iteration if dims > ~1000.

    Returns (eigenvals, eigenvecs) as NumPy arrays.
    """
    if not GPU_AVAILABLE or torch is None:
        raise RuntimeError("GPU mode requested but PyTorch/CUDA not available.")

    # Convert H_cpu to COO, then to PyTorch sparse_coo tensor on GPU
    H_coo = H_cpu.tocoo()
    indices = torch.LongTensor(np.vstack((H_coo.row, H_coo.col))).to(device)
    values = torch.complex(
        torch.from_numpy(H_coo.data.real).to(device),
        torch.from_numpy(H_coo.data.imag).to(device)
    )
    shape = H_coo.shape
    H_torch = torch.sparse_coo_tensor(indices, values, size=shape, device=device)

    # Build HdagH = H^† H
    Hdag = H_torch.transpose(0, 1).conj()
    HdagH_sparse = torch.sparse.mm(Hdag, H_torch)
    # For small dims, convert to dense
    n = shape[0]
    if n <= 1000:
        HdagH = HdagH_sparse.to_dense()
        # Use torch.linalg.eigh (Hermitian)
        eigvals_t, eigvecs_t = torch.linalg.eigh(HdagH)
        # Sort by |eigvals|
        idx = torch.argsort(torch.abs(eigvals_t))[:num_eigs]
        sel_vals = eigvals_t[idx].cpu().numpy()
        sel_vecs = eigvecs_t[:, idx].cpu().numpy()
        return sel_vals, sel_vecs
    else:
        # For larger dims, fallback to CPU SciPy
        print("⚠ GPU “large matrix” path not fully optimized. Falling back to CPU.")
        return solve_constraint_cpu(H_cpu, num_eigs=num_eigs)


# -----------------------------------------------------------------------------
# 5) Compute expectation ⟨E⟩ and ⟨T00⟩ from the “physical” state ψ
# -----------------------------------------------------------------------------
def compute_expectation_E_and_T00(r_grid, E_x, E_phi, exotic, psi, mu_vals, nu_vals):
    """
    Toy routine: assume <E^x(r_i)> ≈ average(mu_vals)*γ, etc.
    T00(r_i) ≈ |E_x − E_phi| + exotic[i]. 
    Returns two dicts: E_out, T00_out.
    """
    avg_mu = np.mean(mu_vals)
    avg_nu = np.mean(nu_vals)
    γ, ℓ2 = 1.0, 1.0  # set Immirzi and ℓ_Pl² = 1
    Ex_expect = [(γ * ℓ2 * avg_mu) for _ in r_grid]
    Ephi_expect = [(γ * ℓ2 * avg_nu) for _ in r_grid]
    T00_vals = [abs(Ex_expect[i] - Ephi_expect[i]) + exotic[i] for i in range(len(r_grid))]

    E_out = {"r": list(r_grid), "E_x": Ex_expect, "E_phi": Ephi_expect}
    T_out = {"r": list(r_grid), "T00": T00_vals}
    return E_out, T_out


# -----------------------------------------------------------------------------
# 6) Quantum‐corrected stability (toy Sturm–Liouville)
# -----------------------------------------------------------------------------
def quantum_stability(r_grid, E_x_q, E_phi_q, wormhole_ndjson, output_ndjson):
    """
    Read the classical wormhole solutions (NDJSON),
    reconstruct g_rr(r) ∼ E_phi_q/E_x_q, build a finite‐difference SL operator,
    and solve L ψ = ω² W ψ via SciPy for a few modes. Write NDJSON at output_ndjson.
    """
    import ndjson
    # Load classical wormhole NDJSON (to get “label” and r_throat)
    with open(wormhole_ndjson) as f:
        wh_data = ndjson.load(f)

    spectrum = []
    for entry in wh_data:
        b0 = entry.get("r_throat", r_grid[0])
        mask = r_grid >= b0
        r_sub = r_grid[mask]
        g_rr_sub = (E_phi_q[mask] / (E_x_q[mask] + 1e-12)).tolist()

        # Build a simple tridiagonal SL operator on r_sub
        n = len(r_sub)
        dr = r_sub[1] - r_sub[0]
        diag = np.zeros(n)
        offd = -np.ones(n - 1) / (dr**2)

        for i in range(n):
            p = 1.0
            diag[i] = 2 * p/(dr**2)

        from scipy.sparse import diags
        L = diags([offd, diag, offd], offsets=[-1,0,1], format="csr")
        W = diags(np.ones(n), offsets=0, format="csr")

        evals, _ = spla.eigs(L, M=W, k=3, sigma=0.0)
        for idx, ev in enumerate(evals):
            evv = float(ev.real)
            growth = float(np.sqrt(abs(evv))) if evv < 0 else 0.0
            spectrum.append({
                "label": f"{entry['label']}_qmode{idx}",
                "eigenvalue": evv,
                "growth_rate": growth,
                "stable": (evv >= 0),
                "method": "quantum-corrected"
            })

    # Write NDJSON
    with open(output_ndjson, "w") as f:
        writer = ndjson.writer(f)
        writer.writerows(spectrum)


# -----------------------------------------------------------------------------
# 7) Export quantum observables to JSON files
# -----------------------------------------------------------------------------
def export_quantum_observables(outdir, E_out, T00_out, spectrum_out):
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "expectation_E.json"), "w") as f:
        json.dump(E_out, f, indent=2)
    with open(os.path.join(outdir, "expectation_T00.json"), "w") as f:
        json.dump(T00_out, f, indent=2)
    with open(os.path.join(outdir, "quantum_spectrum.json"), "w") as f:
        json.dump(spectrum_out, f, indent=2)
    print(f"✓ Written quantum files to {outdir}/")


# -----------------------------------------------------------------------------
# 8) Single “main” to tie everything together
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Unified LQG solver (CPU/GPU).")
    parser.add_argument("--lattice",   required=True, help="JSON with midisuperspace data")
    parser.add_argument("--outdir",   required=True, help="Where to write quantum outputs")
    parser.add_argument("--mu-vals",  nargs="+", type=int, default=[-1,0,1], help="μ labels")
    parser.add_argument("--nu-vals",  nargs="+", type=int, default=[-1,0,1], help="ν labels")
    parser.add_argument("--use-gpu",  action="store_true", help="If set, use GPU/PyTorch")
    parser.add_argument("--device",   default="cuda:0", help="PyTorch device if --use-gpu")
    parser.add_argument("--num-eigs", type=int, default=1, help="How many near-zero modes")
    args = parser.parse_args()

    # 1) Load reduced variables
    r_grid, E_x, E_phi, K_x, K_phi, exotic = load_reduced_variables(args.lattice)

    # 2) Build Hamiltonian on small truncated basis
    H = build_hamiltonian_matrix(r_grid, E_x, E_phi, K_x, K_phi, exotic,
                                 mu_vals=args.mu_vals, nu_vals=args.nu_vals)

    # 3) Solve constraint (CPU or GPU)
    if args.use_gpu:
        if not GPU_AVAILABLE:
            print("❌ GPU requested but not available. Exiting.")
            sys.exit(1)
        print("🔷 Running GPU‐accelerated solver …")
        eigvals, eigvecs = solve_constraint_gpu_torch(H, num_eigs=args.num_eigs, device=args.device)
    else:
        print("⚙️  Running CPU‐only solver …")
        eigvals, eigvecs = solve_constraint_cpu(H, num_eigs=args.num_eigs)

    # 4) Choose the first near-zero eigenvector as physical state ψ
    psi = eigvecs[:, 0]

    # 5) Compute expectation values ⟨E⟩ and ⟨T00⟩
    E_out, T00_out = compute_expectation_E_and_T00(r_grid, E_x, E_phi, exotic, psi,
                                                  mu_vals=args.mu_vals, nu_vals=args.nu_vals)

    # 6) Compute a toy quantum‐corrected stability spectrum
    spectrum_out = []
    # Reconstruct arrays for E_x_q, E_phi_q
    E_x_q   = np.array(E_out["E_x"])
    E_phi_q = np.array(E_out["E_phi"])
    # Assume classical wormhole NDJSON is at a known relative path
    wormhole_ndjson = "warp-predictive-framework/outputs/wormhole_solutions.ndjson"
    quantum_spec_ndjson = os.path.join(args.outdir, "quantum_stability_spectrum.ndjson")
    quantum_stability(r_grid, E_x_q, E_phi_q, wormhole_ndjson, quantum_spec_ndjson)

    # 7) Write out E‐ and T00‐expectations plus quantum_spectrum.json
    export_quantum_observables(args.outdir, E_out, T00_out, spectrum_out)

    print("✅ All done. Quantum outputs in:", args.outdir)


if __name__ == "__main__":
    main()
        self.constraint = constraint
        self.physical_states = []
        self.eigenvalues = []
        self.use_gpu = use_gpu and GPU_AVAILABLE
        
        if self.use_gpu:
            print(f"🔬 GPU solver initialized (CuPy)")
            try:
                # Check GPU memory
                mempool = cp.get_default_memory_pool()
                print(f"   GPU memory: {mempool.free_bytes() / 1e9:.1f} GB free")
                print(f"   GPU device: {cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)['name'].decode()}")
            except Exception as e:
                print(f"   GPU info unavailable: {e}")        else:
            print("🖥️  CPU solver initialized")
    
    def solve_master_constraint(self, n_states: int = 5, 
                              tolerance: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve master constraint eigenvalue problem.
        
        Physical states correspond to near-zero eigenvalues.
        
        Args:
            n_states: Number of lowest eigenvalues to compute
            tolerance: Convergence tolerance for eigenvalues
            
        Returns:
            (eigenvalues, eigenvectors) with shape (n_states,) and (dim, n_states)
        """
        print("Computing master constraint operator...")
        start_time = time.time()
        M = self.constraint.master_constraint_operator()
        build_time = time.time() - start_time
        
        print(f"Master constraint matrix: {M.shape}")
        print(f"Non-zero elements: {M.nnz}")
        print(f"Sparsity: {M.nnz / (M.shape[0]**2):.6f}")
        print(f"Build time: {build_time:.2f}s")
        
        if self.use_gpu and GPU_AVAILABLE:
            return self._solve_gpu(M, n_states, tolerance)
        else:
            return self._solve_cpu(M, n_states, tolerance)
    
    def _solve_gpu(self, M_cpu: sp.spmatrix, n_states: int, tolerance: float) -> Tuple[np.ndarray, np.ndarray]:
        """GPU-accelerated eigenvalue solving using CuPy"""
        print("🚀 Using GPU acceleration (CuPy)...")
        
        try:
            # Transfer matrix to GPU
            print("   Transferring matrix to GPU...")
            transfer_start = time.time()
            
            M_csr = M_cpu.tocsr()
            M_gpu = cusp.csr_matrix(
                (cp.asarray(M_csr.data),
                 cp.asarray(M_csr.indices), 
                 cp.asarray(M_csr.indptr)),
                shape=M_csr.shape
            )
            
            transfer_time = time.time() - transfer_start
            print(f"   Transfer time: {transfer_time:.2f}s")
            
            # Check GPU memory usage
            mempool = cp.get_default_memory_pool()
            print(f"   GPU memory used: {mempool.used_bytes() / 1e9:.2f} GB")
            
            # Choose solver based on matrix size
            solve_start = time.time()
            
            if M_gpu.shape[0] <= 500:
                # Dense eigendecomposition for small matrices
                print("   Using dense GPU eigendecomposition...")
                M_dense = M_gpu.toarray()
                eigenvals, eigenvecs = cp.linalg.eigh(M_dense)
                
                # Sort by eigenvalue magnitude
                idx = cp.argsort(eigenvals)
                eigenvals = eigenvals[idx][:n_states]
                eigenvecs = eigenvecs[:, idx][:, :n_states]
                
            else:
                # Sparse iterative eigendecomposition
                print("   Using sparse GPU eigendecomposition...")
                try:
                    eigenvals, eigenvecs = cusla.eigsh(
                        M_gpu, k=n_states, which='SM',  # Smallest magnitude
                        tol=tolerance, maxiter=1000
                    )
                    
                    # Sort by eigenvalue
                    idx = cp.argsort(eigenvals)
                    eigenvals = eigenvals[idx]
                    eigenvecs = eigenvecs[:, idx]
                    
                except Exception as e:
                    print(f"   Sparse GPU solver failed: {e}")
                    print("   Falling back to dense GPU solver...")
                    M_dense = M_gpu.toarray()
                    eigenvals, eigenvecs = cp.linalg.eigh(M_dense)
                    idx = cp.argsort(eigenvals)
                    eigenvals = eigenvals[idx][:n_states]
                    eigenvecs = eigenvecs[:, idx][:, :n_states]
            
            solve_time = time.time() - solve_start
            print(f"   Solve time: {solve_time:.2f}s")
            
            # Transfer results back to CPU
            print("   Transferring results to CPU...")
            eigenvals_cpu = cp.asnumpy(eigenvals)
            eigenvecs_cpu = cp.asnumpy(eigenvecs)
            
            # Store results
            self.eigenvalues = eigenvals_cpu
            self.physical_states = eigenvecs_cpu
            
            total_time = time.time() - solve_start + transfer_time
            print(f"   🎯 Total GPU time: {total_time:.2f}s")
            
            return eigenvals_cpu, eigenvecs_cpu
            
        except Exception as e:
            print(f"   ❌ GPU solving failed: {e}")
            print("   Falling back to CPU...")
            return self._solve_cpu(M_cpu, n_states, tolerance)
    
    def _solve_cpu(self, M: sp.spmatrix, n_states: int, tolerance: float) -> Tuple[np.ndarray, np.ndarray]:
        """CPU eigenvalue solving (original implementation)"""
        print("🖥️  Using CPU solver...")
        solve_start = time.time()
        
        if M.shape[0] <= 100:
            # Small matrices: use dense eigendecomposition
            print("   Using dense eigendecomposition...")
            M_dense = M.toarray()
            eigenvals, eigenvecs = np.linalg.eigh(M_dense)
            
            # Sort by eigenvalue magnitude
            idx = np.argsort(eigenvals)
            eigenvals = eigenvals[idx][:n_states]
            eigenvecs = eigenvecs[:, idx][:, :n_states]
            
        else:
            # Large matrices: use sparse iterative solver
            print("   Using sparse eigendecomposition...")
            
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
                print(f"   ARPACK convergence warning: {e}")
                eigenvals = e.eigenvalues
                eigenvecs = e.eigenvectors
                
                if eigenvals is not None:
                    idx = np.argsort(eigenvals)
                    eigenvals = eigenvals[idx]
                    eigenvecs = eigenvecs[:, idx]
        
        solve_time = time.time() - solve_start
        print(f"   CPU solve time: {solve_time:.2f}s")
        
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
    parser.add_argument("--gpu", action="store_true",
                       help="Use GPU acceleration (requires CuPy or PyTorch)")
    parser.add_argument("--backend", type=str, default="auto",
                       choices=["auto", "cupy", "torch", "cpu"],
                       help="GPU backend to use")
    parser.add_argument("--benchmark", action="store_true",
                       help="Run benchmark comparing all available backends")
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
