#!/usr/bin/env python3
"""
Fixed GPU Solver using PyTorch instead of CuPy

This version replaces CuPy with PyTorch for GPU acceleration,
solving the CUDA compatibility issues we encountered.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import argparse
import json
import os
import time
from typing import Dict, Tuple, List, Optional

# GPU acceleration imports - PyTorch instead of CuPy
try:
    import torch
    import torch.sparse
    if torch.cuda.is_available():
        GPU_AVAILABLE = True
        print(f"🚀 GPU acceleration available (PyTorch) - {torch.cuda.get_device_name(0)}")
    else:
        GPU_AVAILABLE = False
        print("⚠️  PyTorch available but CUDA not detected")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️  PyTorch not available, GPU flag will be ignored")

# Keep the original imports for LQG components
try:
    from kinematical_hilbert import MidisuperspaceHilbert, load_lattice_from_reduced_variables
    from hamiltonian_constraint import HamiltonianConstraint
except ImportError as e:
    print(f"Warning: LQG modules not available: {e}")
    print("This is expected if running the solver in isolation")

def scipy_to_torch_sparse(scipy_matrix, device='cuda'):
    """Convert scipy sparse matrix to PyTorch sparse tensor"""
    if not isinstance(scipy_matrix, sp.coo_matrix):
        scipy_matrix = scipy_matrix.tocoo()
    
    indices = torch.stack([
        torch.from_numpy(scipy_matrix.row).long(),
        torch.from_numpy(scipy_matrix.col).long()
    ])
    values = torch.from_numpy(scipy_matrix.data).float()
    
    return torch.sparse_coo_tensor(
        indices, values, scipy_matrix.shape, 
        device=device, dtype=torch.float32
    )

def pytorch_sparse_eigsh(A_scipy, k=5, which='SM', tol=1e-10, maxiter=1000, device='cuda'):
    """
    PyTorch-based sparse eigenvalue solver for symmetric matrices
    """
    print(f"   Converting sparse matrix to PyTorch ({device})...")
    
    # Convert to PyTorch sparse tensor
    A_torch = scipy_to_torch_sparse(A_scipy, device)
    n = A_torch.shape[0]
    
    print(f"   Matrix: {A_torch.shape}, nnz: {A_torch._nnz()}")
    print(f"   GPU memory after transfer: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    if n <= 1000:
        # Small matrices: dense eigendecomposition
        print("   Using dense eigendecomposition on GPU...")
        A_dense = A_torch.to_dense()
        eigenvals, eigenvecs = torch.linalg.eigh(A_dense)
        
        # Sort by eigenvalue magnitude and take smallest k
        if which == 'SM':
            idx = torch.argsort(torch.abs(eigenvals))[:k]
        else:
            idx = torch.argsort(eigenvals)[:k]
            
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
    else:
        # Large matrices: iterative method
        print("   Using iterative power method on GPU...")
        
        # Initialize random vectors
        V = torch.randn(n, k, device=device, dtype=torch.float32)
        V = torch.nn.functional.normalize(V, dim=0)
        
        eigenvals_list = []
        eigenvecs_list = []
        
        for i in range(k):
            v = V[:, i].unsqueeze(1)
            prev_eigenval = 0
            
            # Power iteration for this eigenvalue
            for iter_count in range(maxiter // k):
                # Apply matrix
                v_new = torch.sparse.mm(A_torch, v)
                
                # Orthogonalize against previous eigenvectors
                for j in range(len(eigenvecs_list)):
                    proj = torch.dot(eigenvecs_list[j].squeeze(), v_new.squeeze())
                    v_new = v_new - proj * eigenvecs_list[j]
                
                # Normalize
                v_new = torch.nn.functional.normalize(v_new, dim=0)
                
                # Compute Rayleigh quotient (eigenvalue estimate)
                Av = torch.sparse.mm(A_torch, v_new)
                eigenval = torch.dot(v_new.squeeze(), Av.squeeze())
                
                # Check convergence
                if abs(eigenval - prev_eigenval) < tol:
                    break
                    
                v = v_new
                prev_eigenval = eigenval
            
            eigenvals_list.append(eigenval.cpu().numpy())
            eigenvecs_list.append(v.cpu().numpy())
        
        eigenvals = np.array(eigenvals_list)
        eigenvecs = np.column_stack(eigenvecs_list)
        
        # Sort by eigenvalue magnitude
        idx = np.argsort(np.abs(eigenvals))
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Convert back to torch tensors for consistency
        eigenvals = torch.from_numpy(eigenvals)
        eigenvecs = torch.from_numpy(eigenvecs)
    
    # Return numpy arrays (like scipy.sparse.linalg.eigsh)
    return eigenvals.cpu().numpy(), eigenvecs.cpu().numpy()

class ConstraintSolverPyTorch:
    """
    GPU-accelerated constraint solver using PyTorch
    Drop-in replacement for the CuPy version
    """
    
    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu and GPU_AVAILABLE
        self.eigenvalues = []
        self.physical_states = []
        
        if self.use_gpu:
            print(f"🔬 GPU solver initialized (PyTorch)")
            print(f"   Device: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB total")
        else:
            print("🖥️  CPU solver initialized")
    
    def solve_master_constraint_matrix(self, M: sp.spmatrix, n_states: int = 5, 
                                     tolerance: float = 1e-10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Solve eigenvalue problem for master constraint matrix
        """
        print(f"Master constraint matrix: {M.shape}")
        print(f"Non-zero elements: {M.nnz}")
        print(f"Sparsity: {M.nnz / (M.shape[0]**2):.6f}")
        
        if self.use_gpu:
            return self._solve_gpu(M, n_states, tolerance)
        else:
            return self._solve_cpu(M, n_states, tolerance)
    
    def _solve_gpu(self, M: sp.spmatrix, n_states: int, tolerance: float) -> Tuple[np.ndarray, np.ndarray]:
        """GPU-accelerated solving using PyTorch"""
        print("🚀 Using GPU acceleration (PyTorch)...")
        
        try:
            solve_start = time.time()
            eigenvals, eigenvecs = pytorch_sparse_eigsh(
                M, k=n_states, which='SM', tol=tolerance, device='cuda'
            )
            solve_time = time.time() - solve_start
            
            print(f"   🎯 GPU solve time: {solve_time:.2f}s")
            print(f"   Final GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            
            self.eigenvalues = eigenvals
            self.physical_states = eigenvecs
            
            return eigenvals, eigenvecs
            
        except Exception as e:
            print(f"   ❌ GPU solving failed: {e}")
            print("   Falling back to CPU...")
            return self._solve_cpu(M, n_states, tolerance)
    
    def _solve_cpu(self, M: sp.spmatrix, n_states: int, tolerance: float) -> Tuple[np.ndarray, np.ndarray]:
        """CPU solving using scipy"""
        print("🖥️  Using CPU solver...")
        solve_start = time.time()
        
        try:
            eigenvals, eigenvecs = spla.eigsh(
                M, k=n_states, which='SM', tol=tolerance, maxiter=1000
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
        
        return eigenvals, eigenvecs

def test_pytorch_solver():
    """Test the PyTorch GPU solver with a synthetic matrix"""
    print("🧪 Testing PyTorch GPU constraint solver...")
    
    # Create test symmetric sparse matrix
    n = 1000
    A = sp.random(n, n, density=0.01, format='csr')
    A = A + A.T  # Make symmetric
    A = A + sp.eye(n) * 0.1  # Make positive definite
    
    print(f"Test matrix: {A.shape}, nnz: {A.nnz}")
    
    # Test CPU solver
    print("\n--- CPU Test ---")
    solver_cpu = ConstraintSolverPyTorch(use_gpu=False)
    start_time = time.time()
    eigenvals_cpu, eigenvecs_cpu = solver_cpu.solve_master_constraint_matrix(A, n_states=5)
    cpu_time = time.time() - start_time
    print(f"CPU Results: {eigenvals_cpu}")
    
    if GPU_AVAILABLE:
        # Test GPU solver
        print("\n--- GPU Test ---")
        solver_gpu = ConstraintSolverPyTorch(use_gpu=True)
        start_time = time.time()
        eigenvals_gpu, eigenvecs_gpu = solver_gpu.solve_master_constraint_matrix(A, n_states=5)
        gpu_time = time.time() - start_time
        print(f"GPU Results: {eigenvals_gpu}")
        
        # Compare results
        error = np.mean(np.abs(eigenvals_cpu - eigenvals_gpu))
        speedup = cpu_time / gpu_time
        print(f"\nComparison:")
        print(f"  Eigenvalue error: {error:.2e}")
        print(f"  Speedup: {speedup:.2f}x")
        
        return speedup, error
    else:
        print("GPU test skipped - no CUDA available")
        return None, None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test PyTorch GPU constraint solver")
    parser.add_argument("--test", action="store_true", help="Run test")
    parser.add_argument("--gpu", action="store_true", help="Test GPU acceleration")
    
    args = parser.parse_args()
    
    if args.test:
        test_pytorch_solver()
    else:
        print("Use --test to run the constraint solver test")
        print(f"GPU available: {GPU_AVAILABLE}")
        if GPU_AVAILABLE:
            print(f"Device: {torch.cuda.get_device_name(0)}")
