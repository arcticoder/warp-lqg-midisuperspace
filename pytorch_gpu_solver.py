#!/usr/bin/env python3
"""
GPU acceleration fix for solve_constraint.py - using PyTorch instead of CuPy

This fixes the CUDA compatibility issue where CuPy (built for CUDA 12.0) 
doesn't work with CUDA 12.9, but PyTorch does.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import argparse
import json
import os
import time
from typing import Dict, Tuple, List, Optional

# GPU acceleration imports with PyTorch fallback
try:
    import torch
    if torch.cuda.is_available():
        GPU_AVAILABLE = True
        print(f"🚀 GPU acceleration available (PyTorch) - {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    else:
        GPU_AVAILABLE = False
        print("⚠️  PyTorch available but CUDA not detected")
except ImportError:
    GPU_AVAILABLE = False
    print("⚠️  PyTorch not available, GPU flag will be ignored")

def pytorch_sparse_eigsh(A_scipy, k=6, which='SM', tol=1e-10, maxiter=1000):
    """
    PyTorch-based sparse eigenvalue solver for symmetric matrices
    Converts scipy sparse matrix to PyTorch and solves on GPU
    """
    if not GPU_AVAILABLE:
        raise RuntimeError("GPU not available for PyTorch")
    
    print("   Converting sparse matrix to PyTorch format...")
    # Convert scipy sparse matrix to PyTorch sparse tensor
    A_coo = A_scipy.tocoo()
    indices = torch.stack([
        torch.from_numpy(A_coo.row).long(),
        torch.from_numpy(A_coo.col).long()
    ])
    values = torch.from_numpy(A_coo.data).float()
    A_sparse = torch.sparse_coo_tensor(indices, values, A_coo.shape, device='cuda')
    
    print(f"   Matrix transferred to GPU: {A_sparse.shape}")
    print(f"   GPU memory usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    # For small matrices, convert to dense and use torch.symeig
    if A_sparse.shape[0] <= 1000:
        print("   Using dense eigendecomposition on GPU...")
        A_dense = A_sparse.to_dense()
        eigenvals, eigenvecs = torch.linalg.eigh(A_dense)
        
        # Sort and select smallest magnitude eigenvalues
        idx = torch.argsort(torch.abs(eigenvals))[:k]
        eigenvals = eigenvals[idx]
        eigenvecs = eigenvecs[:, idx]
        
        # Convert back to numpy
        eigenvals_np = eigenvals.cpu().numpy()
        eigenvecs_np = eigenvecs.cpu().numpy()
        
    else:
        # For larger matrices, we need an iterative solver
        # PyTorch doesn't have sparse iterative eigensolvers built-in
        # So we'll use a simple power iteration method on GPU
        print("   Using power iteration on GPU for large matrix...")
        
        # Initialize random vectors on GPU
        n = A_sparse.shape[0]
        V = torch.randn(n, k, device='cuda', dtype=torch.float32)
        V = torch.nn.functional.normalize(V, dim=0)
        
        # Power iteration with deflation
        eigenvals_list = []
        eigenvecs_list = []
        
        for i in range(k):
            v = V[:, i].unsqueeze(1)
            
            # Power iteration
            for _ in range(maxiter // k):
                v_new = torch.sparse.mm(A_sparse, v)
                
                # Orthogonalize against previous eigenvectors
                for j in range(len(eigenvecs_list)):
                    proj = torch.dot(eigenvecs_list[j].squeeze(), v_new.squeeze())
                    v_new = v_new - proj * eigenvecs_list[j]
                
                v_new = torch.nn.functional.normalize(v_new, dim=0)
                v = v_new
            
            # Compute eigenvalue
            Av = torch.sparse.mm(A_sparse, v)
            eigenval = torch.dot(v.squeeze(), Av.squeeze())
            
            eigenvals_list.append(eigenval.cpu().numpy())
            eigenvecs_list.append(v.cpu().numpy())
        
        eigenvals_np = np.array(eigenvals_list)
        eigenvecs_np = np.column_stack(eigenvecs_list)
        
        # Sort by eigenvalue magnitude
        idx = np.argsort(np.abs(eigenvals_np))
        eigenvals_np = eigenvals_np[idx]
        eigenvecs_np = eigenvecs_np[:, idx]
    
    print(f"   GPU eigensolve completed. Final memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    return eigenvals_np, eigenvecs_np

def test_gpu_solver():
    """Test the PyTorch GPU eigenvalue solver"""
    print("🧪 Testing PyTorch GPU eigenvalue solver...")
    
    # Create a test symmetric matrix
    n = 500
    A = sp.random(n, n, density=0.01, format='csr')
    A = A + A.T  # Make symmetric
    A = A + sp.eye(n) * 0.1  # Make positive definite
    
    print(f"Test matrix: {A.shape}, nnz: {A.nnz}")
    
    if GPU_AVAILABLE:
        print("🚀 Testing GPU solver...")
        start_time = time.time()
        eigenvals_gpu, eigenvecs_gpu = pytorch_sparse_eigsh(A, k=5)
        gpu_time = time.time() - start_time
        print(f"GPU eigenvalues: {eigenvals_gpu}")
        print(f"GPU time: {gpu_time:.3f}s")
    
    print("🖥️ Testing CPU solver...")
    start_time = time.time()
    eigenvals_cpu, eigenvecs_cpu = spla.eigsh(A, k=5, which='SM')
    cpu_time = time.time() - start_time
    print(f"CPU eigenvalues: {eigenvals_cpu}")
    print(f"CPU time: {cpu_time:.3f}s")
    
    if GPU_AVAILABLE:
        speedup = cpu_time / gpu_time
        print(f"Speedup: {speedup:.2f}x")
        
        # Check accuracy
        error = np.mean(np.abs(np.sort(eigenvals_gpu) - np.sort(eigenvals_cpu)))
        print(f"Eigenvalue error: {error:.2e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test PyTorch GPU eigenvalue solver")
    parser.add_argument("--test", action="store_true", help="Run test")
    
    args = parser.parse_args()
    
    if args.test:
        test_gpu_solver()
    else:
        print("Use --test to run the GPU solver test")
        print(f"GPU available: {GPU_AVAILABLE}")
        if GPU_AVAILABLE:
            print(f"Device: {torch.cuda.get_device_name(0)}")
            print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
