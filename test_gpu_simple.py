#!/usr/bin/env python3
"""
Simple GPU test to verify CuPy actually uses the GPU
"""

import numpy as np
import time
import argparse

try:
    import cupy as cp
    GPU_AVAILABLE = True
    print("✅ CuPy available")
except ImportError:
    GPU_AVAILABLE = False
    print("❌ CuPy not available")

def test_gpu_utilization(use_gpu=False, matrix_size=2000):
    """Test that should visibly use GPU resources"""
    print(f"\n🔬 Testing {'GPU' if use_gpu else 'CPU'} computation...")
    print(f"Matrix size: {matrix_size}x{matrix_size}")
    
    if use_gpu and GPU_AVAILABLE:
        print("🚀 Using GPU (CuPy)")
        # Create large matrices on GPU
        start_time = time.time()
        A = cp.random.rand(matrix_size, matrix_size, dtype=cp.float32)
        B = cp.random.rand(matrix_size, matrix_size, dtype=cp.float32)
        
        # Force GPU memory allocation
        cp.cuda.Device().synchronize()
        
        print(f"   GPU memory allocated: {cp.get_default_memory_pool().used_bytes() / 1e9:.2f} GB")
        
        # Intensive computation that should show up in nvidia-smi
        for i in range(10):
            C = cp.dot(A, B)  # Matrix multiplication
            A = C + cp.random.rand(matrix_size, matrix_size, dtype=cp.float32)
            cp.cuda.Device().synchronize()  # Ensure computation completes
            print(f"   Iteration {i+1}/10 completed")
            time.sleep(0.5)  # Give time to check nvidia-smi
            
        end_time = time.time()
        result = cp.sum(C).get()  # Transfer result back to CPU
        
    else:
        print("🖥️ Using CPU (NumPy)")
        start_time = time.time()
        A = np.random.rand(matrix_size, matrix_size).astype(np.float32)
        B = np.random.rand(matrix_size, matrix_size).astype(np.float32)
        
        # Intensive computation
        for i in range(10):
            C = np.dot(A, B)
            A = C + np.random.rand(matrix_size, matrix_size).astype(np.float32)
            print(f"   Iteration {i+1}/10 completed")
            time.sleep(0.5)
            
        end_time = time.time()
        result = np.sum(C)
    
    print(f"⏱️  Computation time: {end_time - start_time:.2f}s")
    print(f"📊 Result sum: {result:.6e}")
    print("\n🔍 Check 'nvidia-smi' in another terminal to see GPU utilization!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GPU utilization")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")
    parser.add_argument("--size", type=int, default=2000, help="Matrix size")
    
    args = parser.parse_args()
    
    if args.gpu and not GPU_AVAILABLE:
        print("❌ GPU requested but CuPy not available")
        exit(1)
    
    test_gpu_utilization(use_gpu=args.gpu, matrix_size=args.size)
