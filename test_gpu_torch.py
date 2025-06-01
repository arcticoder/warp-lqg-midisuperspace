#!/usr/bin/env python3
"""
GPU test using PyTorch instead of CuPy
"""

import numpy as np
import time
import argparse

try:
    import torch
    GPU_AVAILABLE = torch.cuda.is_available()
    if GPU_AVAILABLE:
        print(f"✅ PyTorch CUDA available - Device: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ PyTorch CUDA not available")
except ImportError:
    GPU_AVAILABLE = False
    print("❌ PyTorch not available")

def test_gpu_utilization_torch(use_gpu=False, matrix_size=2000):
    """Test that should visibly use GPU resources with PyTorch"""
    print(f"\n🔬 Testing {'GPU' if use_gpu else 'CPU'} computation...")
    print(f"Matrix size: {matrix_size}x{matrix_size}")
    
    device = torch.device("cuda" if use_gpu and GPU_AVAILABLE else "cpu")
    print(f"🚀 Using device: {device}")
    
    # Create large matrices
    start_time = time.time()
    A = torch.rand(matrix_size, matrix_size, dtype=torch.float32, device=device)
    B = torch.rand(matrix_size, matrix_size, dtype=torch.float32, device=device)
    
    if use_gpu and GPU_AVAILABLE:
        print(f"   GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"   GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    # Intensive computation that should show up in nvidia-smi
    for i in range(10):
        C = torch.mm(A, B)  # Matrix multiplication
        A = C + torch.rand(matrix_size, matrix_size, dtype=torch.float32, device=device)
        
        if use_gpu and GPU_AVAILABLE:
            torch.cuda.synchronize()  # Ensure computation completes
        
        print(f"   Iteration {i+1}/10 completed")
        time.sleep(0.5)  # Give time to check nvidia-smi
        
    end_time = time.time()
    result = torch.sum(C).item()
    
    print(f"⏱️  Computation time: {end_time - start_time:.2f}s")
    print(f"📊 Result sum: {result:.6e}")
    
    if use_gpu and GPU_AVAILABLE:
        print(f"   Final GPU memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    
    print("\n🔍 Check 'nvidia-smi' in another terminal to see GPU utilization!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test GPU utilization with PyTorch")
    parser.add_argument("--gpu", action="store_true", help="Use GPU acceleration")
    parser.add_argument("--size", type=int, default=2000, help="Matrix size")
    
    args = parser.parse_args()
    
    if args.gpu and not GPU_AVAILABLE:
        print("❌ GPU requested but PyTorch CUDA not available")
        exit(1)
    
    test_gpu_utilization_torch(use_gpu=args.gpu, matrix_size=args.size)
