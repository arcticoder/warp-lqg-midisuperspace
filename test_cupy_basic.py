#!/usr/bin/env python3
"""
Quick test to see if CuPy actually works with CUDA 12.x
"""

def test_cupy_basic():
    try:
        import cupy as cp
        print(f"✅ CuPy import successful: {cp.__version__}")
        
        # Test basic GPU operations
        print("Testing basic GPU operations...")
        a = cp.array([1, 2, 3, 4, 5])
        b = cp.array([5, 4, 3, 2, 1])
        c = a + b
        result = cp.sum(c)
        print(f"✅ Basic operations work: {result}")
        
        # Test memory operations
        print("Testing GPU memory...")
        large_array = cp.random.rand(1000, 1000)
        print(f"✅ Large array creation works: {large_array.shape}")
        
        # Test random operations (this was failing before)
        print("Testing random operations...")
        random_matrix = cp.random.rand(100, 100)
        print(f"✅ Random operations work: {random_matrix.mean()}")
        
        print("🎉 CuPy is working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ CuPy test failed: {e}")
        return False

if __name__ == "__main__":
    test_cupy_basic()
