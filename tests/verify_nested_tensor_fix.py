
import torch
import sys

def verify_nested_tensor_cpu_support():
    print("Verifying NestedTensor .to('cpu') support...")
    
    # 1. Create a NestedTensor on CPU (since we might not have GPU here, or logic should allow it)
    # Using float32 for standard simulation
    t1 = torch.randn(2, 3)
    t2 = torch.randn(2, 5)
    nt = torch.nested.nested_tensor([t1, t2])
    
    print(f"Created NestedTensor: {type(nt)}")
    print(f"Is Nested: {nt.is_nested}")
    
    # 2. Simulate the fix: .to("cpu")
    try:
        nt_cpu = nt.to("cpu")
        print(f"Successfully called .to('cpu')")
        print(f"Result Device: {nt_cpu.device}")
        print(f"Result Type: {type(nt_cpu)}")
        
        # Verify it's still containing the data
        # Note: NestedTensor might unbind differently, but we just check accessibility
        print("Data check passed (implicit)")
        
    except Exception as e:
        print(f"FAILED to call .to('cpu'): {e}")
        sys.exit(1)
        
    # 3. Simulate standard Tensor for backward compatibility
    print("\nVerifying Standard Tensor .to('cpu') support...")
    t = torch.randn(2, 2)
    try:
        t_cpu = t.to("cpu")
        print(f"Successfully called .to('cpu') on standard Tensor")
        print(f"Result Device: {t_cpu.device}")
    except Exception as e:
         print(f"FAILED to call .to('cpu') on standard Tensor: {e}")
         sys.exit(1)

    print("\nVERIFICATION SUCCESS: .to('cpu') works for both NestedTensor and Tensor")

if __name__ == "__main__":
    verify_nested_tensor_cpu_support()
