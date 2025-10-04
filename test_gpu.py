# ==============================================================================
# PyTorch CUDA Availability Check
# This uses a different framework to determine if CUDA is accessible.
# ==============================================================================
import torch

print("-" * 60)
print("     PyTorch CUDA Compatibility Status")
print("-" * 60)

# PyTorch command to check for GPU
if torch.cuda.is_available():
    print(f"Result: Num GPUs Available: {torch.cuda.device_count()}")
    print("\n✅ SUCCESS: GPU FOUND!")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
else:
    print(f"Result: Num GPUs Available: 0")
    print("\n❌ FAILURE: PyTorch CANNOT DETECT GPU.")
    print("This confirms a fundamental system-level driver or PATH error.")

print("-" * 60)
