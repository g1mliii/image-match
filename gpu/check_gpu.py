import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"Device count: {torch.cuda.device_count()}")
    print(f"Device name: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"Device memory: {props.total_memory / 1024**3:.2f} GB")
    print(f"Compute capability: {props.major}.{props.minor}")
else:
    print("No GPU detected")
