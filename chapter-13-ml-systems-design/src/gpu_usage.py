# gpu_usage.py - Practical GPU usage patterns in Python
import numpy as np
import time

# ── Check GPU availability ────────────────────────────────────────
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
    MPS_AVAILABLE  = torch.backends.mps.is_available()  # Apple Silicon
    if CUDA_AVAILABLE:
        device = torch.device('cuda')
        print(f'GPU available: {torch.cuda.get_device_name(0)}')
        print(f'GPU memory:    {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
        print(f'CUDA version:  {torch.version.cuda}')
    elif MPS_AVAILABLE:
        device = torch.device('mps')
        print('Apple Silicon GPU (MPS) available')
    else:
        device = torch.device('cpu')
        print('No GPU available, using CPU')
    print(f'PyTorch device: {device}')

    # ── CPU vs GPU benchmark ──────────────────────────────────────
    print('\n=== CPU vs GPU Matrix Multiplication Benchmark ===')
    sizes = [1000, 5000, 10000]
    for n in sizes:
        A_cpu = torch.randn(n, n)
        B_cpu = torch.randn(n, n)

        # CPU
        t0 = time.time()
        C_cpu = torch.mm(A_cpu, B_cpu)
        cpu_time = time.time() - t0

        # GPU
        if CUDA_AVAILABLE or MPS_AVAILABLE:
            A_gpu = A_cpu.to(device)
            B_gpu = B_cpu.to(device)
            torch.cuda.synchronize() if CUDA_AVAILABLE else None
            t0 = time.time()
            C_gpu = torch.mm(A_gpu, B_gpu)
            torch.cuda.synchronize() if CUDA_AVAILABLE else None
            gpu_time = time.time() - t0
            speedup = cpu_time / gpu_time
            print(f'  {n}×{n}: CPU={cpu_time*1000:.1f}ms  GPU={gpu_time*1000:.1f}ms  Speedup={speedup:.1f}x')
        else:
            print(f'  {n}×{n}: CPU={cpu_time*1000:.1f}ms  (no GPU)')

    # ── GPU memory management ─────────────────────────────────────
    if CUDA_AVAILABLE:
        print('\n=== GPU Memory Management ===')
        print(f'Allocated: {torch.cuda.memory_allocated()/1e9:.3f} GB')
        print(f'Reserved:  {torch.cuda.memory_reserved()/1e9:.3f} GB')

        # Best practice: use float16 for 2x memory savings
        large_float32 = torch.randn(10000, 10000, dtype=torch.float32)
        large_float16 = torch.randn(10000, 10000, dtype=torch.float16)
        print(f'float32 size: {large_float32.element_size()*large_float32.numel()/1e9:.3f} GB')
        print(f'float16 size: {large_float16.element_size()*large_float16.numel()/1e9:.3f} GB')
        print('float16 uses 50% less memory - critical for large models')

except ImportError:
    print('PyTorch not installed. GPU features unavailable.')

# ── When to use GPU/TPU vs CPU ───────────────────────────────────
print('\n=== Hardware Selection Guide ===')
guide = {
    'Use CPU for': [
        'Classical ML (sklearn, XGBoost, LightGBM) - already optimised for CPU',
        'Small datasets (< 100MB) - GPU overhead outweighs benefit',
        'Feature engineering with Pandas/Spark',
        'Model serving for sklearn/ONNX models (fast CPU inference)',
    ],
    'Use GPU for': [
        'Deep learning training (any framework)',
        'Image/video/audio processing',
        'Large language model fine-tuning',
        'High-throughput neural network inference',
        'cuML: GPU-accelerated sklearn equivalents',
    ],
    'Use TPU for': [
        'Training on JAX or TensorFlow at very large scale',
        'LLM pre-training (Google uses TPUs for Gemini/PaLM)',
        'When on Google Cloud and running JAX workloads',
    ],
}
for category, items in guide.items():
    print(f'\n{category}:')
    for item in items:
        print(f'  • {item}')
