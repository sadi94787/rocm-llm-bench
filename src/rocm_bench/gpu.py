"""GPU detection and info utilities for AMD ROCm devices."""

import torch


def is_rocm_available() -> bool:
    """Check if ROCm backend is available."""
    return torch.version.hip is not None


def get_rocm_version() -> str | None:
    """Get ROCm version string."""
    return torch.version.hip if is_rocm_available() else None


def get_gpu_count() -> int:
    """Get number of available GPUs."""
    return torch.cuda.device_count() if torch.cuda.is_available() else 0


def get_gpu_info() -> list[dict]:
    """Get detailed info for all available GPUs."""
    gpus = []
    for i in range(get_gpu_count()):
        props = torch.cuda.get_device_properties(i)
        gpus.append({
            "index": i,
            "name": props.name,
            "total_memory_gb": round(props.total_mem / 1e9, 2),
            "major": props.major,
            "minor": props.minor,
            "multi_processor_count": props.multi_processor_count,
        })
    return gpus


def get_system_info() -> dict:
    """Get full system info for benchmark reports."""
    return {
        "rocm_available": is_rocm_available(),
        "rocm_version": get_rocm_version(),
        "torch_version": torch.__version__,
        "gpu_count": get_gpu_count(),
        "gpus": get_gpu_info(),
        "cuda_version": torch.version.cuda,
    }


def print_system_info():
    """Print formatted system info."""
    info = get_system_info()
    print(f"ROCm:     {info['rocm_version'] or 'N/A'}")
    print(f"PyTorch:  {info['torch_version']}")
    print(f"CUDA/HIP: {info['cuda_version']}")
    print(f"GPUs:     {info['gpu_count']}")
    for gpu in info["gpus"]:
        print(f"  [{gpu['index']}] {gpu['name']} ({gpu['total_memory_gb']} GB)")
