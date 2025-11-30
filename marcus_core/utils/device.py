"""
Device Utilities
================

Device detection and management for PyTorch.
"""

from __future__ import annotations

from typing import Dict, Any, Optional
import logging

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


logger = logging.getLogger(__name__)

# Global device setting
_current_device: Optional[str] = None


def is_cuda_available() -> bool:
    """Check if CUDA is available."""
    if not HAS_TORCH:
        return False
    return torch.cuda.is_available()


def is_mps_available() -> bool:
    """Check if MPS (Apple Silicon) is available."""
    if not HAS_TORCH:
        return False
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def get_device(prefer: Optional[str] = None) -> str:
    """
    Get the best available device.
    
    Priority: prefer > CUDA > MPS > CPU
    
    Args:
        prefer: Preferred device (cuda, mps, cpu)
    
    Returns:
        Device string
    """
    global _current_device
    
    if _current_device is not None:
        return _current_device
    
    if prefer is not None:
        prefer = prefer.lower()
        
        if prefer == "cuda" and is_cuda_available():
            return "cuda"
        elif prefer == "mps" and is_mps_available():
            return "mps"
        elif prefer == "cpu":
            return "cpu"
        else:
            logger.warning(f"Preferred device {prefer} not available")
    
    if is_cuda_available():
        return "cuda"
    elif is_mps_available():
        return "mps"
    else:
        return "cpu"


def set_device(device: str) -> None:
    """
    Set the global device.
    
    Args:
        device: Device string (cuda, mps, cpu)
    """
    global _current_device
    _current_device = device
    logger.info(f"Device set to: {device}")


def get_device_info() -> Dict[str, Any]:
    """
    Get detailed device information.
    
    Returns:
        Dictionary with device info
    """
    info = {
        "pytorch_available": HAS_TORCH,
        "cuda_available": is_cuda_available(),
        "mps_available": is_mps_available(),
        "current_device": get_device(),
    }
    
    if not HAS_TORCH:
        return info
    
    info["pytorch_version"] = torch.__version__
    
    if is_cuda_available():
        info["cuda_version"] = torch.version.cuda
        info["cudnn_version"] = torch.backends.cudnn.version()
        info["gpu_count"] = torch.cuda.device_count()
        
        gpu_info = []
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            gpu_info.append({
                "name": props.name,
                "total_memory_gb": props.total_memory / (1024 ** 3),
                "compute_capability": f"{props.major}.{props.minor}",
                "multi_processor_count": props.multi_processor_count,
            })
        info["gpus"] = gpu_info
    
    if is_mps_available():
        info["mps_device"] = "Apple Silicon GPU"
    
    return info


def get_memory_info(device: Optional[str] = None) -> Dict[str, float]:
    """
    Get memory usage information.
    
    Args:
        device: Device to query (default: current device)
    
    Returns:
        Dictionary with memory info (in GB)
    """
    if not HAS_TORCH:
        return {}
    
    device = device or get_device()
    
    if device == "cuda" and is_cuda_available():
        return {
            "allocated": torch.cuda.memory_allocated() / (1024 ** 3),
            "cached": torch.cuda.memory_reserved() / (1024 ** 3),
            "max_allocated": torch.cuda.max_memory_allocated() / (1024 ** 3),
        }
    
    return {}


def clear_memory(device: Optional[str] = None) -> None:
    """
    Clear memory cache.
    
    Args:
        device: Device to clear (default: all devices)
    """
    if not HAS_TORCH:
        return
    
    device = device or get_device()
    
    if device == "cuda" and is_cuda_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        logger.info("Cleared CUDA memory cache")
    
    if device == "mps" and is_mps_available():
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
            logger.info("Cleared MPS memory cache")


def to_device(
    tensor_or_model: Any,
    device: Optional[str] = None,
) -> Any:
    """
    Move tensor or model to device.
    
    Args:
        tensor_or_model: PyTorch tensor or model
        device: Target device
    
    Returns:
        Tensor/model on device
    """
    if not HAS_TORCH:
        return tensor_or_model
    
    device = device or get_device()
    return tensor_or_model.to(device)


def benchmark_device(
    device: Optional[str] = None,
    size: int = 1000,
    iterations: int = 100,
) -> Dict[str, float]:
    """
    Benchmark device performance.
    
    Args:
        device: Device to benchmark
        size: Matrix size
        iterations: Number of iterations
    
    Returns:
        Benchmark results (ops/sec)
    """
    if not HAS_TORCH:
        return {}
    
    device = device or get_device()
    
    import time
    
    # Create random matrices
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # Warmup
    for _ in range(10):
        c = torch.mm(a, b)
    
    # Synchronize if on GPU
    if device == "cuda":
        torch.cuda.synchronize()
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        c = torch.mm(a, b)
    
    if device == "cuda":
        torch.cuda.synchronize()
    
    elapsed = time.perf_counter() - start
    
    ops_per_sec = iterations / elapsed
    gflops = (2 * size ** 3 * iterations) / elapsed / 1e9
    
    return {
        "device": device,
        "matrix_size": size,
        "iterations": iterations,
        "elapsed_sec": elapsed,
        "ops_per_sec": ops_per_sec,
        "gflops": gflops,
    }
