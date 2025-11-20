from dataclasses import dataclass
from typing import Dict

# GPU Specifications
# References: 
# 1. https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
# 2. https://www.techpowerup.com/gpu-specs
# 3. NVIDIA GPU Data Sheet Webpage
# Note: Unified Cache includes L1 cache and shared memory
GPUSpec = {
    "A100-40": {
        "fp64": 9.7, "tf64": 19.5, "fp32": 19.5, "tf32": 156, "fp16": 78, "tf16": 312, 
        "mem_bw": 1555, "pcie_bw": 64, "nvlink_bw": 600, 
        "base_clock": 765, "boost_clock": 1410, "mem_clock": 1215,
        "max_warps_sm": 64, "reg_size_sm": 256, "shmem_sm": 164, "uni_cache_sm": 192,
        "num_sm": 108
    },
    "A100-80": {
        "fp64": 9.7, "tf64": 19.5, "fp32": 19.5, "tf32": 156, "fp16": 78, "tf16": 312, 
        "mem_bw": 1935, "pcie_bw": 64, "nvlink_bw": 600, 
        "base_clock": 1065, "boost_clock": 1410, "mem_clock": 1512,
        "max_warps_sm": 64, "reg_size_sm": 256, "shmem_sm": 164, "uni_cache_sm": 192,
        "num_sm": 108, 
    },
    "A40": {
        "fp64": 0.58, "tf64": 0, "fp32": 37.4, "tf32": 74.8, "fp16": 37.4, "tf16": 149.7, 
        "mem_bw": 696, "pcie_bw": 64, "nvlink_bw": 112.5, 
        "base_clock": 1305, "boost_clock": 1740, "mem_clock": 1812, 
        "max_warps_sm": 48, "reg_size_sm": 256, "shmem_sm": 100, "uni_cache_sm": 128,
        "num_sm": 84
    },
    "H100-SXM": {
        "fp64": 34, "tf64": 67, "fp32": 67, "tf32": 989, "fp16": 133.8, "tf16": 1979, 
        "mem_bw": 3350, "pcie_bw": 128, "nvlink_bw": 900, 
        "base_clock": 1590, "boost_clock": 1980, "mem_clock": 1313, 
        "max_warps_sm": 64, "reg_size_sm": 256, "shmem_sm": 228, "uni_cache_sm": 256,
        "num_sm": 132
    },
    "H100-NVL": {
        "fp64": 30, "tf64": 60, "fp32": 60, "tf32": 835, "fp16": 133.8, "tf16": 1671, 
        "mem_bw": 3900, "pcie_bw": 128, "nvlink_bw": 600, 
        "base_clock": 1080, "boost_clock": 1785, "mem_clock": 1593,
        "max_warps_sm": 64, "reg_size_sm": 256, "shmem_sm": 228, "uni_cache_sm": 256,
        "num_sm": 114
    }
}


@dataclass
class GPU:
    """Encapsulates GPU specifications"""
    name: str
    specs: Dict[str, float]
    
    def __init__(self, gpu_name: str):
        self.name = gpu_name
        self.specs = GPUSpec.get(gpu_name)

    def get_name(self) -> str:
        return self.name

    def get_specs(self, key: str, default: float = 0.0) -> float:
        """Safe getter for spec values"""
        return self.specs.get(key, default)
    
    def __getitem__(self, key: str) -> float:
        """Allow dictionary-style access"""
        return self.specs[key]