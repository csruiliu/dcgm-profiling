import pandas as pd
import argparse
import re
import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# For reference: 
# 1. https://images.nvidia.com/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf
# 2. https://www.techpowerup.com/gpu-specs
# 3. NVIDIA GPU Data Sheet Webpage
# Unified Cache includes include L1 cache and shared memory
GPUS = {
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
    },
}


# Parametric GPU specs for research models
PARAMETRIC_GPUS = {
    "R100": {"multipliers": {"fp64": 3.0, "tf64": 3.0, "fp32": 6.0, "tf32": 6.0, 
                             "fp16": 3.0, "tf16": 3.0, "mem_bw": 8.0, "pcie_bw": 25.0, 
                             "nvlink_bw": 6.0}, "alpha_gpu": 4.0, "alpha_cpu": 3.0},
    "R100-UNI": {"multipliers": {"fp64": 4.0, "tf64": 4.0, "fp32": 8.0, "tf32": 8.0, 
                                 "fp16": 4.0, "tf16": 4.0, "mem_bw": 1.5, "pcie_bw": 25.0, 
                                 "nvlink_bw": 6.0}, "alpha_gpu": 4.0, "alpha_cpu": 3.0},
    "GPU-M-IO-A-H14": {"multipliers": {"fp64": 1.0, "tf64": 1.0, "fp32": 1.0, "tf32": 1.0, 
                                       "fp16": 1.0, "tf16": 1.0, "mem_bw": 4.0, "pcie_bw": 4.0, 
                                       "nvlink_bw": 4.0}, "alpha_gpu": 1.0, "alpha_cpu": 3.0},
    "GPU-F-IO-A-H14": {"multipliers": {"fp64": 4.0, "tf64": 4.0, "fp32": 4.0, "tf32": 4.0, 
                                       "fp16": 4.0, "tf16": 4.0, "mem_bw": 1.0, "pcie_bw": 4.0, 
                                       "nvlink_bw": 4.0}, "alpha_gpu": 4.0, "alpha_cpu": 3.0},
    "GPU-M-IO-A-H22": {"multipliers": {"fp64": 2.0, "tf64": 2.0, "fp32": 2.0, "tf32": 2.0, 
                                       "fp16": 2.0, "tf16": 2.0, "mem_bw": 2.0, "pcie_bw": 4.0, 
                                       "nvlink_bw": 4.0}, "alpha_gpu": 2.0, "alpha_cpu": 3.0},
    "GPU-F-IO-A-H22": {"multipliers": {"fp64": 2.0, "tf64": 2.0, "fp32": 2.0, "tf32": 2.0, 
                                       "fp16": 2.0, "tf16": 2.0, "mem_bw": 2.0, "pcie_bw": 4.0, 
                                       "nvlink_bw": 4.0}, "alpha_gpu": 2.0, "alpha_cpu": 3.0},
    "GPU-M-IO-A-H24": {"multipliers": {"fp64": 2.0, "tf64": 2.0, "fp32": 2.0, "tf32": 2.0, 
                                       "fp16": 2.0, "tf16": 2.0, "mem_bw": 4.0, "pcie_bw": 4.0, 
                                       "nvlink_bw": 4.0}, "alpha_gpu": 2.0, "alpha_cpu": 3.0},
    "GPU-F-IO-A-H24": {"multipliers": {"fp64": 4.0, "tf64": 4.0, "fp32": 4.0, "tf32": 4.0, 
                                       "fp16": 4.0, "tf16": 4.0, "mem_bw": 2.0, "pcie_bw": 4.0, 
                                       "nvlink_bw": 4.0}, "alpha_gpu": 4.0, "alpha_cpu": 3.0},
}


# Initialize parametric GPUs based on A100-40 baseline
BASE_GPU = "A100-40"
for gpu_name, config in PARAMETRIC_GPUS.items():
    base_specs = GPUS[BASE_GPU]
    new_specs = {}
    for key, value in base_specs.items():
        multiplier = config.get("multipliers").get(key, 1.0)
        new_specs[key] = value * multiplier
    new_specs.update({k: v for k, v in config.items() if k != "multipliers"})
    GPUS[gpu_name] = new_specs


# Metric mappings
METRIC_MAPPINGS = {
    'ref': {'TENSO': 'ref_tf64', 'FP64A': 'ref_fp64', 'FP32A': 'ref_fp32', 'FP16A': 'ref_fp16'},
    'target': {'TENSO': 'target_tf64', 'FP64A': 'target_fp64', 'FP32A': 'target_fp32', 'FP16A': 'target_fp16'}
}

PRECISION_MAPPINGS = {
    'ref': {'tf64': 'ref_tf64', 'tf32': 'ref_tf32', 'tf16': 'ref_tf16'},
    'target': {'tf64': 'target_tf64', 'tf32': 'target_tf32', 'tf16': 'target_tf16'}
}


@dataclass
class TimeSlice:
    """Container for time-based metrics"""
    start_idx: int = 0
    end_idx: Optional[int] = None
    
    def slice_list(self, data: List) -> List:
        """Apply slicing to a list"""
        return data[self.start_idx:self.end_idx] if self.end_idx else data[self.start_idx:]


class MetricsProcessor:
    """Handles metrics file processing"""
    GPU_PATTERN = re.compile(r'^GPU \d+\s')
    HEADER_PATTERN = re.compile(r'^#Entity')

    @staticmethod
    def is_float(value: str) -> bool:
        """Check if a string can be converted to float"""
        try:
            float(value)
            return True
        except ValueError:
            return False

    @classmethod
    def process_file(cls, file_path: str, metric_names: List[str]) -> pd.DataFrame:
        """Read and process the metrics file"""
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        header_columns, metric_indices = cls._parse_header(lines, metric_names)
        gpu_data = cls._extract_gpu_data(lines, metric_indices, len(header_columns))
        
        return pd.DataFrame(gpu_data, columns=metric_names)

    @classmethod
    def _parse_header(cls, lines: List[str], metric_names: List[str]) -> Tuple[List[str], List[int]]:
        """Parse header and find metric column indices"""
        for line in lines:
            if cls.HEADER_PATTERN.match(line):
                header_columns = [col.strip() for col in re.split(r'\s{2,}', line.strip())]
                metric_indices = cls._get_metric_indices(header_columns, metric_names)
                return header_columns, metric_indices
        raise ValueError("Could not find header line in the data file")

    @staticmethod
    def _get_metric_indices(header_columns: List[str], metric_names: List[str]) -> List[int]:
        """Map requested metrics to their column indices"""
        metric_indices = []
        for metric in metric_names:
            if metric not in header_columns:
                raise ValueError(
                    f"Metric '{metric}' not found in data file. "
                    f"Available metrics: {header_columns[1:]}"
                )
            metric_indices.append(header_columns.index(metric) - 1)
        return metric_indices

    @classmethod
    def _extract_gpu_data(cls, lines: List[str], metric_indices: List[int], num_columns: int) -> List[List[float]]:
        """Extract GPU data from lines"""
        gpu_data = []
        for line in lines:
            if cls.HEADER_PATTERN.match(line):
                continue
            if cls.GPU_PATTERN.match(line):
                values = re.split(r'\s{3,}', line.strip())
                numeric_values = [float(v) for v in values if cls.is_float(v)]
                
                if len(numeric_values) >= num_columns - 1:
                    selected_values = [numeric_values[i] for i in metric_indices]
                    gpu_data.append(selected_values)
                else:
                    print(f"Warning: Line has insufficient data columns: {line.strip()}")
        return gpu_data


class PerformanceProfiler:
    """Unified performance profiler for analysis, modeling, and prediction"""
    
    def __init__(self, sample_interval_ms: float):
        self.sample_intv = sample_interval_ms / 1000
    
    def _get_gpu_spec(self, gpu_arch: str) -> Dict[str, float]:
        """Get GPU specifications"""
        if gpu_arch not in GPUS:
            raise ValueError(f"GPU architecture '{gpu_arch}' not found in GPU_SPECS")
        return GPUS[gpu_arch]
    
    def _bound_smocc(self, sm_occ_ref: float, ref_gpu: Dict[str, float], tgt_gpu: Dict[str, float]) -> tuple[float, float]:
        """Bound target GPU using reference GPU specification"""
        smocc_tgt_lower = sm_occ_ref * min(
            tgt_gpu.get("max_warps_sm") / ref_gpu.get("max_warps_sm"),
            tgt_gpu.get("reg_size_sm") / ref_gpu.get("reg_size_sm"),
            tgt_gpu.get("shmem_sm") / ref_gpu.get("shmem_sm")
        )
        smocc_tgt_upper = sm_occ_ref * max(
            tgt_gpu.get("max_warps_sm") / ref_gpu.get("max_warps_sm"),
            tgt_gpu.get("reg_size_sm") / ref_gpu.get("reg_size_sm"),
            tgt_gpu.get("shmem_sm") / ref_gpu.get("shmem_sm")
        )
    
        return smocc_tgt_lower, smocc_tgt_upper

    def calc_time_components(self, row, metrics: List[str], gpu: Dict[str, float]) -> Dict[str, float]:
        """Calculate various time components for a single row"""
        mv = {metric: getattr(row, metric) for metric in metrics}
        
        t_flop = self.sample_intv * sum(mv.get(m) for m in ['TENSO', 'FP64A', 'FP32A', 'FP16A'])
        t_dram = self.sample_intv * mv.get('DRAMA')
   
        gract = mv.get('GRACT')
        t_kernel = self.sample_intv * gract         
        t_othernode = max(self.sample_intv * (1 - gract), 0)
        
        return {
            't_flop': t_flop,
            't_dram': t_dram,
            't_kernel': t_kernel,
            't_othernode': t_othernode
        }
    
    def get_time_slice(self, overall_runtime_ms: float, start_ts: Optional[float], end_ts: Optional[float], data_length: int) -> TimeSlice:
        """Calculate time slice indices"""
        finish_idx = min(int(overall_runtime_ms / (self.sample_intv * 1000)), data_length)

        time_slice = TimeSlice(end_idx=finish_idx)
        
        if start_ts is not None or end_ts is not None:
            time_slice.start_idx = max(0, int(start_ts / (self.sample_intv * 1000))) if start_ts else 0
            time_slice.end_idx = min(finish_idx, int(end_ts / (self.sample_intv * 1000))) if end_ts else finish_idx
            if time_slice.start_idx > time_slice.end_idx:
                raise ValueError("End timestamp is earlier than start timestamp")
        
        return time_slice
    
    def _calc_components_all_rows(self, profiled_df: pd.DataFrame, metrics: List[str], gpu: Dict[str, float]) -> List[Dict[str, float]]:
        """Calculate time components for all rows in the dataframe"""
        return [
            self.calc_time_components(row, metrics, gpu)
            for row in profiled_df.itertuples(index=False)
        ]
    
    def _slice_components(self, components_list: List[Dict[str, float]], time_slice: TimeSlice) -> Dict[str, List[float]]:
        """Slice components based on time slice"""
        return {
            key: [tc[key] for tc in components_list][time_slice.start_idx:time_slice.end_idx]
            for key in components_list[0].keys()
        }
    
    def model(self, profiled_df: pd.DataFrame, metrics: List[str], 
              overall_runtime_ms: float, start_ts: Optional[float], 
              end_ts: Optional[float], gpu_arch: str, precision: str):
        """Model performance on reference hardware"""
        gpu = self._get_gpu_spec(gpu_arch)
        
        # Calculate components for all rows
        time_components_list = self._calc_components_all_rows(profiled_df, metrics, gpu)

        # Get time slice
        time_slice = self.get_time_slice(overall_runtime_ms, start_ts, end_ts, len(time_components_list))
        
        # Slice components
        sliced_components = self._slice_components(time_components_list, time_slice)
        
        # Calculate FLOPS and DRAM bandwidth
        flop = np.mean(sliced_components['t_flop']) / self.sample_intv * gpu.get(precision)
        dram = np.mean(sliced_components['t_dram']) / self.sample_intv * gpu.get("mem_bw")
        
        # Add total time calculations
        sliced_components['t_total'] = [
            sliced_components.get('t_kernel')[i] + 
            sliced_components.get('t_othernode')[i]
            for i in range(len(sliced_components.get('t_kernel')))
        ]

        # Print results
        self._print_results(sliced_components, flop, dram, gpu_arch, "Reference")
    
    def predict(self, profiled_df: pd.DataFrame, metrics: List[str],
                overall_runtime_ms: float, start_ts: Optional[float], end_ts: Optional[float], 
                ref_gpu_arch: str, tgt_gpu_arch: str, precision: str, 
                flop_util_bound_switch: float = 1.0, mem_util_bound_switch: float = 1.0):
        """Predict performance on target hardware"""
        ref_gpu = self._get_gpu_spec(ref_gpu_arch)
        tgt_gpu = self._get_gpu_spec(tgt_gpu_arch)
        
        # Calculate target metrics
        target_metrics = self._calculate_target_metrics(
            profiled_df, metrics, ref_gpu, tgt_gpu, precision,
            flop_util_bound_switch, mem_util_bound_switch
        )
        
        # Get time slice
        time_slice = self.get_time_slice(overall_runtime_ms, start_ts, end_ts, len(target_metrics.get('t_total_lower')))

        # Slice metrics
        sliced_metrics = {
            key: values[time_slice.start_idx:time_slice.end_idx]
            for key, values in target_metrics.items()
        }
        
        # Calculate estimated FLOPS and memory bandwidth
        est_flops = (
            np.mean(sliced_metrics.get('tensor_ref')) * tgt_gpu.get(precision) +
            np.mean(sliced_metrics.get('fp64a_ref')) * tgt_gpu.get("fp64") +
            np.mean(sliced_metrics.get('fp32a_ref')) * tgt_gpu.get("fp32") +
            np.mean(sliced_metrics.get('fp16a_ref')) * tgt_gpu.get("fp16")
        )
        est_mem_bw = np.mean(sliced_metrics['drama_ref']) * tgt_gpu["mem_bw"]

        # Print predictions
        self._print_results(sliced_metrics, est_flops, est_mem_bw, tgt_gpu_arch, "Target")
    
    def _calculate_target_metrics(self, profiled_df: pd.DataFrame, metrics: List[str],
                                  ref_gpu: Dict, tgt_gpu: Dict, precision: str,
                                  flop_util: float, mem_util: float) -> Dict[str, List[float]]:
        """Calculate metrics for target hardware"""
        results = {
            't_kernel_lower': [], 't_kernel_upper': [],
            't_othernode': [], 't_total_lower': [], 't_total_upper': [],
            'drama_ref': [], 'tensor_ref': [], 'fp64a_ref': [], 'fp32a_ref': [], 'fp16a_ref': [],
            'smocc_ref': [], 'smocc_tgt': [], 'gract_ref': []
        }
        
        for row in profiled_df.itertuples(index=False):
            mv = {metric: getattr(row, metric) for metric in metrics}
            
            # Get and store the reference metrecis
            drama_ref = mv.get('DRAMA')
            tensor_ref = mv.get('TENSO')
            fp64a_ref = mv.get('FP64A')
            fp32a_ref = mv.get('FP32A')
            fp16a_ref = mv.get('FP16A')
            smocc_ref = mv.get('SMOCC')
            gract_ref = mv.get('GRACT') 
            results['gract_ref'].append(gract_ref)
            results['drama_ref'].append(drama_ref)
            results['tensor_ref'].append(tensor_ref)
            results['fp64a_ref'].append(fp64a_ref)
            results['fp32a_ref'].append(fp32a_ref)
            results['fp16a_ref'].append(fp16a_ref)
            results['smocc_ref'].append(smocc_ref)

            # Get hardware specification
            boost_clock_ref = ref_gpu.get("boost_clock")
            boost_clock_tgt = tgt_gpu.get("boost_clock")
            max_warp_sm_ref = ref_gpu.get("max_warps_sm")
            max_warp_sm_tgt = tgt_gpu.get("max_warps_sm")
            num_sm_ref = ref_gpu.get("num_sm")
            num_sm_tgt = tgt_gpu.get("num_sm")
            mem_bw_ref = ref_gpu.get("mem_bw")
            mem_bw_tgt = tgt_gpu.get("mem_bw")
            flop_fp64_ref = ref_gpu.get("fp64")
            flop_fp32_ref = ref_gpu.get("fp32")
            flop_fp16_ref = ref_gpu.get("fp16")

            # Calculate reference components
            ref_components = self.calc_time_components(row, metrics, ref_gpu)

            # Some scale ratio
            boost_clock_ratio = boost_clock_ref / boost_clock_tgt
            warp_ratio = max_warp_sm_ref / max_warp_sm_tgt
            sm_ratio = num_sm_ref / num_sm_tgt

            dram_intensity_ref = drama_ref / gract_ref if gract_ref != 0 else 0

            # Bound SMOCC and Calculate SMOCC scale factor
            smocc_tgt_lower, smocc_tgt_upper = self._bound_smocc(smocc_ref, ref_gpu, tgt_gpu)
            
            kernel_scale_lower = smocc_tgt_lower * warp_ratio * sm_ratio * boost_clock_ratio
            kernel_scale_upper = smocc_tgt_upper * warp_ratio * sm_ratio * boost_clock_ratio
            
            p_dram_ref = ref_gpu.get("mem_bw") * drama_ref / gract_ref if gract_ref !=0 else 0
            p_dram_tgt_lower = min(mem_bw_ref * dram_intensity_ref * kernel_scale_lower, mem_bw_tgt)
            p_dram_tgt_upper = min(mem_bw_ref * dram_intensity_ref * kernel_scale_upper, mem_bw_tgt)
            
            t_kernel_dram_tgt_lower = ref_components["t_kernel"] * (p_dram_ref / p_dram_tgt_lower) if p_dram_tgt_lower !=0 else 0
            t_kernel_dram_tgt_upper = ref_components["t_kernel"] * (p_dram_ref / p_dram_tgt_upper) if p_dram_tgt_upper !=0 else 0

            # compute workload type scale factor
            p_flop_ref = ((ref_gpu.get(precision) * tensor_ref / gract_ref) + 
                          (flop_fp64_ref * fp64a_ref / gract_ref) + 
                          (flop_fp32_ref * fp32a_ref / gract_ref) + 
                          (flop_fp16_ref * fp16a_ref / gract_ref)) if gract_ref !=0 else 0
            
            p_flop_tgt_lower = ((ref_gpu.get(precision) * tensor_ref / gract_ref * kernel_scale_lower) + 
                                (flop_fp64_ref * fp64a_ref / gract_ref * kernel_scale_lower) + 
                                (flop_fp32_ref * fp32a_ref / gract_ref * kernel_scale_lower) + 
                                (flop_fp16_ref * fp16a_ref / gract_ref * kernel_scale_lower)) if gract_ref * kernel_scale_lower != 0 else 0
            p_flop_tgt_upper = ((ref_gpu.get(precision) * tensor_ref / gract_ref * kernel_scale_upper) + 
                                (flop_fp64_ref * fp64a_ref / gract_ref * kernel_scale_upper) + 
                                (flop_fp32_ref * fp32a_ref / gract_ref * kernel_scale_upper) + 
                                (flop_fp16_ref * fp16a_ref / gract_ref * kernel_scale_upper)) if gract_ref * kernel_scale_lower != 0 else 0

            t_kernel_flop_tgt_lower = ref_components["t_kernel"] * (p_flop_ref / p_flop_tgt_lower) if p_flop_tgt_lower !=0 else 0
            t_kernel_flop_tgt_upper = ref_components["t_kernel"] * (p_flop_ref / p_flop_tgt_upper) if p_flop_tgt_upper !=0 else 0

            t_kernel_tgt_lower = max(t_kernel_dram_tgt_lower, t_kernel_flop_tgt_lower)
            t_kernel_tgt_upper = max(t_kernel_dram_tgt_upper, t_kernel_flop_tgt_upper)
            results['t_kernel_lower'].append(t_kernel_tgt_lower)
            results['t_kernel_upper'].append(t_kernel_tgt_upper)

            # Scale interconnect times
            t_othernode_tgt = ref_components['t_othernode']
            
            results['t_othernode'].append(t_othernode_tgt)
            
            # Calculate totals
            t_total_lower = t_kernel_tgt_lower + t_othernode_tgt
            t_total_total = t_kernel_tgt_upper + t_othernode_tgt
            
            results['t_total_lower'].append(t_total_lower)
            results['t_total_upper'].append(t_total_total)
        
        return results
    
    def _print_results(self, metrics: Dict[str, List[float]], flops: float, mem_bw: float, gpu_arch: str, hw_type: str):
        """Unified print function for both model and predict results"""
        print(f"============ {hw_type} Hardware: {gpu_arch} ============")
        print(f"Estimate TFLOPS on {hw_type} Hardware: {flops:.2f}")
        print(f"Estimate GPU Memory Bandwidth on {hw_type} Hardware: {mem_bw:.2f}")
        if hw_type == "Reference":
            print(f"Estimate Kernel Time On {hw_type} Hardware: {sum(metrics['t_kernel']):.2f}")
            print(f"Estimate otherNode Time On {hw_type} Hardware: {sum(metrics['t_othernode']):.2f}")
            print(f"Estimate Total Runtime On {hw_type} Hardware: {sum(metrics['t_total']):.2f}")
        else:
            print(f"Estimate Kernel Time On {hw_type} Hardware [Lower SMOCC]: {sum(metrics['t_kernel_lower']):.2f}")
            print(f"Estimate otherGPU Time On {hw_type} Hardware [Upper SMOCC]: {sum(metrics['t_kernel_upper']):.2f}")
            print(f"Estimate otherNode Time On {hw_type} Hardware: {sum(metrics['t_othernode']):.2f}")
            print(f"Estimate Total Runtime On {hw_type} Hardware [Lower SMOCC]: {sum(metrics['t_total_lower']):.2f}")
            print(f"Estimate Total Runtime On {hw_type} Hardware [Upper SMOCC]: {sum(metrics['t_total_upper']):.2f}")
        print()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--dcgm_file', required=True, help='DCGM output file path')
    parser.add_argument('-d', '--sample_interval_ms', type=int, required=True,
                       help='Sample interval in milliseconds')
    parser.add_argument('-st', '--start_timestamp', type=int, default=0, help='Start timestamp (ms)')
    parser.add_argument('-et', '--end_timestamp', type=int, help='End timestamp (ms)')
    parser.add_argument('-o', '--overall_runtime_ms', type=int, required=True,
                       help='Overall runtime in milliseconds')
    parser.add_argument('-rg', '--ref_gpu_architect', required=True, choices=list(GPUS.keys()),
                       help='Reference GPU architecture')
    parser.add_argument('-tg', '--target_gpu_architect', choices=list(GPUS.keys()),
                       help='Target GPU architecture')
    parser.add_argument('--metrics', type=lambda s: s.split(','), required=True,
                       help='Comma-separated list of metrics')
    parser.add_argument('-tp', '--tensor_precision', required=True, choices=['tf64', 'tf32', 'tf16'],
                       help='Tensor precision type')
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # if end timestamp is not provided, then use overall runtime 
    end_timestamp = args.overall_runtime_ms if args.end_timestamp is None else args.end_timestamp

    # Process metrics file
    profiled_df = MetricsProcessor.process_file(args.dcgm_file, args.metrics)

    # Create unified profiler
    profiler = PerformanceProfiler(args.sample_interval_ms)

    # Model reference performance
    profiler.model(
        profiled_df, args.metrics, args.overall_runtime_ms,
        args.start_timestamp, end_timestamp,
        args.ref_gpu_architect, args.tensor_precision
    )

    # Predict target performance if specified
    if args.target_gpu_architect:
        profiler.predict(
            profiled_df, args.metrics, args.overall_runtime_ms,
            args.start_timestamp, end_timestamp,
            args.ref_gpu_architect, args.target_gpu_architect,
            args.tensor_precision
        )
    

if __name__=="__main__":
    main()