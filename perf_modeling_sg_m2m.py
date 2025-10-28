import pandas as pd
import argparse
import re
import math
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# I got the numbers from nvidia official website and https://www.techpowerup.com/gpu-specs
# shared_mem_per_sm include L1 cache and shared memory
GPUS = {
    "A100-40": {
        "fp64": 9.7, "tf64": 19.5, "fp32": 19.5, "tf32": 156, "fp16": 78, "tf16": 312, 
        "mem_bw": 1555, "pcie_bw": 64, "nvlink_bw": 600, 
        "base_clock": 765, "boost_clock": 1410, "mem_clock": 1215,
        "num_sm": 108, "max_warps_per_sm": 64, "reg_per_sm": 256, "shared_mem_per_sm": 192
    },
    "A100-80": {
        "fp64": 9.7, "tf64": 19.5, "fp32": 19.5, "tf32": 156, "fp16": 78, "tf16": 312, 
        "mem_bw": 1935, "pcie_bw": 64, "nvlink_bw": 600, 
        "base_clock": 1065, "boost_clock": 1410, "mem_clock": 1512,
        "num_sm": 108, "max_warps_per_sm": 64, "reg_per_sm": 256, "shared_mem_per_sm": 192
    },
    "A40": {
        "fp64": 0.58, "tf64": 0, "fp32": 37.4, "tf32": 74.8, "fp16": 37.4, "tf16": 149.7, 
        "mem_bw": 696, "pcie_bw": 64, "nvlink_bw": 112.5, 
        "base_clock": 1305, "boost_clock": 1740, "mem_clock": 1812,
        "num_sm": 84, "max_warps_per_sm": 48, "reg_per_sm": 256, "shared_mem_per_sm": 128
    },
    "H100-SXM": {
        "fp64": 34, "tf64": 67, "fp32": 67, "tf32": 989, "fp16": 133.8, "tf16": 1979, 
        "mem_bw": 3350, "pcie_bw": 128, "nvlink_bw": 900, 
        "base_clock": 1590, "boost_clock": 1980, "mem_clock": 1313, 
        "num_sm": 132, "max_warps_per_sm": 64, "reg_per_sm": 256, "shared_mem_per_sm": 228
    },
    "H100-NVL": {
        "fp64": 30, "tf64": 60, "fp32": 60, "tf32": 835, "fp16": 133.8, "tf16": 1671, 
        "mem_bw": 3900, "pcie_bw": 128, "nvlink_bw": 600, 
        "base_clock": 1080, "boost_clock": 1785, "mem_clock": 1593,
        "num_sm": 114, "max_warps_per_sm": 64, "reg_per_sm": 256, "shared_mem_per_sm": 228
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
        multiplier = config["multipliers"].get(key, 1.0)
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
    
    def _scale_sm_occupancy(self, sm_occ_ref: float, ref_gpu: Dict[str, float], tgt_gpu: Dict[str, float]) -> float:
        """Scale SM occupancy from reference to target GPU"""
        '''
        resource_ratio = min(
            tgt_gpu.get("max_warps_per_sm", 1) / ref_gpu.get("max_warps_per_sm", 1),
            tgt_gpu.get("reg_per_sm", 1e10) / ref_gpu.get("reg_per_sm", 1e10),
            tgt_gpu.get("shared_mem_per_sm", 1e10) / ref_gpu.get("shared_mem_per_sm", 1e10)
        )
        '''
        sm_occ_ref = 0.01 if sm_occ_ref == 0 else sm_occ_ref
 
        resource_ratio = min(
            tgt_gpu.get("max_warps_per_sm", 1),
            tgt_gpu.get("reg_per_sm", 1e10) / ref_gpu.get("reg_per_sm", 1e10) * ref_gpu.get("max_warps_per_sm", 1) * sm_occ_ref,
            tgt_gpu.get("shared_mem_per_sm", 1e10) / ref_gpu.get("shared_mem_per_sm", 1e10) * ref_gpu.get("max_warps_per_sm", 1) * sm_occ_ref
        ) / (ref_gpu.get("max_warps_per_sm", 1) * sm_occ_ref)

        return min(sm_occ_ref * resource_ratio, 1.0)

    def _scale_sm_occupancy_roofline(self, mv: Dict[str, float], smocc_scale: float, 
                                     ref_components: Dict[str, float], ref_gpu: Dict, tgt_gpu: Dict, precision: str):
        """Apply estimated SM occupancy to roofline time"""
        
        t_flop_tgt = self.sample_intv * sum(
            mv.get(m, 0) * ref_gpu[precision] / tgt_gpu[precision]
            if m == 'TENSO' else
            mv.get(m, 0) * ref_gpu[m.lower()[:4]] / tgt_gpu[m.lower()[:4]]
            for m in ['TENSO', 'FP64A', 'FP32A', 'FP16A']
        )
        t_dram_tgt = ref_components['t_dram'] * (ref_gpu["mem_bw"] / tgt_gpu["mem_bw"])

        t_pcie_tgt = ref_components['t_pcie'] * (ref_gpu["pcie_bw"] / tgt_gpu["pcie_bw"])
        t_nvlink_tgt = ref_components['t_nvlink'] * (ref_gpu["nvlink_bw"] / tgt_gpu["nvlink_bw"])
        t_other_node_tgt = ref_components['t_other_node']

        t_total_tgt = t_flop_tgt + t_dram_tgt + t_pcie_tgt + t_nvlink_tgt + t_other_node_tgt
        
        flop_intensity = t_flop_tgt / t_total_tgt
        dram_intensity = t_dram_tgt / t_total_tgt
        # print(f"flop_intensity: {flop_intensity}, dram_intensity:{dram_intensity}")
        
        flop_smocc_scale = smocc_scale * flop_intensity
        dram_smocc_scale = smocc_scale * dram_intensity
        # print(f"smocc_scale: {smocc_scale}")
        return flop_smocc_scale, dram_smocc_scale

    def calc_time_components(self, row, metrics: List[str], gpu: Dict[str, float]) -> Dict[str, float]:
        """Calculate various time components for a single row"""
        mv = {metric: getattr(row, metric) for metric in metrics}
        
        t_flop = self.sample_intv * sum(mv.get(m, 0) for m in ['TENSO', 'FP64A', 'FP32A', 'FP16A'])
        t_dram = self.sample_intv * mv.get('DRAMA', 0)
        
        t_roofline_overlap = max(t_flop, t_dram)
        t_roofline_sequential = t_flop + t_dram
        
        gract = mv.get('GRACT', 0)
        t_other_gpu_overlap = max(0, self.sample_intv * gract - t_roofline_overlap)
        t_other_gpu_sequential = max(0, self.sample_intv * gract - t_roofline_sequential)
        
        t_pcie = (mv.get('PCITX', 0) + mv.get('PCIRX', 0)) * self.sample_intv / (gpu["pcie_bw"] * 1e9)
        t_nvlink = (mv.get('NVLTX', 0) + mv.get('NVLRX', 0)) * self.sample_intv / (gpu["nvlink_bw"] * 1e9)
        t_other_node = max(0, self.sample_intv * (1 - gract) - t_pcie - t_nvlink)
        
        return {
            't_flop': t_flop,
            't_dram': t_dram,
            't_roofline_overlap': t_roofline_overlap,
            't_roofline_sequential': t_roofline_sequential,
            't_other_gpu_overlap': t_other_gpu_overlap,
            't_other_gpu_sequential': t_other_gpu_sequential,
            't_pcie': t_pcie,
            't_nvlink': t_nvlink,
            't_other_node': t_other_node
        }
    
    def get_time_slice(self, overall_runtime_ms: float, start_ts: Optional[float], end_ts: Optional[float], data_length: int) -> TimeSlice:
        """Calculate time slice indices"""
        finish_idx = min(int(overall_runtime_ms / (self.sample_intv * 1000)), data_length)
        
        time_slice = TimeSlice(end_idx=finish_idx)
        
        if start_ts is not None or end_ts is not None:
            time_slice.start_idx = max(0, int(start_ts / (self.sample_intv * 1000))) if start_ts else 0
            time_slice.end_idx = min(finish_idx, int(end_ts / (self.sample_intv * 1000))) if end_ts else finish_idx
            
            if time_slice.start_idx >= time_slice.end_idx:
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
        flop = np.mean(sliced_components['t_flop']) / self.sample_intv * gpu[precision]
        dram = np.mean(sliced_components['t_dram']) / self.sample_intv * gpu["mem_bw"]
        
        # Add total time calculations
        sliced_components['t_total_overlap'] = [
            sliced_components['t_roofline_overlap'][i] + 
            sliced_components['t_other_gpu_overlap'][i] + 
            sliced_components['t_pcie'][i] +
            sliced_components['t_nvlink'][i] +
            sliced_components['t_other_node'][i]
            for i in range(len(sliced_components['t_roofline_overlap']))
        ]
        sliced_components['t_total_sequential'] = [
            sliced_components['t_roofline_sequential'][i] + 
            sliced_components['t_other_gpu_sequential'][i] + 
            sliced_components['t_pcie'][i] +
            sliced_components['t_nvlink'][i] +
            sliced_components['t_other_node'][i]
            for i in range(len(sliced_components['t_roofline_sequential']))
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
        time_slice = self.get_time_slice(overall_runtime_ms, start_ts, end_ts, len(target_metrics['t_total_overlap']))
        
        # Slice metrics
        sliced_metrics = {
            key: values[time_slice.start_idx:time_slice.end_idx]
            for key, values in target_metrics.items()
        }
        
        # Calculate estimated FLOPS and memory bandwidth
        est_flops = (
            np.mean(sliced_metrics['tensor_ref']) * tgt_gpu[precision] +
            np.mean(sliced_metrics['fp64a_ref']) * tgt_gpu["fp64"] +
            np.mean(sliced_metrics['fp32a_ref']) * tgt_gpu["fp32"] +
            np.mean(sliced_metrics['fp16a_ref']) * tgt_gpu["fp16"]
        )
        est_mem_bw = np.mean(sliced_metrics['drama_ref']) * tgt_gpu["mem_bw"]

        # Print predictions
        self._print_results(sliced_metrics, est_flops, est_mem_bw, tgt_gpu_arch, "Target")
    
    def _calculate_target_metrics(self, profiled_df: pd.DataFrame, metrics: List[str],
                                  ref_gpu: Dict, tgt_gpu: Dict, precision: str,
                                  flop_util: float, mem_util: float) -> Dict[str, List[float]]:
        """Calculate metrics for target hardware"""
        results = {
            't_roofline_overlap': [], 't_roofline_sequential': [],
            't_other_gpu_overlap': [], 't_other_gpu_sequential': [],
            't_other_node': [], 't_total_overlap': [], 't_total_sequential': [],
            'drama_ref': [], 'tensor_ref': [], 'fp64a_ref': [], 'fp32a_ref': [], 'fp16a_ref': [],
            'sm_occ_ref': [], 'sm_occ_tgt': []
        }
        
        for row in profiled_df.itertuples(index=False):
            mv = {metric: getattr(row, metric) for metric in metrics}
            
            # Store reference metrics
            results['drama_ref'].append(mv.get('DRAMA', 0))
            results['tensor_ref'].append(mv.get('TENSO', 0))
            results['fp64a_ref'].append(mv.get('FP64A', 0))
            results['fp32a_ref'].append(mv.get('FP32A', 0))
            results['fp16a_ref'].append(mv.get('FP16A', 0))
            
            # Some scale ratio
            compute_util = mv.get('TENSO', 0) + mv.get('FP64A', 0) + mv.get('FP32A', 0) + mv.get('FP16A', 0)
            dram_util = mv.get('DRAMA', 0)
            boost_clock_ratio = ref_gpu["boost_clock"] / tgt_gpu["boost_clock"]
            mem_clock_ratio = ref_gpu["mem_clock"] / tgt_gpu["mem_clock"]
            warp_ratio = ref_gpu.get("max_warps_per_sm", 1) / tgt_gpu.get("max_warps_per_sm", 1)
            stream_ratio = ref_gpu.get("num_streams", 1) / tgt_gpu.get("num_streams", 1)

            # Calculate SM Occupancy scaling if available
            sm_occ_ref = mv.get('SMOCC', 0)
            sm_occ_tgt = self._scale_sm_occupancy(sm_occ_ref, ref_gpu, tgt_gpu)

            results['sm_occ_ref'].append(sm_occ_ref)
            results['sm_occ_tgt'].append(sm_occ_tgt)
            
            # Calculate efficiency scale (avoid division by zero)
            if sm_occ_tgt > 0.01 and sm_occ_ref > 0.01:
                smocc_scale = sm_occ_ref / sm_occ_tgt
            else:
                smocc_scale = 1.0
            
            # Calculate reference components
            ref_components = self.calc_time_components(row, metrics, ref_gpu)
            
            # flop_smocc_scale, dram_smocc_scale = self._scale_sm_occupancy_roofline(mv, smocc_scale, ref_components, ref_gpu, tgt_gpu, precision)
            flop_roofline_scale = stream_ratio * warp_ratio * boost_clock_ratio * smocc_scale
            dram_roofline_scale = stream_ratio * warp_ratio * boost_clock_ratio * smocc_scale * mem_clock_ratio
            
            if compute_util > dram_util:
                t_flop_target = self.sample_intv * sum(
                    mv.get(m, 0) * ref_gpu[precision] / tgt_gpu[precision]
                    if m == 'TENSO' else
                    mv.get(m, 0) * ref_gpu[m.lower()[:4]] / tgt_gpu[m.lower()[:4]]
                    for m in ['TENSO', 'FP64A', 'FP32A', 'FP16A']
                )
                t_dram_target = ref_components['t_dram'] * ref_gpu["mem_bw"] / tgt_gpu["mem_bw"]
            else:
                t_flop_target = self.sample_intv * sum(
                    mv.get(m, 0) * max(ref_gpu[precision] / tgt_gpu[precision], ref_gpu[precision] / tgt_gpu[precision] * flop_roofline_scale)
                    if m == 'TENSO' else
                    mv.get(m, 0) * max(ref_gpu[m.lower()[:4]] / tgt_gpu[m.lower()[:4]], ref_gpu[m.lower()[:4]] / tgt_gpu[m.lower()[:4]] * flop_roofline_scale)
                    for m in ['TENSO', 'FP64A', 'FP32A', 'FP16A']
                )
                t_dram_target = ref_components['t_dram'] * max(ref_gpu["mem_bw"] / tgt_gpu["mem_bw"], dram_roofline_scale)

            t_roofline_overlap = max(t_flop_target, t_dram_target)
            t_roofline_sequential = (t_flop_target + t_dram_target)
            
            results['t_roofline_overlap'].append(t_roofline_overlap)
            results['t_roofline_sequential'].append(t_roofline_sequential)

            t_other_gpu_overlap = ref_components['t_other_gpu_overlap'] * flop_roofline_scale
            t_other_gpu_sequential = ref_components['t_other_gpu_sequential'] * flop_roofline_scale
            
            results['t_other_gpu_overlap'].append(t_other_gpu_overlap)
            results['t_other_gpu_sequential'].append(t_other_gpu_sequential)
            
            # Scale interconnect times
            t_pcie = ref_components['t_pcie'] * (ref_gpu["pcie_bw"] / tgt_gpu["pcie_bw"])
            t_nvlink = ref_components['t_nvlink'] * (ref_gpu["nvlink_bw"] / tgt_gpu["nvlink_bw"])
            t_other_node = ref_components['t_other_node']
            
            results['t_other_node'].append(t_other_node)
            
            # Calculate totals
            t_total_overlap = t_roofline_overlap + t_other_gpu_overlap + t_pcie + t_nvlink + t_other_node
            t_total_sequential = t_roofline_sequential + t_other_gpu_sequential + t_pcie + t_nvlink + t_other_node
            
            results['t_total_overlap'].append(t_total_overlap)
            results['t_total_sequential'].append(t_total_sequential)
        
        return results
    
    def _print_results(self, metrics: Dict[str, List[float]], flops: float, mem_bw: float, gpu_arch: str, hw_type: str):
        """Unified print function for both model and predict results"""
        print(f"============ {hw_type} Hardware: {gpu_arch} ============")
        print(f"Estimate TFLOPS on {hw_type} Hardware: {flops:.2f}")
        print(f"Estimate GPU Memory Bandwidth on {hw_type} Hardware: {mem_bw:.2f}")
        print(f"Estimate Roofline Time On {hw_type} Hardware [Overlap Scenario]: "
              f"{sum(metrics['t_roofline_overlap']):.2f}")
        print(f"Estimate Roofline Time On {hw_type} Hardware [Sequential Scenario]: "
              f"{sum(metrics['t_roofline_sequential']):.2f}")
        print(f"Estimate otherGPU Time On {hw_type} Hardware [Overlap Scenario]: "
              f"{sum(metrics['t_other_gpu_overlap']):.2f}")
        print(f"Estimate otherGPU Time On {hw_type} Hardware [Sequential Scenario]: "
              f"{sum(metrics['t_other_gpu_sequential']):.2f}")
        print(f"Estimate otherNode Time On {hw_type} Hardware: {sum(metrics['t_other_node']):.2f}")
        print(f"Estimate Total Runtime On {hw_type} Hardware [Overlap Scenario]: "
              f"{sum(metrics['t_total_overlap']):.2f}")
        print(f"Estimate Total Runtime On {hw_type} Hardware [Sequential Scenario]: "
              f"{sum(metrics['t_total_sequential']):.2f}")
        print()


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--dcgm_file', required=True, help='DCGM output file path')
    parser.add_argument('-d', '--sample_interval_ms', type=int, required=True,
                       help='Sample interval in milliseconds')
    parser.add_argument('-st', '--start_timestamp', type=int, help='Start timestamp (ms)')
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
    parser.add_argument('-fu', '--flop_util', type=float, default=1.0,
                       help='FLOPS utilization when bound switches')
    parser.add_argument('-mu', '--mem_util', type=float, default=1.0,
                       help='Memory utilization when bound switches')
    return parser.parse_args()


def main():
    args = parse_arguments()

    # Process metrics file
    profiled_df = MetricsProcessor.process_file(args.dcgm_file, args.metrics)

    # Create unified profiler
    profiler = PerformanceProfiler(args.sample_interval_ms)

    # Model reference performance
    profiler.model(
        profiled_df, args.metrics, args.overall_runtime_ms,
        args.start_timestamp, args.end_timestamp,
        args.ref_gpu_architect, args.tensor_precision
    )

    # Predict target performance if specified
    if args.target_gpu_architect:
        profiler.predict(
            profiled_df, args.metrics, args.overall_runtime_ms,
            args.start_timestamp, args.end_timestamp,
            args.ref_gpu_architect, args.target_gpu_architect,
            args.tensor_precision, args.flop_util, args.mem_util
        )
    

if __name__=="__main__":
    main()