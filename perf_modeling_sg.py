import argparse
import re
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

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


@dataclass
class MetricValues:
    """Container for extracted metric values from a row"""
    gract: float = 0.0
    drama: float = 0.0
    tenso: float = 0.0
    fp64a: float = 0.0
    fp32a: float = 0.0
    fp16a: float = 0.0
    smocc: float = 0.0
    pcitx: float = 0.0
    pcirx: float = 0.0
    nvltx: float = 0.0
    nvlrx: float = 0.0
    
    @classmethod
    def from_row(cls, row, metrics: List[str]) -> 'MetricValues':
        """Create MetricValues from a dataframe row"""
        return cls(
            gract=getattr(row, 'GRACT', 0.0) if 'GRACT' in metrics else 0.0,
            drama=getattr(row, 'DRAMA', 0.0) if 'DRAMA' in metrics else 0.0,
            tenso=getattr(row, 'TENSO', 0.0) if 'TENSO' in metrics else 0.0,
            fp64a=getattr(row, 'FP64A', 0.0) if 'FP64A' in metrics else 0.0,
            fp32a=getattr(row, 'FP32A', 0.0) if 'FP32A' in metrics else 0.0,
            fp16a=getattr(row, 'FP16A', 0.0) if 'FP16A' in metrics else 0.0,
            smocc=getattr(row, 'SMOCC', 0.0) if 'SMOCC' in metrics else 0.0,
            pcitx=getattr(row, 'PCITX', 0.0) if 'SMOCC' in metrics else 0.0,
            pcirx=getattr(row, 'PCIRX', 0.0) if 'SMOCC' in metrics else 0.0,
            nvltx=getattr(row, 'NVLTX', 0.0) if 'SMOCC' in metrics else 0.0,
            nvlrx=getattr(row, 'NVLRX', 0.0) if 'SMOCC' in metrics else 0.0,
        )
    
    def get_flop_sum(self) -> float:
        """Sum of all FLOP-related metrics"""
        return self.tenso + self.fp64a + self.fp32a + self.fp16a


@dataclass
class TimeComponents:
    """Container for calculated time components"""
    t_flop: float = 0.0
    t_dram: float = 0.0
    t_kernel: float = 0.0
    t_othernode: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            't_flop': self.t_flop,
            't_dram': self.t_dram,
            't_kernel': self.t_kernel,
            't_othernode': self.t_othernode
        }


@dataclass
class TimeSlice:
    """Container for time-based metrics with slicing functionality"""
    start_idx: int = 0
    end_idx: Optional[int] = None
    
    def slice_list(self, data: List) -> List:
        """Apply slicing to a list"""
        return data[self.start_idx:self.end_idx]
    
    def slice_dict(self, data: Dict[str, List]) -> Dict[str, List]:
        """Apply slicing to all lists in a dictionary"""
        return {
            key: values[self.start_idx:self.end_idx]
            for key, values in data.items()
        }
    
    def slice_dataframe(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply slicing to a list"""
        return data[self.start_idx:self.end_idx]


class TimeCalculator:
    """Handles time-related calculations"""
    
    def __init__(self, sample_interval_ms: float):
        self.sample_intv = sample_interval_ms / 1000
    
    def calc_components(self, metrics: MetricValues) -> TimeComponents:
        """Calculate time components from metrics"""
        t_flop = self.sample_intv * metrics.get_flop_sum()
        t_dram = self.sample_intv * metrics.drama
        t_kernel = self.sample_intv * metrics.gract
        t_othernode = max(self.sample_intv * (1 - metrics.gract), 0)
        
        return TimeComponents(
            t_flop=t_flop,
            t_dram=t_dram,
            t_kernel=t_kernel,
            t_othernode=t_othernode
        )
    
    def get_time_slice(self, overall_runtime_ms: float, start_ts: Optional[float], 
                       end_ts: Optional[float], data_length: int) -> TimeSlice:
        """Calculate time slice indices"""
        finish_idx = min(
            int(overall_runtime_ms / (self.sample_intv * 1000)), 
            data_length
        )
        start_idx = int((start_ts or 0) / (self.sample_intv * 1000))
        
        if end_ts is not None:
            end_idx = min(finish_idx, int(end_ts / (self.sample_intv * 1000)))
            if start_idx > end_idx:
                raise ValueError("End timestamp is earlier than start timestamp")
        else:
            end_idx = finish_idx
        
        return TimeSlice(start_idx=start_idx, end_idx=end_idx)


class MetricIntensityCalculator:
    """Calculates computational intensities"""
    
    def metric_intensities(self, metrics: MetricValues) -> Dict[str, float]:
        """Calculate all intensity metrics"""
        if metrics.gract < 0.01:
            return {
                'drama_gract': 0.0, 'tenso_gract': 0.0, 'fp64a_gract': 0.0, 
                'fp32a_gract': 0.0, 'fp16a_gract': 0.0, 'smocc_gract': 0.0
            }
        
        return {
            'drama_gract': metrics.drama / metrics.gract,
            'tenso_gract': metrics.tenso / metrics.gract,
            'fp64a_gract': metrics.fp64a / metrics.gract,
            'fp32a_gract': metrics.fp32a / metrics.gract,
            'fp16a_gract': metrics.fp16a / metrics.gract,
            'smocc_gract': metrics.smocc / metrics.gract
        }


class ScaleCalculator:
    """Calculates computational intensities"""

    INTENSITY_THRESHOLD = 0.01

    def __init__(self, ref_gpu: GPU, tgt_gpu: GPU, tensor_prec: str):
        self.ref_gpu = ref_gpu
        self.tgt_gpu = tgt_gpu
        
        # Precompute ratios
        self._compute_ratios(tensor_prec)

        # Initialize state
        self.cur_smocc = 0
        self.cur_warps_ref = 0
        self.cur_warps_tgt = {'lower': 0, 'mid': 0, 'upper': 0}
        self.scale_smocc = {'lower': 0, 'mid': 0, 'upper': 0}

    def _compute_ratios(self, tensor_prec: str):
        """Compute all GPU spec ratios once during initialization"""
        self.reg_sm_limit = self._get_ratio("reg_size_sm")
        self.shmem_sm_limit = self._get_ratio("shmem_sm")
        self.bw_ratio = self._get_ratio("mem_bw")
        self.tensor_ratio = self._get_ratio(tensor_prec)
        self.fp64_ratio = self._get_ratio("fp64")
        self.fp32_ratio = self._get_ratio("fp32")
        self.fp16_ratio = self._get_ratio("fp16")

    def _get_ratio(self, spec: str) -> float:
        """Helper to compute target/reference ratio for a given spec"""
        return self.tgt_gpu.get_specs(spec) / self.ref_gpu.get_specs(spec)

    def _estimate_warps(self):
        ref_max_warps = self.ref_gpu.get_specs("max_warps_sm")
        tgt_max_warps = self.tgt_gpu.get_specs("max_warps_sm")

        self.cur_warps_ref = self.cur_smocc * ref_max_warps

        self.cur_warps_tgt['lower'] = min(
            self.cur_warps_ref * self.reg_sm_limit,
            self.cur_warps_ref * self.shmem_sm_limit,
            tgt_max_warps
        )
        
        self.cur_warps_tgt['mid'] = min(
            self.cur_warps_ref * (self.reg_sm_limit + self.shmem_sm_limit) * 0.5,
            tgt_max_warps
        )   

        self.cur_warps_tgt['upper'] = min(
            max(self.cur_warps_ref * self.reg_sm_limit,
                self.cur_warps_ref * self.shmem_sm_limit),
            tgt_max_warps
        )
        
    def refresh_smocc(self, smocc_ref: float):
        self.cur_smocc = smocc_ref
        self._estimate_warps()

    def smocc_scale(self) -> Tuple[float, float, float]:
        """Calculate SM occupancy scaling factors"""
        k_smocc_ref = self._compute_k_smocc(self.cur_warps_ref, self.ref_gpu)
        
        for level in ['lower', 'mid', 'upper']:
            k_smocc_tgt = self._compute_k_smocc(self.cur_warps_tgt[level], self.tgt_gpu)
            self.scale_smocc[level] = k_smocc_tgt / k_smocc_ref if k_smocc_ref != 0 else 0
        
        return self.scale_smocc['lower'], self.scale_smocc['mid'], self.scale_smocc['upper']

    def est_dram_tgt(self, drama_ref: float) -> Tuple[float, float, float]:
        return (
            min(self.ref_gpu["mem_bw"] * drama_ref * self.scale_smocc['lower'], self.tgt_gpu["mem_bw"]),
            min(self.ref_gpu["mem_bw"] * drama_ref * self.scale_smocc['mid'], self.tgt_gpu["mem_bw"]),
            min(self.ref_gpu["mem_bw"] * drama_ref * self.scale_smocc['upper'], self.tgt_gpu["mem_bw"])
        )

    def est_flop_tgt(self, tenso_ref: float, fp64a_ref: float, fp32a_ref: float, fp16a_ref: float, tf_prec: str) -> Tuple[float, float, float]:
        flop_tgt_lower = flop_tgt_mid = flop_tgt_upper = 0
        for mv, spec in [(tenso_ref, tf_prec), (fp64a_ref, 'fp64'), (fp32a_ref, 'fp32'), (fp16a_ref, 'fp16')]:
            flop_tgt_lower += min(self.ref_gpu[spec] * mv * self.scale_smocc['lower'], self.tgt_gpu[spec])
            flop_tgt_mid += min(self.ref_gpu[spec] * mv * self.scale_smocc['mid'], self.tgt_gpu[spec])
            flop_tgt_upper += min(self.ref_gpu[spec] * mv * self.scale_smocc['upper'], self.tgt_gpu[spec])
        
        return flop_tgt_lower, flop_tgt_mid, flop_tgt_upper

    def _compute_k_smocc(self, warps: float, gpu: GPU) -> float:
        """Compute k_smocc value for given warps and GPU"""
        return warps * gpu.get_specs("num_sm") * gpu.get_specs("boost_clock")

    def dram_scale(self, dram_ref: float) -> Tuple[float, float, float]:
        """Calculate DRAM bandwidth scaling factors"""
        return self._compute_scale(dram_ref, self.bw_ratio)

    def tensor_scale(self, tensor_ref: float) -> Tuple[float, float, float]:
        """Calculate tensor core scaling factors"""
        return self._compute_scale(tensor_ref, self.tensor_ratio)

    def fp64_scale(self, fp64_ref: float) -> Tuple[float, float, float]:
        """Calculate FP64 scaling factors"""
        return self._compute_scale(fp64_ref, self.fp64_ratio)
    
    def fp32_scale(self, fp32_ref: float) -> Tuple[float, float, float]:
        """Calculate FP32 scaling factors"""
        return self._compute_scale(fp32_ref, self.fp32_ratio)

    def fp16_scale(self, fp16_ref: float) -> Tuple[float, float, float]:
        """Calculate FP16 scaling factors"""
        return self._compute_scale(fp16_ref, self.fp16_ratio)

    def _compute_scale(self, intensity_ref: float, ratio: float) -> Tuple[float, float, float]:
        """Generic method to compute scaling factors for any intensity metric"""
        if intensity_ref < self.INTENSITY_THRESHOLD:
            return np.inf, np.inf, np.inf
        
        scale_factor = ratio / intensity_ref
        return (
            min(self.scale_smocc['lower'], scale_factor),
            min(self.scale_smocc['mid'], scale_factor),
            min(self.scale_smocc['upper'], scale_factor)
        )

class ResultsFormatter:
    """Formats and prints results"""
    
    @staticmethod
    def print_reference_results(metrics: Dict[str, List[float]], flops: float, mem_bw: float, gpu_name: str):
        """Print reference hardware results"""
        print(f"\n{'='*60}")
        print(f"Reference Hardware: {gpu_name}\n")
        print(f"Estimated TFLOPS: {flops:.2f}")
        print(f"Estimated GPU Memory Bandwidth: {mem_bw:.2f} GB/s")
        
        print(f"\nEstimated Kernel Time: {sum(metrics['t_kernel']):.2f} s")
        print(f"Estimated Other Node Time: {sum(metrics['t_othernode']):.2f} s")
        print(f"Estimated Total Runtime: {sum(metrics['t_total']):.2f} s")
        print(f"{'='*60}\n")
    
    @staticmethod
    def print_target_results(metrics: Dict[str, List[float]], flops: Dict[str, float], mem_bw: Dict[str, float], gpu_name: str):
        """Print target hardware results"""
        print(f"\n{'='*60}")
        print(f"Target Hardware: {gpu_name}")
        
        print(f'Estimated TFLOPS [Lower SMOCC]: {flops.get("flop_smocc_lower"):.2f} GB/s')
        print(f'Estimated TFLOPS [Mid SMOCC]: {flops.get("flop_smocc_mid"):.2f} GB/s')
        print(f'Estimated TFLOPS [Upper SMOCC]: {flops.get("flop_smocc_upper"):.2f} GB/s')
        
        print(f'Estimated GPU Memory Bandwidth [Lower SMOCC]: {mem_bw.get("dram_smocc_lower"):.2f} GB/s')
        print(f'Estimated GPU Memory Bandwidth [Mid SMOCC]: {mem_bw.get("dram_smocc_mid"):.2f} GB/s')
        print(f'Estimated GPU Memory Bandwidth [Upper SMOCC]: {mem_bw.get("dram_smocc_upper"):.2f} GB/s')

        print(f"\nEstimated Kernel Time [Lower SMOCC]: {sum(metrics['t_kernel_lower']):.2f} s")
        print(f"Estimated Kernel Time [Mid SMOCC]:   {sum(metrics['t_kernel_mid']):.2f} s")
        print(f"Estimated Kernel Time [Upper SMOCC]: {sum(metrics['t_kernel_upper']):.2f} s")
        
        print(f"\nEstimated Other Node Time: {sum(metrics['t_othernode']):.2f} s")
        
        print(f"\nEstimated Total Runtime [Lower SMOCC]: {sum(metrics['t_total_lower']):.2f} s")
        print(f"Estimated Total Runtime [Mid SMOCC]:   {sum(metrics['t_total_mid']):.2f} s")
        print(f"Estimated Total Runtime [Upper SMOCC]: {sum(metrics['t_total_upper']):.2f} s")
        print(f"{'='*60}\n")


class BaseProfiler(ABC):
    """Abstract base class for profilers"""
    
    def __init__(self, sample_interval_ms: float):
        self.time_calc = TimeCalculator(sample_interval_ms)
        self.intensity_calc = MetricIntensityCalculator()
        self.formatter = ResultsFormatter()
        
    @abstractmethod
    def run(self, *args, **kwargs):
        """Run the profiling/prediction"""
        pass


class ReferenceProfiler(BaseProfiler):
    """Profiles performance on reference hardware"""
    def __init__(self, sample_interval_ms, gpu_name):
        self.gpu = GPU(gpu_name=gpu_name)
        super().__init__(sample_interval_ms)

    def run(self, profiled_df: pd.DataFrame, metrics: List[str], 
            overall_runtime_ms: float, start_ts: Optional[float], 
            end_ts: Optional[float], tensor_prec: str):
        """Model performance on reference hardware"""        
        # Calculate components for all rows
        components_list = self._calc_all_components(profiled_df, metrics)
        
        # Get time slice
        time_slice = self.time_calc.get_time_slice(
            overall_runtime_ms, start_ts, end_ts, len(components_list)
        )
        
        # Slice and aggregate components
        sliced = self._slice_and_aggregate(components_list, time_slice)
        
        # Calculate performance metrics
        flops = self._calc_flops(time_slice.slice_dataframe(profiled_df), metrics, tensor_prec)
        membw = self._calc_membw(time_slice.slice_dataframe(profiled_df), metrics)
        
        # Print results
        self.formatter.print_reference_results(sliced, flops, membw, self.gpu.get_name())
    
    def _calc_all_components(self, profiled_df: pd.DataFrame, metrics: List[str]) -> List[TimeComponents]:
        """Calculate time components for all rows"""
        return [
            self.time_calc.calc_components(MetricValues.from_row(row, metrics))
            for row in profiled_df.itertuples(index=False)
        ]
    
    def _slice_and_aggregate(self, components_list: List[TimeComponents], time_slice: TimeSlice) -> Dict[str, List[float]]:
        """Slice components and add total time"""
        sliced = {
            key: [comp.to_dict()[key] for comp in components_list][time_slice.start_idx:time_slice.end_idx]
            for key in components_list[0].to_dict().keys()
        }
        
        # Add total time
        sliced['t_total'] = [
            sliced['t_kernel'][i] + sliced['t_othernode'][i]
            for i in range(len(sliced['t_kernel']))
        ]
        
        return sliced
    
    def _calc_flops(self, profiled_df: pd.DataFrame, metrics: List[str], tensor_prec: str) -> float:
        """Calculate FLOPS"""
        flop_sum = 0
        for row in profiled_df.itertuples(index=False):
            mv = MetricValues.from_row(row, metrics)
            intensities = self.intensity_calc.metric_intensities(mv)
            tensor_util = intensities['tenso_gract'] * self.gpu.get_specs(tensor_prec)
            fp64_util = intensities['fp64a_gract'] * self.gpu.get_specs("fp64")
            fp32_util = intensities['fp32a_gract'] * self.gpu.get_specs("fp32")
            fp16_util = intensities['fp16a_gract'] * self.gpu.get_specs("fp16")
            flop_sum += tensor_util + fp64_util + fp32_util + fp16_util

        return flop_sum / len(profiled_df)
    
    def _calc_membw(self, profiled_df: pd.DataFrame, metrics: List[str]) -> float:
        """Calculate memory bandwidth"""
        dram_sum = 0
        for row in profiled_df.itertuples(index=False):
            mv = MetricValues.from_row(row, metrics)
            intensities = self.intensity_calc.metric_intensities(mv)
            dram_sum += intensities['drama_gract'] * self.gpu.get_specs("mem_bw")
        
        return dram_sum / len(profiled_df)


class TargetPredictor(BaseProfiler):
    """Predicts performance on target hardware"""
    def __init__(self, sample_interval_ms, ref_gpu_name, tgt_gpu_name):
        self.ref_gpu = GPU(gpu_name=ref_gpu_name)
        self.tgt_gpu = GPU(gpu_name=tgt_gpu_name)
        super().__init__(sample_interval_ms)

    def run(self, profiled_df: pd.DataFrame, metrics: List[str],
            overall_runtime_ms: float, start_ts: Optional[float], 
            end_ts: Optional[float], tensor_prec: str):
        """Predict performance on target hardware"""        
        # Calculate target metrics
        target_metrics = self._calc_target_metrics(
            profiled_df, metrics, self.ref_gpu, self.tgt_gpu, tensor_prec
        )
        
        # Get time slice
        time_slice = self.time_calc.get_time_slice(
            overall_runtime_ms, start_ts, end_ts, len(target_metrics['t_total_lower'])
        )
        
        # Slice metrics
        sliced_metrics = time_slice.slice_dict(target_metrics)
        
        # Calculate estimated FLOPS and memory bandwidth
        est_flops = self._calc_est_flops(target_metrics)
        est_membw = self._calc_est_membw(target_metrics)
        
        # Print predictions
        self.formatter.print_target_results(sliced_metrics, est_flops, est_membw, self.tgt_gpu.get_name())
    
    def _calc_target_metrics(self, profiled_df: pd.DataFrame, metrics: List[str],
                             ref_gpu: GPU, tgt_gpu: GPU, tensor_prec: str) -> Dict[str, List[float]]:
        """Calculate metrics for target hardware"""
        results = {
            't_kernel_lower': [], 't_kernel_upper': [], 't_kernel_mid': [],
            't_othernode': [], 't_total_lower': [], 't_total_upper': [], 't_total_mid': [],
            'drama_ref': [], 'tensor_ref': [], 'fp64a_ref': [], 'fp32a_ref': [], 'fp16a_ref': [],
            'total_dram_tgt_lower': [], 'total_dram_tgt_mid': [], 'total_dram_tgt_upper': [],
            'total_flop_tgt_lower': [], 'total_flop_tgt_mid': [], 'total_flop_tgt_upper': []
        }
        
        scale_calc = ScaleCalculator(ref_gpu, tgt_gpu, tensor_prec)
        
        for row in profiled_df.itertuples(index=False):
            mv = MetricValues.from_row(row, metrics)
            
            # Store reference metrics
            results['drama_ref'].append(mv.drama)
            results['tensor_ref'].append(mv.tenso)
            results['fp64a_ref'].append(mv.fp64a)
            results['fp32a_ref'].append(mv.fp32a)
            results['fp16a_ref'].append(mv.fp16a)
            
            # Calculate intensities
            intensities = self.intensity_calc.metric_intensities(mv)

            # Calculate reference components
            ref_components = self.time_calc.calc_components(mv)
            
            # Estimate Warps on target GPU
            scale_calc.refresh_smocc(intensities['smocc_gract'])
            
            # Calculate kernel scales using smocc, dram, tensor, fp64, fp32, fp16
            smocc_lower, smocc_mid, smocc_upper = scale_calc.smocc_scale()
            dram_lower, dram_mid, dram_upper = scale_calc.dram_scale(intensities['drama_gract'])
            tensor_lower, tensor_mid, tensor_upper = scale_calc.tensor_scale(intensities['tenso_gract'])
            fp64_lower, fp64_mid, fp64_upper = scale_calc.fp64_scale(intensities['fp64a_gract'])
            fp32_lower, fp32_mid, fp32_upper = scale_calc.fp32_scale(intensities['fp32a_gract'])
            fp16_lower, fp16_mid, fp16_upper = scale_calc.fp16_scale(intensities['fp16a_gract'])
            
            # Estimate bandwidth and flop
            dram_lower_tgt, dram_mid_tgt, dram_upper_tgt = scale_calc.est_dram_tgt(intensities['drama_gract'])  
            flop_lower_tgt, flop_mid_tgt, flop_upper_tgt = scale_calc.est_flop_tgt(
                intensities['tenso_gract'], intensities['fp64a_gract'], intensities['fp32a_gract'], intensities['fp16a_gract'], tensor_prec
            )
            for key, value in {
                'total_dram_tgt_lower': dram_lower_tgt,
                'total_dram_tgt_mid': dram_mid_tgt,
                'total_dram_tgt_upper': dram_upper_tgt,
                'total_flop_tgt_lower': flop_lower_tgt,
                'total_flop_tgt_mid': flop_mid_tgt,
                'total_flop_tgt_upper': flop_upper_tgt,
            }.items():
                results[key].append(value)

            # Select bounded resources
            kernel_scale_lower = min(smocc_lower, dram_lower, tensor_lower, fp64_lower, fp32_lower, fp16_lower)
            kernel_scale_mid = min(smocc_mid, dram_mid, tensor_mid, fp64_mid, fp32_mid, fp16_mid)
            kernel_scale_upper = min(smocc_upper, dram_upper, tensor_upper, fp64_upper, fp32_upper, fp16_upper)

            # Calculate kernel times for each scenario
            for scale, suffix in [(kernel_scale_lower, 'lower'), (kernel_scale_mid, 'mid'), (kernel_scale_upper, 'upper')]:
                t_kernel = ref_components.t_kernel / scale if scale != 0 else 0
                results[f't_kernel_{suffix}'].append(t_kernel)
            
            # Other node time (unchanged)
            t_othernode = ref_components.t_othernode
            results['t_othernode'].append(t_othernode)
            
            # Calculate totals
            results['t_total_lower'].append(results['t_kernel_lower'][-1] + t_othernode)
            results['t_total_mid'].append(results['t_kernel_mid'][-1] + t_othernode)
            results['t_total_upper'].append(results['t_kernel_upper'][-1] + t_othernode)

        return results
        
    def _calc_est_flops(self, results: Dict[str, List]) -> Dict[str, float]:
        """Calculate estimated FLOPS"""
        return {
            "flop_smocc_lower": np.mean(results.get('total_flop_tgt_lower')),
            "flop_smocc_mid": np.mean(results.get('total_flop_tgt_mid')),
            "flop_smocc_upper": np.mean(results.get('total_flop_tgt_upper'))
        }
    
    def _calc_est_membw(self, results: Dict[str, List]) -> Dict[str, float]:
        """Calculate estimated memory bandwidth"""
        return {
            "dram_smocc_lower": np.mean(results.get('total_dram_tgt_lower')),
            "dram_smocc_mid": np.mean(results.get('total_dram_tgt_mid')),
            "dram_smocc_upper": np.mean(results.get('total_dram_tgt_upper'))
        }


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='GPU Performance Profiler and Predictor',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('-f', '--dcgm_file', required=True, help='DCGM output file path')
    parser.add_argument('-d', '--sample_interval_ms', type=int, required=True, help='Sample interval in milliseconds')
    parser.add_argument('-st', '--start_timestamp', type=int, default=0, help='Start timestamp (ms, default: 0)')
    parser.add_argument('-et', '--end_timestamp', type=int, default=None, help='End timestamp (ms, default: None)')
    parser.add_argument('-o', '--overall_runtime_ms', type=int, required=True, help='Overall runtime in milliseconds')
    parser.add_argument('-rg', '--ref_gpu', required=True, choices=list(GPUSpec.keys()), help='Reference GPU')
    parser.add_argument('-tg', '--tgt_gpu', choices=list(GPUSpec.keys()), help='Target GPU(optional)')
    parser.add_argument('--metrics', type=lambda s: s.split(','), required=True, help='Comma-separated list of metrics (e.g., GRACT,DRAMA,SMOCC)')
    parser.add_argument('-tp', '--tensor_precision', required=True, choices=['tf64', 'tf32', 'tf16'], help='Tensor precision type')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Process metrics file
    profiled_df = MetricsProcessor.process_file(args.dcgm_file, args.metrics)

    # Create and run reference profiler
    ref_profiler = ReferenceProfiler(args.sample_interval_ms, args.ref_gpu)
    ref_profiler.run(
        profiled_df, args.metrics, args.overall_runtime_ms,
        args.start_timestamp, args.end_timestamp, args.tensor_precision
    )
    
    # Create target predictor and run if specified
    if args.tgt_gpu:
        tgt_predictor = TargetPredictor(args.sample_interval_ms, args.ref_gpu, args.tgt_gpu)
        tgt_predictor.run(
            profiled_df, args.metrics, args.overall_runtime_ms,
            args.start_timestamp, args.end_timestamp, args.tensor_precision
        )


if __name__=="__main__":
    main()