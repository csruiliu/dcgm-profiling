import argparse
import re
import os
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
from collections import Counter

from gpu_specs import GPU, GPUSpec
    

class MetricsProcessor:
    """Handles metrics file processing for single or multiple GPUs"""
    GPU_PATTERN = re.compile(r'^GPU \d+\s')
    HEADER_PATTERN = re.compile(r'^#Entity')

    def __init__(self, num_gpu: int, metric_names: List[str]):
        self.num_gpu = num_gpu
        self.metric_names = metric_names

    @staticmethod
    def is_float(value: str) -> bool:
        """Check if a string can be converted to float"""
        try:
            float(value)
            return True
        except ValueError:
            return False

    def process_files(self, dcgm_input: str) -> List[pd.DataFrame]:
        """Process input files or directory"""
        if os.path.isdir(dcgm_input):
            print(f"Processing folder: {dcgm_input}")
            file_paths = self._scan_and_organize_gpu_files(dcgm_input)
            return self._process_multiple_files(file_paths)
        elif os.path.isfile(dcgm_input):
            print(f"Processing single file with {self.num_gpu} GPUs: {dcgm_input}")
            return self._process_single_file(dcgm_input)
        else:
            raise ValueError(f"Input path '{dcgm_input}' is neither a valid file nor a directory")

    def _process_single_file(self, file_path: str) -> List[pd.DataFrame]:
        """Process a single file containing multiple GPU data"""
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        header_columns, metric_indices = self._parse_header(lines)
        gpu_data = self._extract_multi_gpu_data(lines, metric_indices, len(header_columns))
        
        return self._create_dataframes(gpu_data)

    def _process_multiple_files(self, file_paths: List[str]) -> List[pd.DataFrame]:
        """Process multiple files, each containing single GPU data"""
        gpu_dfs = []
        
        for logical_gpu_id, file_path in enumerate(file_paths):
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"File not found: {file_path}")
            
            print(f"Processing file {file_path} as logical GPU {logical_gpu_id}")
            
            with open(file_path, 'r') as file:
                lines = file.readlines()
            
            header_columns, metric_indices = self._parse_header(lines)
            gpu_data = self._extract_single_gpu_data(lines, metric_indices, len(header_columns))
            
            if gpu_data:
                df = pd.DataFrame(gpu_data, columns=self.metric_names)
                gpu_dfs.append(df)
                print(f"Logical GPU {logical_gpu_id}: Created DataFrame with {len(gpu_data)} rows")
            else:
                gpu_dfs.append(pd.DataFrame(columns=self.metric_names))
                print(f"Warning: No data found for logical GPU {logical_gpu_id} in file {file_path}")
        
        return gpu_dfs

    def _parse_header(self, lines: List[str]) -> Tuple[List[str], List[int]]:
        """Parse header and find metric column indices"""
        for line in lines:
            if self.HEADER_PATTERN.match(line):
                header_columns = [col.strip() for col in re.split(r'\s{2,}', line.strip())]
                metric_indices = self._get_metric_indices(header_columns)
                return header_columns, metric_indices
        raise ValueError("Could not find header line in the data file")

    def _get_metric_indices(self, header_columns: List[str]) -> List[int]:
        """Map requested metrics to their column indices"""
        metric_indices = []
        for metric in self.metric_names:
            if metric not in header_columns:
                raise ValueError(
                    f"Metric '{metric}' not found in data file. "
                    f"Available metrics: {header_columns[1:]}"
                )
            metric_indices.append(header_columns.index(metric) - 1)
        return metric_indices

    def _extract_multi_gpu_data(self, lines: List[str], metric_indices: List[int], 
                                num_columns: int) -> Dict[int, List[List[float]]]:
        """Extract data from a file with multiple GPUs"""
        gpu_data = {}
        
        for line in lines:
            if self.HEADER_PATTERN.match(line):
                continue
            if self.GPU_PATTERN.match(line):
                parts = re.split(r'\s{3,}', line.strip())
                
                # Extract GPU ID
                gpu_match = re.search(r'GPU (\d+)', parts[0])
                if not gpu_match:
                    continue
                
                gpu_id = int(gpu_match.group(1))
                
                # Extract numeric values
                values = parts[1:]
                numeric_values = [0.0 if v.strip().lower() == 'n/a' else float(v) 
                                for v in values if self.is_float(v) or v.strip().lower() == 'n/a']
                
                if len(numeric_values) >= num_columns - 1:
                    selected_values = [numeric_values[i] for i in metric_indices]
                    
                    if gpu_id not in gpu_data:
                        gpu_data[gpu_id] = []
                    gpu_data[gpu_id].append(selected_values)
                else:
                    print(f"Warning: Line has insufficient data columns: {line.strip()}")
        
        return gpu_data

    def _extract_single_gpu_data(self, lines: List[str], metric_indices: List[int], 
                                 num_columns: int) -> List[List[float]]:
        """Extract data from a file with single GPU (all GPU lines are for same logical GPU)"""
        gpu_data = []
        
        for line in lines:
            if self.HEADER_PATTERN.match(line):
                continue
            if self.GPU_PATTERN.match(line):
                parts = re.split(r'\s{3,}', line.strip())
                values = parts[1:]
                numeric_values = [0.0 if v.strip().lower() == 'n/a' else float(v) 
                                for v in values if self.is_float(v) or v.strip().lower() == 'n/a']
                
                if len(numeric_values) >= num_columns - 1:
                    selected_values = [numeric_values[i] for i in metric_indices]
                    gpu_data.append(selected_values)
                else:
                    print(f"Warning: Line has insufficient data columns: {line.strip()}")
        
        return gpu_data

    def _create_dataframes(self, gpu_data: Dict[int, List[List[float]]]) -> List[pd.DataFrame]:
        """Create DataFrames from GPU data dictionary"""
        gpu_dfs = []
        for gpu_id in sorted(gpu_data.keys()):
            if gpu_data[gpu_id]:
                df = pd.DataFrame(gpu_data[gpu_id], columns=self.metric_names)
                gpu_dfs.append(df)
            else:
                gpu_dfs.append(pd.DataFrame(columns=self.metric_names))
        
        return gpu_dfs

    def _scan_and_organize_gpu_files(self, folder_path: str) -> List[str]:
        """Scan a folder for GPU data files and organize them by logical GPU ID"""
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        file_patterns = ['*.out', '*.txt']
        all_files = []
        for pattern in file_patterns:
            all_files.extend(folder_path.glob(pattern))
        
        if not all_files:
            raise FileNotFoundError(f"No data files found in {folder_path}")
        
        print(f"Found {len(all_files)} potential GPU data files in {folder_path}")
        
        return self._organize_by_file_content(all_files)

    def _organize_by_file_content(self, all_files: List[Path]) -> List[str]:
        """Organize files by analyzing their content"""
        file_info = []
        
        for file_path in all_files:
            try:
                with open(file_path, 'r') as f:
                    content = ""
                    for i, line in enumerate(f):
                        content += line
                        if i > 10:
                            break
                
                gpu_pattern = re.compile(r'GPU (\d+)')
                gpu_matches = gpu_pattern.findall(content)
                
                if gpu_matches:
                    gpu_counter = Counter(gpu_matches)
                    most_common_gpu_id = int(gpu_counter.most_common(1)[0][0])
                    total_lines = len(gpu_matches)
                    file_info.append((file_path, most_common_gpu_id, total_lines))
                else:
                    print(f"Warning: No GPU data found in {file_path}")
                    file_info.append((file_path, -1, 0))
            
            except Exception as e:
                print(f"Warning: Could not read file {file_path}: {e}")
                file_info.append((file_path, -1, 0))
        
        # Sort by filename for deterministic ordering
        file_info.sort(key=lambda x: x[0].name)
        
        if len(file_info) != self.num_gpu:
            print(f"Content-based matching found {len(file_info)} valid files, expected {self.num_gpu}.")
            raise ValueError(f"Expected {self.num_gpu} files but found {len(file_info)}")
        
        # Sort by GPU ID and line count
        file_info.sort(key=lambda x: (x[1], x[2]))
        
        organized_files = [str(info[0]) for info in file_info]
        
        print("File organization by content analysis:")
        for logical_gpu_id, (file_path, detected_gpu_id, line_count) in enumerate(file_info):
            if detected_gpu_id >= 0:
                print(f"  Logical GPU {logical_gpu_id}: {file_path.name} "
                      f"(detected GPU {detected_gpu_id}, {line_count} data lines)")
            else:
                print(f"  Logical GPU {logical_gpu_id}: {file_path.name} "
                      f"(GPU ID unknown, {line_count} data lines)")
        
        return organized_files


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
            pcitx=getattr(row, 'PCITX', 0.0) if 'PCITX' in metrics else 0.0,
            pcirx=getattr(row, 'PCIRX', 0.0) if 'PCIRX' in metrics else 0.0,
            nvltx=getattr(row, 'NVLTX', 0.0) if 'NVLTX' in metrics else 0.0,
            nvlrx=getattr(row, 'NVLRX', 0.0) if 'NVLRX' in metrics else 0.0,
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
    t_pcie: float = 0.0
    t_nvlink: float = 0.0
    t_othernode: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary"""
        return {
            't_flop': self.t_flop,
            't_dram': self.t_dram,
            't_kernel': self.t_kernel,
            't_pcie': self.t_pcie,
            't_nvlink': self.t_nvlink,
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


class TimeCalculator:
    """Handles time-related calculations"""
    
    def __init__(self, sample_interval_ms: float, gpu: GPU):
        self.sample_intv = sample_interval_ms / 1000
        self.gpu = gpu
    
    def calc_components(self, metrics: MetricValues) -> TimeComponents:
        """Calculate time components from metrics"""
        t_flop = self.sample_intv * metrics.get_flop_sum()
        t_dram = self.sample_intv * metrics.drama
        t_kernel = self.sample_intv * metrics.gract
        
        t_pcie = ((metrics.pcitx + metrics.pcirx) * self.sample_intv / 
                  (1e9 * self.gpu.get_specs("pcie_bw")))
        
        t_nvlink = ((metrics.nvltx + metrics.nvlrx) * self.sample_intv / 
                    (1e9 * self.gpu.get_specs("nvlink_bw")))
        
        t_othernode = max(self.sample_intv * (1 - metrics.gract) - t_pcie - t_nvlink, 0)
        
        return TimeComponents(
            t_flop=t_flop,
            t_dram=t_dram,
            t_kernel=t_kernel,
            t_pcie=t_pcie,
            t_nvlink=t_nvlink,
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
        if metrics.gract == 0:
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
    """Calculates scaling factors for target GPU"""

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
        self.pcie_ratio = self._get_ratio("pcie_bw")
        self.nvlink_ratio = self._get_ratio("nvlink_bw")

    def _get_ratio(self, spec: str) -> float:
        """Helper to compute target/reference ratio for a given spec"""
        ref_val = self.ref_gpu.get_specs(spec)
        tgt_val = self.tgt_gpu.get_specs(spec)
        return tgt_val / ref_val if ref_val != 0 else 0

    def _estimate_warps(self):
        """Estimate warps on target GPU"""
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
        """Update SMOCC and recalculate warps"""
        self.cur_smocc = smocc_ref
        self._estimate_warps()

    def smocc_scale(self) -> Tuple[float, float, float]:
        """Calculate SM occupancy scaling factors"""
        k_smocc_ref = self._compute_k_smocc(self.cur_warps_ref, self.ref_gpu)
        
        for level in ['lower', 'mid', 'upper']:
            k_smocc_tgt = self._compute_k_smocc(self.cur_warps_tgt[level], self.tgt_gpu)
            self.scale_smocc[level] = k_smocc_tgt / k_smocc_ref if k_smocc_ref != 0 else 0
        
        return self.scale_smocc['lower'], self.scale_smocc['mid'], self.scale_smocc['upper']

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

    def pcie_scale(self) -> float:
        """Calculate PCIe bandwidth scaling factor"""
        return self.pcie_ratio

    def nvlink_scale(self) -> float:
        """Calculate NVLink bandwidth scaling factor"""
        return self.nvlink_ratio

    def _compute_scale(self, intensity_ref: float, ratio: float) -> Tuple[float, float, float]:
        """Generic method to compute scaling factors for any intensity metric"""
        if intensity_ref < self.INTENSITY_THRESHOLD:
            return np.inf, np.inf, np.inf
        
        scale_factor = ratio / intensity_ref if intensity_ref != 0 else 0
        return (
            min(self.scale_smocc['lower'], scale_factor),
            min(self.scale_smocc['mid'], scale_factor),
            min(self.scale_smocc['upper'], scale_factor)
        )


class ResultsFormatter:
    """Formats and prints results"""
    
    @staticmethod
    def print_reference_results(metrics: Dict[str, List[float]], flops: float, 
                               mem_bw: float, gpu_name: str):
        """Print reference hardware results"""
        print(f"\n{'='*60}")
        print(f"Reference Hardware: {gpu_name}")
        
        print(f"\nEstimated Kernel Time: {sum(metrics['t_kernel']):.2f} s")
        print(f"Estimated PCIe Time: {sum(metrics['t_pcie']):.2f} s")
        print(f"Estimated NVLink Time: {sum(metrics['t_nvlink']):.2f} s")
        print(f"Estimated Other Node Time: {sum(metrics['t_othernode']):.2f} s")
        print(f"Estimated Total Runtime: {sum(metrics['t_total']):.2f} s")
        print(f"{'='*60}\n")
    
    @staticmethod
    def print_target_results(metrics: Dict[str, List[float]], flops: float, 
                            mem_bw: float, gpu_name: str):
        """Print target hardware results"""
        print(f"\n{'='*60}")
        print(f"Target Hardware: {gpu_name}")
        
        print(f"\nEstimated Kernel Time [Lower SMOCC]: {sum(metrics['t_kernel_lower']):.2f} s")
        print(f"Estimated Kernel Time [Mid SMOCC]:   {sum(metrics['t_kernel_mid']):.2f} s")
        print(f"Estimated Kernel Time [Upper SMOCC]: {sum(metrics['t_kernel_upper']):.2f} s")
        
        print(f"\nEstimated PCIe Time: {sum(metrics['t_pcie']):.2f} s")
        print(f"Estimated NVLink Time: {sum(metrics['t_nvlink']):.2f} s")
        print(f"Estimated Other Node Time: {sum(metrics['t_othernode']):.2f} s")
        
        print(f"\nEstimated Total Runtime [Lower SMOCC]: {sum(metrics['t_total_lower']):.2f} s")
        print(f"Estimated Total Runtime [Mid SMOCC]:   {sum(metrics['t_total_mid']):.2f} s")
        print(f"Estimated Total Runtime [Upper SMOCC]: {sum(metrics['t_total_upper']):.2f} s")
        print(f"{'='*60}\n")


class BaseProfiler(ABC):
    """Abstract base class for profilers"""
    
    def __init__(self, sample_interval_ms: float):
        self.sample_interval_ms = sample_interval_ms
        self.intensity_calc = MetricIntensityCalculator()
        self.formatter = ResultsFormatter()
        
    @abstractmethod
    def run(self, *args, **kwargs):
        """Run the profiling/prediction"""
        pass


class ReferenceProfiler(BaseProfiler):
    """Profiles performance on reference hardware for multiple GPUs"""
    
    def __init__(self, sample_interval_ms: float, gpu_name: str):
        self.gpu = GPU(gpu_name=gpu_name)
        super().__init__(sample_interval_ms)
        self.time_calc = TimeCalculator(sample_interval_ms, self.gpu)

    def run(self, gpu_dfs: List[pd.DataFrame], metrics: List[str], 
            overall_runtime_ms: float, agg_interval_ms: float,
            start_ts: Optional[float], end_ts: Optional[float], 
            tensor_prec: str):
        """Model performance on reference hardware for multiple GPUs"""
        
        # Process each GPU
        all_gpu_metrics = []
        for gpu_id, df in enumerate(gpu_dfs):
            if df.empty:
                raise ValueError(f"GPU {gpu_id} DataFrame is empty")
            
            # print(f"\nProcessing GPU {gpu_id}...")
            gpu_metrics = self._process_single_gpu(df, metrics, overall_runtime_ms, 
                                                   start_ts, end_ts, tensor_prec)
            all_gpu_metrics.append(gpu_metrics)
        
        # Aggregate across GPUs
        aggregated_metrics = self._aggregate_multi_gpu(all_gpu_metrics, agg_interval_ms)
        
        # Calculate overall performance metrics
        flops = self._calc_flops(aggregated_metrics, tensor_prec)
        mem_bw = self._calc_mem_bw(aggregated_metrics)
        
        # Print results
        self.formatter.print_reference_results(aggregated_metrics, flops, mem_bw, 
                                              self.gpu.get_name())
    
    def _process_single_gpu(self, df: pd.DataFrame, metrics: List[str],
                           overall_runtime_ms: float, start_ts: Optional[float],
                           end_ts: Optional[float], tensor_prec: str) -> Dict[str, List[float]]:
        """Process a single GPU's data"""
        # Calculate components for all rows
        components_list = [
            self.time_calc.calc_components(MetricValues.from_row(row, metrics))
            for row in df.itertuples(index=False)
        ]
        
        # Get time slice
        time_slice = self.time_calc.get_time_slice(
            overall_runtime_ms, start_ts, end_ts, len(components_list)
        )
        
        # Slice and aggregate components
        sliced = self._slice_and_aggregate(components_list, time_slice)
        
        return sliced
    
    def _slice_and_aggregate(self, components_list: List[TimeComponents], 
                            time_slice: TimeSlice) -> Dict[str, List[float]]:
        """Slice components and add total time"""
        sliced = {
            key: [comp.to_dict()[key] for comp in components_list][time_slice.start_idx:time_slice.end_idx]
            for key in components_list[0].to_dict().keys()
        }
        
        # Add total time
        sliced['t_total'] = [
            sliced['t_kernel'][i] + sliced['t_pcie'][i] + 
            sliced['t_nvlink'][i] + sliced['t_othernode'][i]
            for i in range(len(sliced['t_kernel']))
        ]
        
        return sliced
    
    def _aggregate_multi_gpu(self, all_gpu_metrics: List[Dict[str, List[float]]], 
                            agg_interval_ms: float) -> Dict[str, List[float]]:
        """Aggregate metrics across multiple GPUs"""
        # Check all GPUs have same length
        lengths = [len(m['t_total']) for m in all_gpu_metrics]
        if len(set(lengths)) != 1:
            raise ValueError("Not all GPU metric lists are of the same length!")
        
        num_rows = lengths[0]
        agg_samples = agg_interval_ms // self.sample_interval_ms
        
        # Aggregate by taking max across GPUs for each time window
        aggregated = {key: [] for key in all_gpu_metrics[0].keys()}
        
        for start in range(0, num_rows, agg_samples):
            end = min(start + agg_samples, num_rows)
            
            for key in aggregated.keys():
                # Sum within each GPU's window, then take max across GPUs
                window_values = [
                    sum(gpu_metrics[key][row_idx] for row_idx in range(start, end))
                    for gpu_metrics in all_gpu_metrics
                ]
                aggregated[key].append(max(window_values))
        
        return aggregated
    
    def _calc_flops(self, sliced: Dict[str, List[float]], tensor_prec: str) -> float:
        """Calculate FLOPS"""
        return (np.mean(sliced['t_flop']) / self.time_calc.sample_intv * 
                self.gpu.get_specs(tensor_prec))
    
    def _calc_mem_bw(self, sliced: Dict[str, List[float]]) -> float:
        """Calculate memory bandwidth"""
        return (np.mean(sliced['t_dram']) / self.time_calc.sample_intv * 
                self.gpu.get_specs("mem_bw"))


class TargetPredictor(BaseProfiler):
    """Predicts performance on target hardware for multiple GPUs"""
    
    def __init__(self, sample_interval_ms: float, ref_gpu_name: str, tgt_gpu_name: str):
        self.ref_gpu = GPU(gpu_name=ref_gpu_name)
        self.tgt_gpu = GPU(gpu_name=tgt_gpu_name)
        super().__init__(sample_interval_ms)
        self.time_calc = TimeCalculator(sample_interval_ms, self.ref_gpu)

    def run(self, gpu_dfs: List[pd.DataFrame], metrics: List[str],
            overall_runtime_ms: float, agg_interval_ms: float,
            start_ts: Optional[float], end_ts: Optional[float], 
            tensor_prec: str):
        """Predict performance on target hardware for multiple GPUs"""
        
        # Process each GPU
        all_gpu_metrics = []
        for gpu_id, df in enumerate(gpu_dfs):
            if df.empty:
                raise ValueError(f"GPU {gpu_id} DataFrame is empty")
            
            # print(f"\nPredicting for GPU {gpu_id}...")
            gpu_metrics = self._predict_single_gpu(df, metrics, overall_runtime_ms,
                                                   start_ts, end_ts, tensor_prec)
            all_gpu_metrics.append(gpu_metrics)
        
        # Aggregate across GPUs
        aggregated_metrics = self._aggregate_multi_gpu(all_gpu_metrics, agg_interval_ms)
        
        # Calculate estimated FLOPS and memory bandwidth
        est_flops = self._calc_est_flops(aggregated_metrics, tensor_prec)
        est_mem_bw = self._calc_est_mem_bw(aggregated_metrics)
        
        # Print predictions
        self.formatter.print_target_results(aggregated_metrics, est_flops, est_mem_bw, 
                                           self.tgt_gpu.get_name())
    
    def _predict_single_gpu(self, df: pd.DataFrame, metrics: List[str],
                           overall_runtime_ms: float, start_ts: Optional[float],
                           end_ts: Optional[float], tensor_prec: str) -> Dict[str, List[float]]:
        """Predict performance for a single GPU"""
        # Calculate target metrics
        target_metrics = self._calc_target_metrics(df, metrics, tensor_prec)
        
        # Get time slice
        time_slice = self.time_calc.get_time_slice(
            overall_runtime_ms, start_ts, end_ts, len(target_metrics['t_total_lower'])
        )
        
        # Slice metrics
        sliced_metrics = time_slice.slice_dict(target_metrics)
        
        return sliced_metrics
    
    def _calc_target_metrics(self, df: pd.DataFrame, metrics: List[str],
                            tensor_prec: str) -> Dict[str, List[float]]:
        """Calculate metrics for target hardware"""
        results = {
            't_kernel_lower': [], 't_kernel_mid': [], 't_kernel_upper': [],
            't_pcie': [], 't_nvlink': [], 't_othernode': [],
            't_total_lower': [], 't_total_mid': [], 't_total_upper': [],
            'drama_ref': [], 'tensor_ref': [], 'fp64a_ref': [], 
            'fp32a_ref': [], 'fp16a_ref': []
        }
        
        scale_calc = ScaleCalculator(self.ref_gpu, self.tgt_gpu, tensor_prec)
        
        for row in df.itertuples(index=False):
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
            
            # Calculate kernel scales
            smocc_lower, smocc_mid, smocc_upper = scale_calc.smocc_scale()
            dram_lower, dram_mid, dram_upper = scale_calc.dram_scale(intensities['drama_gract'])
            tensor_lower, tensor_mid, tensor_upper = scale_calc.tensor_scale(intensities['tenso_gract'])
            fp64_lower, fp64_mid, fp64_upper = scale_calc.fp64_scale(intensities['fp64a_gract'])
            fp32_lower, fp32_mid, fp32_upper = scale_calc.fp32_scale(intensities['fp32a_gract'])
            fp16_lower, fp16_mid, fp16_upper = scale_calc.fp16_scale(intensities['fp16a_gract'])
            
            kernel_scale_lower = min(smocc_lower, dram_lower, tensor_lower, 
                                    fp64_lower, fp32_lower, fp16_lower)
            kernel_scale_mid = min(smocc_mid, dram_mid, tensor_mid, 
                                  fp64_mid, fp32_mid, fp16_mid)
            kernel_scale_upper = min(smocc_upper, dram_upper, tensor_upper, 
                                    fp64_upper, fp32_upper, fp16_upper)
            
            # Calculate kernel times for each scenario
            for scale, suffix in [(kernel_scale_lower, 'lower'), 
                                 (kernel_scale_mid, 'mid'), 
                                 (kernel_scale_upper, 'upper')]:
                t_kernel = ref_components.t_kernel / scale if scale != 0 else 0
                results[f't_kernel_{suffix}'].append(t_kernel)
            
            # Calculate communication times
            pcie_scale = scale_calc.pcie_scale()
            nvlink_scale = scale_calc.nvlink_scale()
            
            t_pcie = ref_components.t_pcie / pcie_scale if pcie_scale != 0 else 0
            t_nvlink = ref_components.t_nvlink / nvlink_scale if nvlink_scale != 0 else 0
            
            results['t_pcie'].append(t_pcie)
            results['t_nvlink'].append(t_nvlink)
            
            # Other node time (unchanged)
            t_othernode = ref_components.t_othernode
            results['t_othernode'].append(t_othernode)
            
            # Calculate totals
            results['t_total_lower'].append(results['t_kernel_lower'][-1] + t_pcie + 
                                           t_nvlink + t_othernode)
            results['t_total_mid'].append(results['t_kernel_mid'][-1] + t_pcie + 
                                         t_nvlink + t_othernode)
            results['t_total_upper'].append(results['t_kernel_upper'][-1] + t_pcie + 
                                           t_nvlink + t_othernode)
        
        return results
    
    def _aggregate_multi_gpu(self, all_gpu_metrics: List[Dict[str, List[float]]], 
                            agg_interval_ms: float) -> Dict[str, List[float]]:
        """Aggregate metrics across multiple GPUs"""
        # Check all GPUs have same length
        lengths = [len(m['t_total_lower']) for m in all_gpu_metrics]
        if len(set(lengths)) != 1:
            raise ValueError("Not all GPU metric lists are of the same length!")
        
        num_rows = lengths[0]
        agg_samples = agg_interval_ms // self.sample_interval_ms
        
        # Aggregate by taking max across GPUs for each time window
        aggregated = {key: [] for key in all_gpu_metrics[0].keys()}
        
        for start in range(0, num_rows, agg_samples):
            end = min(start + agg_samples, num_rows)
            
            for key in aggregated.keys():
                # Sum within each GPU's window, then take max across GPUs
                window_values = [
                    sum(gpu_metrics[key][row_idx] for row_idx in range(start, end))
                    for gpu_metrics in all_gpu_metrics
                ]
                aggregated[key].append(max(window_values))
        
        return aggregated
    
    def _calc_est_flops(self, sliced_metrics: Dict[str, List[float]], 
                       tensor_prec: str) -> float:
        """Calculate estimated FLOPS"""
        return (
            np.mean(sliced_metrics.get('tensor_ref')) * self.tgt_gpu.get_specs(tensor_prec) +
            np.mean(sliced_metrics.get('fp64a_ref')) * self.tgt_gpu.get_specs("fp64") +
            np.mean(sliced_metrics.get('fp32a_ref')) * self.tgt_gpu.get_specs("fp32") +
            np.mean(sliced_metrics.get('fp16a_ref')) * self.tgt_gpu.get_specs("fp16")
        )
    
    def _calc_est_mem_bw(self, sliced_metrics: Dict[str, List[float]]) -> float:
        """Calculate estimated memory bandwidth"""
        return np.mean(sliced_metrics.get('drama_ref')) * self.tgt_gpu.get_specs("mem_bw")


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description='Multi-GPU Performance Profiler and Predictor',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('-f', '--dcgm_input', required=True, help='DCGM input: file or folder path')
    parser.add_argument('-n', '--num_gpu', type=int, required=True, help='Number of GPUs')
    parser.add_argument('-d', '--sample_interval_ms', type=int, required=True, help='Sample interval in milliseconds')
    parser.add_argument('-a', '--agg_interval_ms', type=int, required=True, help='Aggregation interval in milliseconds')
    parser.add_argument('-st', '--start_timestamp', type=int, default=0, help='Start timestamp (ms, default: 0)')
    parser.add_argument('-et', '--end_timestamp', type=int, default=None, help='End timestamp (ms, default: None)')
    parser.add_argument('-o', '--overall_runtime_ms', type=int, required=True, help='Overall runtime in milliseconds')
    parser.add_argument('-rg', '--ref_gpu', required=True, choices=list(GPUSpec.keys()), help='Reference GPU')
    parser.add_argument('-tg', '--tgt_gpu', choices=list(GPUSpec.keys()), help='Target GPU (optional)')
    parser.add_argument('--metrics', type=lambda s: s.split(','), required=True, help='Comma-separated list of metrics')
    parser.add_argument('-tp', '--tensor_precision', required=True, choices=['tf64', 'tf32', 'tf16'], help='Tensor precision type')
    
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Process metrics file
    metrics_processor = MetricsProcessor(args.num_gpu, args.metrics)
    gpu_dfs = metrics_processor.process_files(args.dcgm_input)

    print(f"\nProcessed {len(gpu_dfs)} GPUs")
    for i, df in enumerate(gpu_dfs):
        print(f"GPU {i}: {len(df)} samples")

    # Create and run reference profiler
    ref_profiler = ReferenceProfiler(args.sample_interval_ms, args.ref_gpu)
    ref_profiler.run(
        gpu_dfs, args.metrics, args.overall_runtime_ms, args.agg_interval_ms,
        args.start_timestamp, args.end_timestamp, args.tensor_precision
    )

    # Create target predictor and run if specified
    if args.tgt_gpu:
        tgt_predictor = TargetPredictor(args.sample_interval_ms, args.ref_gpu, args.tgt_gpu)
        tgt_predictor.run(
            gpu_dfs, args.metrics, args.overall_runtime_ms, args.agg_interval_ms,
            args.start_timestamp, args.end_timestamp, args.tensor_precision
        )


if __name__=="__main__":
    main()