import argparse
import pandas as pd
import numpy as np

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

from gpu_specs import GPU, GPUSpec
from data_classes import MetricValues, TimeComponents, TimeSlice
from performance_calculators import MetricIntensityCalculator, ScaleCalculator, TimeCalculator
from job_processor import JobProcessor 
from utils import ResultsFormatter


class BaseProfiler(ABC):
    """Abstract base class for profilers"""
    
    def __init__(self, sample_interval_ms: float, ref_gpu: GPU):
        self.time_calc = TimeCalculator(sample_interval_ms, ref_gpu)
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
        super().__init__(sample_interval_ms, self.gpu)

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
            self.time_calc.calc_components_sg(MetricValues.from_row(row, metrics))
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
        super().__init__(sample_interval_ms, self.ref_gpu)

    def run(self, profiled_df: pd.DataFrame, metrics: List[str],
            overall_runtime_ms: float, start_ts: Optional[float], 
            end_ts: Optional[float], tensor_prec: str):
        """Predict performance on target hardware"""        
        # Calculate target metrics
        target_metrics = self._calc_target_metrics(
            profiled_df, metrics, tensor_prec
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
    
    def _calc_target_metrics(self, profiled_df: pd.DataFrame, metrics: List[str], tensor_prec: str) -> Dict[str, List[float]]:
        """Calculate metrics for target hardware"""
        results = {
            't_kernel_lower': [], 't_kernel_upper': [], 't_kernel_mid': [],
            't_othernode': [], 't_total_lower': [], 't_total_upper': [], 't_total_mid': [],
            'drama_ref': [], 'tensor_ref': [], 'fp64a_ref': [], 'fp32a_ref': [], 'fp16a_ref': [],
            'total_dram_tgt_lower': [], 'total_dram_tgt_mid': [], 'total_dram_tgt_upper': [],
            'total_flop_tgt_lower': [], 'total_flop_tgt_mid': [], 'total_flop_tgt_upper': []
        }
        
        scale_calc = ScaleCalculator(self.ref_gpu, self.tgt_gpu, tensor_prec)
        
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
            ref_components = self.time_calc.calc_components_sg(mv)
            
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
    
    # Process metrics file for a job
    job_processor = JobProcessor(1, args.metrics)
    profiled_df = job_processor.process_files(args.dcgm_file)

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