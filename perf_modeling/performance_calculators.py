import numpy as np

from typing import Dict, Tuple, Optional

from data_classes import MetricValues
from gpu_specs import GPU
from data_classes import TimeComponents, TimeSlice


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


class TimeCalculator:
    """Handles time-related calculations"""
    
    def __init__(self, sample_interval_ms: float, ref_gpu: GPU):
        self.sample_intv = sample_interval_ms / 1000
        self.gpu = ref_gpu
    
    def calc_components_sg(self, metrics: MetricValues) -> TimeComponents:
        """Calculate time components from metrics"""
        t_flop = self.sample_intv * metrics.get_flop_sum()
        t_dram = self.sample_intv * metrics.drama
        t_kernel = self.sample_intv * metrics.gract
                
        t_othernode = max(self.sample_intv * (1 - metrics.gract), 0)
        
        return TimeComponents(
            t_flop=t_flop,
            t_dram=t_dram,
            t_kernel=t_kernel,
            t_nvlink=0,
            t_othernode=t_othernode
        )
    
    def calc_components_mg(self, metrics: MetricValues) -> TimeComponents:
        """Calculate time components from metrics"""
        t_flop = self.sample_intv * metrics.get_flop_sum()
        t_dram = self.sample_intv * metrics.drama
        t_kernel = self.sample_intv * metrics.gract
        
        t_pcie = ((metrics.pcitx + metrics.pcirx) * self.sample_intv / 
                  (1e9 * self.gpu.get_specs("pcie_bw")))
        
        t_nvlink = ((metrics.nvltx + metrics.nvlrx) * self.sample_intv / 
                    (1e9 * self.gpu.get_specs("nvlink_bw")))
        
        t_othernode = max(self.sample_intv * (1 - metrics.gract) - t_nvlink, 0)
        
        return TimeComponents(
            t_flop=t_flop,
            t_dram=t_dram,
            t_kernel=t_kernel,
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