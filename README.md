# Profiling GPU Performance using DCGM Metrics 

`berkeleyGW`: BerkeleyGW Performance Profiling

`gemm`: GEMM Performance Profiling 

`gpu_util`: GPU Utilization Performance Profiling

`dcgm_analyze.py`: Analyze the raw results from DCGM metrics and plot some figures

`perf_modeling.py`: Modeling and predicting performance of various applications on difference GPUs.


## DCGM Profiling on Perlmutter

## DCGM Profiling on Lawrencium/Einsteinium

DCGM is not currently installed on Lawrencium/Einsteinium systems. To deploy DCGM on these clusters, we'll use a container-based approach with Singularity, which is the default container tool. Use the following commands for deployment:

1. Start dcgm-instance using 4.4.1-2-ubuntu22.04 (the latest version at the time of writing), and `--fakeroot` is the key option.

```
singularity instance start --fakeroot --nv --writable-tmpfs --bind /tmp:/tmp --network=none docker://nvidia/dcgm:4.4.1-2-ubuntu22.04 dcgm-instance
```

2. Start DCGM engine in background

```
singularity exec instance://dcgm-instance nv-hostengine -n &
```

3. Using container-based `dcgmi dmon`, which usually defined in `wrap-dcgmi.sh`.

```
singularity exec instance://dcgm-instance dcgmi dmon -d $dcgm_delay -i 0 -e $dcgm_metrics > $RESULTS_DIR/$dcgm_outfile &
```