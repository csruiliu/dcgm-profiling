# Performance Modeling and Prediction for GPU-Workloads Using DCGM

We use collected Nvidia DCGM data for performance modeling and prediction. We evaluate the model via various benchmarking applications including GEMM, BableStream, BerkeleyGW, LAMMPS, MILC, etc.

## Containerized DCGM deployment on Perlmutter

The deploy containerized DCGM on Perlmutter, we need to add following option in sbatch scripts.

```
#SBATCH --perf=generic
```

Also, we need to use `podman-hpc` to start containerized dcgm. Usually, we should add it in the sbatch script or interactive script as well.

```
podman-hpc run -d -it --name dcgm-container --rm \
    --gpu --cap-add SYS_ADMIN -p 5555:5555 \
    nvcr.io/nvidia/cloud-native/dcgm:4.2.3-1-ubuntu22.04
```

## Containerized DCGM deployment on Lawrencium/Einsteinium

DCGM is not currently installed on Lawrencium/Einsteinium systems. To deploy DCGM on these clusters, we'll use a container-based approach with Singularity, which is the default container tool. Use the following commands for deployment:

1. Start dcgm-instance using 4.4.1-2-ubuntu22.04 (the latest version at the time of writing), and `--fakeroot` is the key option.

```
singularity instance start \
  --fakeroot \
  --nv \
  --writable-tmpfs \
  --bind /tmp:/tmp \
  --network=none \
  docker://nvidia/dcgm:4.4.1-2-ubuntu22.04 \
  dcgm-instance
```

2. Start DCGM engine in background

```
singularity exec instance://dcgm-instance nv-hostengine -n &
```

3. Using container-based `dcgmi dmon`, which usually defined in `wrap-dcgmi.sh`.

```
singularity exec instance://dcgm-instance \
    dcgmi dmon -d $dcgm_delay -i 0 -e $dcgm_metrics \
    > $RESULTS_DIR/$dcgm_outfile &
```