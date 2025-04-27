# Performance Profiling for BerkeleyGW

Following the instructions on the [website](https://gitlab.com/NERSC/N10-benchmarks/berkeleygw-workflow), except for the module loading--using the following instead.

```bash
module swap PrgEnv-gnu PrgEnv-nvidia
module load cray-hdf5-parallel/1.12.2.9
module load cray-fftw/3.3.10.6
module load cray-libsci/23.12.5
module load python 
```

Make sure all the BerkeleyGW-related paths are correct in `scripts/site_path_config.sh`. For instance, `BerkeleyGW` folder, which contains `BerkeleyGW-master` and `berkeleygw-workflow`, should be one level above the current scripts directory.

## DCGM

We use `dcgmi dmon` command to profile the performance under various metrics.

`dcgm_delay` is the parameter for `dcgmi dmon -d` command. It is an integer and in milliseconds. It represents how often to query results from DCGM. [default = 1000 millisecond, Minimum value = 1 millisecond.]