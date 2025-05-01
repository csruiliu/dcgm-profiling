# Performance Profiling for BerkeleyGW

## Compilation on Perlmutter 

Following the instructions on the [website](https://gitlab.com/NERSC/N10-benchmarks/berkeleygw-workflow) to compile BerkeleyGW on [Permutter](https://docs.nersc.gov/systems/perlmutter/architecture), except for the module loading--using the following instead.

```bash
module swap PrgEnv-gnu PrgEnv-nvidia
module load cray-hdf5-parallel/1.12.2.9
module load cray-fftw/3.3.10.6
module load cray-libsci/23.12.5
module load python 
```

BerkeleyGW has mutiple example makefiles, choose `perlmutter.nersc.gov-nvhpc-openacc.mk` for Permutter

## Compilation on Lawrencium

Compiling Berkeley on Lawrencium is not as well documented as on Perlemutter. The following configuration has been valiedated. 

```bash
module load gcc
module load openmpi
module load intel-oneapi-mkl
module load hdf5/1.14.3
```

Please use `lrc-arch.mk` in this repo as the `arch.mk`.


## Environment Paths

Make sure all the BerkeleyGW-related paths are correct in `scripts/site_path_config.sh`. For instance, `BerkeleyGW` folder, which contains `BerkeleyGW-master` and `berkeleygw-workflow`, should be one level above the current scripts directory.


## DCGM

We use `dcgmi dmon` command to profile the performance under various metrics.

`dcgm_delay` is the parameter for `dcgmi dmon -d` command. It is an integer and in milliseconds. It represents how often to query results from DCGM. [default = 1000 millisecond, Minimum value = 1 millisecond.]