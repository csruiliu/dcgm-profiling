# LAMMPS


```bash
git clone https://gitlab.com/NERSC/N10-benchmarks/exaalt.git (from N-10 benchmark])

cd exaalt

./build_lammps_PM.sh

cd benchmarks

cd 0_nano (or 1_micro, 2_tiny, 3_small, 4_medium, 5_reference, 6_target)

sbatch run_nano_A100.sh (or run_tiny_A100.sh, run_small_A100.sh, run_medium_A100.sh, run_reference_A100.sh, run_target_A100.sh)
```


----
Reference:

1. N10 Benchmark, https://gitlab.com/NERSC/N10-benchmarks/exaalt
