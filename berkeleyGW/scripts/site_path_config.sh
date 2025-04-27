#!/bin/bash

#This file sets the paths for all BGW executable, library and input-data.
#It will need to be modified for each install site.
#To minimize the number of files that must be updated with this path data,
#all of the jobscripts will source this file.
#
#Jobscripts will still need to be updated to match
#a) your queue system, or
#b) your compute node configuration.

#N10_BGW=/path/to/berkeleygw-workflow
N10_BGW=$HOME/BerkeleyGW
if [[ -z "${N10_BGW}" ]]; then
    echo "The N10_BGW variable is not defined."
    echo "Please set N10_BGW in site_path_config.sh and try again."
    exit 0
fi

#libraries... you may need to add FFTW, or Scalapack or...
HDF_LIBPATH=
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$HDF_LIBPATH

export N10_BGW=$HOME/BerkeleyGW
export N10_BGW_WORKFLOW=$N10_BGW/berkeleygw-workflow

#executables
BGW_DIR=$N10_BGW/BerkeleyGW-master/bin

#input data
Si_WFN_folder=$N10_BGW_WORKFLOW/Si_WFN_folder
Si214_WFN_folder=$Si_WFN_folder/Si214/WFN_file
Si510_WFN_folder=$Si_WFN_folder/Si510/WFN_file
Si998_WFN_folder=$Si_WFN_folder/Si998/WFN_file
Si2742_WFN_folder=$Si_WFN_folder/Si2742/WFN_file

Si214_Benchmark_folder=$N10_BGW_WORKFLOW/benchmark/small_Si214
Si510_Benchmark_folder_Medium=$N10_BGW_WORKFLOW/benchmark/medium_Si510
Si998_Benchmark_folder_=$N10_BGW_WORKFLOW/benchmark/reference_Si998
Si2742_Benchmark_folder_Small=$N10_BGW_WORKFLOW/benchmark/target_Si2742

#any modules that should be loaded at runtime
module swap PrgEnv-gnu PrgEnv-nvhpc
