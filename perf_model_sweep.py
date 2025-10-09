import pandas as pd
import numpy as np
import os
from pathlib import Path
import argparse
import json
import glob
from dataclasses import dataclass
import matplotlib.pyplot as plt


@dataclass
class Device:
    name: str
    fp16: float
    fp32: float
    fp64: float
    membw: float
    tf16: float
    tf32: float
    tf64: float
    pcie: float
    nvlink: float
    alpha_gpu: float
    alpha_cpu: float



def main():
    ###################################
    # get all parameters
    ###################################
    
if __name__=="__main__":
    main()