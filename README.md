# GRB_Analysis_for_PBH
This repository provides a toolkit for performing energy-based data analysis, statistical modeling, and batch execution of analysis tasks. It is designed to support configurable analysis workflows through parameter files and reusable utility modules.

1. Repository Structure

├── energy_analysis.py      # Main analysis script

├── batch_run.py            # Batch execution script

├── parameter.txt           # Configuration parameters for analysis

├── util/
│   ├── filelib.py          # File I/O utilities
│   ├── likelihood.py       # Likelihood and model-related functions
│   ├── statslib.py         # Statistical analysis utilities
│   └── plotlib.py          # Plotting and visualization utilities


3. Description of Components
(a) energy_analysis.py

The main analysis script. It performs energy-based analysis using parameters defined in parameter.txt, and relies on helper functions provided in the util module.

(b) batch_run.py

A batch execution script that allows running energy_analysis.py multiple times with different configurations or input datasets. This is useful for parameter scans, Monte Carlo studies, or large-scale experiments.

(c) parameter.txt

A plain-text configuration file containing analysis parameters (e.g., energy ranges, bin widths, thresholds, or model settings).
This file allows users to modify analysis behavior without changing the source code.

(d) util/

A collection of reusable utility modules:

filelib.py
Functions for reading, writing, and managing input/output files.

likelihood.py
Implements likelihood functions and related model evaluation routines.

statslib.py
Statistical tools for data analysis, uncertainty estimation, and hypothesis testing.

plotlib.py
Plotting and visualization utilities for generating figures from analysis results.


3. Requirements

Python 3.x

Common scientific Python libraries (e.g., numpy, scipy, matplotlib)


4. Usage
Example of running a Single Analysis
>> python3 energy_analysis.py --param-file parameter.txt --fits-file $FIT_FILE_PATH$

Make sure parameter.txt is properly configured before running.

Run Batch Analysis
>> python3 batch_run.py  --fits-folder  swift/paper  --subfolder paper

This will execute multiple analysis runs based on the batch logic defined in the script.


5. Customization

Modify parameter.txt to adjust analysis settings.

Extend or replace functions in the util/ directory to add new statistical models, likelihoods, or plotting styles.

Use batch_run.py to automate large-scale or repetitive analyses.


6. Contact

Kally Wen, Lynbrook High School, kallywen@gmail.com
