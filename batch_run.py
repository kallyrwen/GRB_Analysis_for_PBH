############################################################
#
#   Batch script to process multiple FIT files and
#   save the results in a summary file.
#
#   Author: Kally Wen
#           Lynbrook High School
#           kallyrwen@gmail.com
#
#   Last modification: Dec 15, 2025
#
############################################################

import argparse
import os
import subprocess

# --- Parse arguments for this wrapper script ---
parser = argparse.ArgumentParser(
    description="Run energy_analysis.py on all FITS/EVT files in a folder recursively"
)
parser.add_argument(
    "--fits-folder",
    type=str,
    help="Path to the folder containing FITS/EVT files",
)
parser.add_argument(
    "--subfolder",
    type=str,
    help="Subfolder to save output files",
)


args = parser.parse_args()

fits_folder = args.fits_folder
param_file = "parameter.txt"
script = "energy_analysis.py"
summary_file = "summary.txt"


# --- Recursively iterate over all files ---
for root, dirs, files in os.walk(fits_folder):
    # Determine the subfolder name relative to the root folder

    with open(summary_file, "a") as f:  # "a" appends results to the file
        f.write(f"\n\n")
        f.write(f"GRB  Energy_Bin")
        f.write(f"  A  t_s  tau1  tau2  A_r  t0_r  tau_r  const_level")
        f.write(f"  PBH_const")
        f.write(
            f"  AIC_Fred  BIC_Fred  AIC_PBH  BIC_PBH   AIC_PBH_Const_Tail  BIC_PBH_Const_Tail  AIC_ERCA  BIC_ERCA"
        )
        f.write(f"  AIC_Preferred  BIC_Preferred\n")

    if args.subfolder:
        subfolder_name = args.subfolder
    else:
        subfolder_name = os.path.relpath(root, fits_folder)

    print(f"subfolder_name = {subfolder_name}")
    print(f"root = {root}")
    print(f"dirs = {dirs}")
    print(f"fits_folder = {fits_folder}")

    for file in files:
        if file.endswith(".evt") or file.endswith(".fits"):
            fits_path = os.path.join(root, file)
            print(f"Processing {fits_path} in folder '{subfolder_name}' ...")
            # Call energy_analysis.py with --folder argument
            subprocess.run(
                [
                    "python3",
                    script,
                    "--param-file",
                    param_file,
                    "--fits-file",
                    fits_path,
                    "--folder",
                    subfolder_name,
                    "--summary",
                    summary_file,
                ]
            )
