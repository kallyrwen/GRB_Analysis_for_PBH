############################################################
#
#   Python script to process FITS file and simulated photon
#   file, and generate plots and analysis output files.
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
import sys
from datetime import datetime

from numpy.linalg import det
from utils.filelib import *
from utils.plotlib import *
from utils.likelihood import *
from utils.statslib import *


# ----------------------------------------------------------
# Helper function to parse command line arguments
# ----------------------------------------------------------
def getArguments():
    parser = argparse.ArgumentParser(description="Process input parameters.")

    # Optional parameter file
    parser.add_argument(
        "--param-file",
        type=str,
        default=None,
        help="Optional path to a parameter file",
    )

    # Regular CLI arguments
    parser.add_argument("--fits-file", type=str, help="Path to detected FITS file")
    parser.add_argument(
        "--source",
        type=str,
        default="swift",
        help="Source of the FIT File (swift or fermi)",
    )
    parser.add_argument(
        "--sim-file", type=str, help="Path to simulated photon input file"
    )
    parser.add_argument(
        "--dts-file", type=str, help="Path to DTS file for simulated photon timestamps"
    )
    parser.add_argument(
        "--area-csv",
        type=str,
        help="CSV file for effective area (for simulated photons only)",
    )
    parser.add_argument(
        "--bin-width", type=float, default=1.0, help="Time bin width (s)"
    )
    parser.add_argument(
        "--energy-bins",
        type=str,
        default="5,10,100,1000,10000,100000",
        help="Comma-separated energy bin edges in keV",
    )
    parser.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="Timestamp of starting point for processing the FIT file",
    )
    parser.add_argument(
        "--end-time",
        type=float,
        default=None,
        help="Timestamp of ending point for processing the FIT file (s)",
    )
    parser.add_argument(
        "--remain-time",
        type=float,
        default=None,
        help="Remain time for the end of the simulation (s)",
    )
    parser.add_argument(
        "--pre-bg-start",
        type=float,
        default=None,
        help="Pre-light-curve background start time (s)",
    )
    parser.add_argument(
        "--pre-bg-end",
        type=float,
        default=None,
        help="Pre-light-curve background end time (s)",
    )
    parser.add_argument(
        "--post-bg-start",
        type=float,
        default=None,
        help="Post-light-curve background start time (s)",
    )
    parser.add_argument(
        "--post-bg-end",
        type=float,
        default=None,
        help="Post-light-curve background end time (s)",
    )
    parser.add_argument(
        "--det-output",
        type=str,
        default="detected_counts.txt",
        help="Detected counts output file",
    )
    parser.add_argument(
        "--sim-output",
        type=str,
        default="simulated_counts.txt",
        help="Simulated counts output file",
    )
    parser.add_argument(
        "--det-plot",
        type=str,
        default="detected_lightcurve.png",
        help="Detected light curve plot",
    )
    parser.add_argument(
        "--sim-plot",
        type=str,
        default="simulated_lightcurve.png",
        help="Simulated light curve plot",
    )
    parser.add_argument(
        "--plot-all",
        type=str,
        default="plot_all_lightcurve.png",
        help="Plot all observed and simulated lightcurves",
    )
    parser.add_argument(
        "--scale",
        type=str,
        default="linear",
        help="Plot scale: linear, logx, logy, loglog",
    )
    parser.add_argument(
        "--chart-style", type=str, default="plot", help="Plot style: plot or step"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="energy_analysis_output.txt",
        help="Model comparison statistics output file",
    )
    parser.add_argument(
        "--folder",
        type=str,
        default="",
        help="Subfolder to save output files in",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="summary.txt",
        help="Path of the summary file",
    )
    parser.add_argument("--show-rise", type=str, default=None, help="Plot rise curve")

    # First, parse param-file arguments if provided
    param_args = []
    if "--param-file" in sys.argv:
        idx = sys.argv.index("--param-file")
        if idx + 1 < len(sys.argv):
            param_file = sys.argv[idx + 1]
            if os.path.exists(param_file):
                with open(param_file, "r") as f:
                    for line in f:
                        line = line.strip()
                        if not line or line.startswith("#"):
                            continue
                        param_args.extend(line.split())
            else:
                print(f"⚠️ Parameter file not found: {param_file}. Ignoring it.")

    # Command-line args come second so they override param-file values
    combined_args = param_args + sys.argv[1:]
    return parser.parse_args(combined_args)


# ----------------------------------------------------------
# Main function
# ----------------------------------------------------------
def main():

    # Parse auguments
    args = getArguments()

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.fits_file:
        fits_base = os.path.splitext(os.path.basename(args.fits_file))[0]
    else:
        fits_base = "no_fits"

    # Remove common suffixes
    for suffix in ["_all", "_short", "_long", "_preslew", "_postslew"]:
        if fits_base.endswith(suffix):
            fits_base = fits_base[: -len(suffix)]
            break  # stop after removing the first matching suffix
    folder_name = f"output/{args.folder}/{fits_base}-{timestamp}"
    os.makedirs(folder_name, exist_ok=True)

    # Construct full output path
    args.det_output = os.path.join(folder_name, args.det_output)
    args.sim_output = os.path.join(folder_name, args.sim_output)
    args.det_plot = os.path.join(folder_name, args.det_plot)
    args.sim_plot = os.path.join(folder_name, args.sim_plot)
    args.output = os.path.join(folder_name, args.output)
    args.plot_all = os.path.join(folder_name, args.plot_all)

    with open(args.output, "w") as f:
        f.write("========== Command Line ==========\n")
        f.write(" ".join(sys.argv) + "\n\n")

        f.write("========== Program Arguments ==========\n")
        for key, value in vars(args).items():
            f.write(f"{key}: {value}\n")
        f.write("\n")  # blank line for readability

    print(f"✅ Saved command line and arguments to {args.output}")

    energy_bins = [float(e) for e in args.energy_bins.split(",")]

    # Process FITS (detected) file
    if args.fits_file:
        if args.source == "swift":
            det_counts, det_times, det_labels = read_swift_file(
                args.fits_file,
                args.bin_width,
                energy_bins,
                args.start_time,
                args.end_time,
            )
        else:
            det_counts, det_times, det_labels = read_fermi_file_list(
                args.fits_file, args.bin_width, energy_bins
            )
        det_bg, det_dev = genBackgroundCounts(
            det_counts,
            det_times,
            det_labels,
            args.pre_bg_start,
            args.pre_bg_end,
            args.post_bg_start,
            args.post_bg_end,
        )
        plot_detected_lightcurve(
            det_counts,
            det_times,
            det_labels,
            det_bg,
            det_dev,
            args.det_plot,
            args.chart_style,
            args.scale,
        )
        write_detected_output(args.det_output, det_counts, det_times, det_labels)
    else:
        det_counts = det_times = det_labels = det_bg = det_dev = None

    # Process simulated photon file
    if args.sim_file:
        new_sim_file = os.path.join(
            folder_name,
            os.path.basename(args.sim_file).replace(".txt", "_processed.txt"),
        )
        update_sim_file_timestamp(args.sim_file, args.dts_file, new_sim_file)
        sim_counts, sim_times = read_sim_file(
            new_sim_file, energy_bins, args.remain_time, args.area_csv
        )
        plot_simulated_lightcurve(
            sim_counts, sim_times, energy_bins, args.sim_plot, args.scale
        )
        write_sim_output(args.sim_output, sim_counts, sim_times, energy_bins)
    else:
        sim_counts = sim_times = sim_labels = None

    # Process per energy bin: align, compute likelihood, fit FRED, compare models
    if sim_counts is not None and det_counts is not None:
        with open(args.output, "a") as f:
            f.write(
                "========== Compute Likelihood and FRED Fit Per Energy Bin ==========\n"
            )

        num_bins = len(energy_bins) - 1
        for i in range(num_bins):
            energy_range = f"{energy_bins[i]}-{energy_bins[i+1]} KeV"
            print(f"\n🔹 Processing energy bin {energy_range}...")

            # Extract one bin's data
            det_col = det_counts[:, i]  # det_counts is array
            sim_array = np.array(
                [sim_counts[k] for k in sorted(sim_counts.keys())]
            )  # shape: (num_rows, num_columns)
            sim_col = sim_array[:, i]  # i-th column
            if sim_col is None:
                print(
                    f"⚠️ Missing simulation data for energy bin {energy_range}, skipping..."
                )
                continue

            bg_col = det_bg[i] if det_bg is not None else None
            dev_col = det_dev[i] if det_dev is not None else None

            # Align simulation and detection
            sim_aligned, sim_time_bins, pre_peak_counts, post_peak_counts = (
                align_sim_data(
                    sim_col,
                    sim_times,
                    args.bin_width,
                    args.pre_bg_end,
                    args.post_bg_start,
                )
            )
            det_aligned, det_time_aligned, sim_aligned = align_det_data(
                det_col,
                det_times,
                pre_peak_counts,
                post_peak_counts,
                sim_aligned,
                sim_time_bins,
            )

            # Compute scale factor and likelihood
            scale_factor = compute_scalefactor(
                det_aligned, sim_aligned, bg_col, dev_col
            )
            likelihood, det_clipped, sim_clipped = compute_likelihood(
                det_aligned, sim_aligned, scale_factor, bg_col
            )
            lh2, det_clipped2, sim_clipped2 = compute_likelihood(
                det_clipped, det_clipped, 1.0, None
            )

            # Write results to output
            with open(args.output, "a") as f:
                f.write(f"\n--------------- Energy Bin {energy_range} --------------\n")
                f.write(f"Scale factor: {scale_factor:.6e}\n")
                f.write(f"Likelihood: {likelihood:.6f}\n")
                f.write(f"Minimal likelihood: {lh2:.6f}\n")
                f.write(f"Likelihood ratio: {likelihood / lh2}\n")

            # FRED Fit per bin
            det_for_fit = det_aligned - bg_col if bg_col is not None else det_aligned

            result = fit_fred_per_energy(
                det_for_fit,
                det_time_aligned,
                args.output,
                energy_range,
            )

            # Compare models for this bin
            if bg_col is not None:
                sim2 = np.clip(np.array(sim_clipped - bg_col), 0, None)
            else:
                sim2 = sim_clipped

            curves = compare_models(
                result,
                sim2,
                energy_range,
                args.output,
                args.summary,
                fits_base,
                dev_col,
            )

            # Plot all per energy bin
            bin_plot_file = args.plot_all.replace(
                ".png", f"_{energy_bins[i]}-{energy_bins[i+1]}.png"
            )
            plot_all_per_energy_bin(
                curves,
                energy_range,
                dev_col,
                bin_plot_file,
                args.chart_style,
                fits_base,
                args.show_rise,
                args.scale,
            )


if __name__ == "__main__":
    main()
