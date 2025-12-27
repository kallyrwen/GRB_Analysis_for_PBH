############################################################
#
#   Python script to plot the lightcurves.
#
#   Author: Kally Wen
#           Lynbrook High School
#           kallywen@gmail.com
#
#   Last modification: Dec 15, 2025
#
############################################################

import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d, make_interp_spline
from scipy.signal import savgol_filter


# ----------------------------------------------------------
#  Plot detected lightcurve
# ----------------------------------------------------------
def plot_detected_lightcurve(
    counts_matrix,
    time_bins,
    energy_bin_labels,
    bg_counts,
    bg_deviation,
    plot_file,
    chart_style="plot",
    scale="linear",
):
    times_plot = (time_bins[:-1] + time_bins[1:]) / 2
    energy_nbins = len(energy_bin_labels)
    if bg_counts is None:
        bg_counts = np.zeros(energy_nbins)
    if bg_deviation is None:
        bg_deviation = np.zeros(energy_nbins)

    new_counts_matrix = counts_matrix[:-1]
    times_plot = times_plot[:-1]

    for i in range(energy_nbins):
        ydata = new_counts_matrix[:, i] - bg_counts[i]
        if chart_style == "step":
            plt.step(
                times_plot,
                ydata,
                where="mid",
                label=f"{energy_bin_labels[i]} KeV",
                linewidth=1.5,
            )
            if np.any(bg_deviation):
                plt.errorbar(
                    times_plot,
                    ydata,
                    yerr=bg_deviation[i],
                    fmt="none",
                    ecolor="gray",
                    elinewidth=1,
                    capsize=2,
                    alpha=0.6,
                )
        else:
            plt.plot(
                times_plot,
                ydata,
                label=f"{energy_bin_labels[i]} KeV",
                marker="o",
                linestyle="-",
                alpha=0.7,
            )

    plt.xlabel("Time (s)")
    plt.ylabel("Counts (background subtracted)")
    plt.title("Detected Photon Light Curve")
    plt.legend()
    plt.grid(True)
    if scale in ["logx", "logy", "loglog"]:
        if scale == "logx":
            plt.xscale("log")
        elif scale == "logy":
            plt.yscale("log")
        else:
            plt.xscale("log")
            plt.yscale("log")
    plt.tight_layout()
    plt.savefig(plot_file, dpi=300)
    plt.close()
    print(f"✅ Detected light curve saved to {plot_file}")


# ----------------------------------------------------------
# Plot simulated lightcurve
# ----------------------------------------------------------
def plot_simulated_lightcurve(
    photon_counts,
    times,
    energy_bin_edges,
    plot_file,
    scale="linear",
):
    num_energy_bins = len(energy_bin_edges) - 1
    valid_t_indices = sorted(photon_counts.keys())
    time_bin_starts = np.array([times[t_idx] for t_idx in valid_t_indices])

    # Build flux matrix (time bins × energy bins)
    flux_matrix = np.array([photon_counts[t_idx] for t_idx in valid_t_indices])

    # Convert flux → arbitrary units by normalizing each energy bin
    flux_matrix_arb = flux_matrix / np.max(flux_matrix, axis=0)

    # 4-color palette with no yellow (cividis)
    colors = [
        "#1f77b4",  # blue
        "#2ca02c",  # green
        "#d62728",  # red
        "#9467bd",  # purple
    ]
    # Plot each energy bin curve
    for e_idx in range(num_energy_bins):
        flux = flux_matrix_arb[:, e_idx]

        plt.plot(
            time_bin_starts,
            flux,
            label=f"{energy_bin_edges[e_idx]}-{energy_bin_edges[e_idx+1]} KeV",
            color=colors[e_idx],
            linewidth=1.5,
            alpha=0.8,
        )

    # Set scale for axes
    if scale == "logx":
        plt.xscale("log")
    elif scale == "logy":
        plt.yscale("log")
    elif scale == "loglog":
        plt.xscale("log")
        plt.yscale("log")
        # Optional default (adjust/remove as needed)
        # plt.ylim(1e-3, 1)

    # X-axis direction
    if scale in ["loglog", "logx"]:
        plt.xlim(100, 0.1)  # decreasing log scale
    else:
        plt.gca().invert_xaxis()

    # Labels and styling
    plt.xlabel("Remaining Time (s)")
    plt.ylabel("arb. units")
    plt.title("Simulated Light Curve by Energy Bins")
    plt.grid(True)
    plt.legend(title="Energy Range")

    plt.tight_layout()
    plt.savefig(plot_file)
    plt.close()

    print(f"✅ Plot simulated light curve saved to '{plot_file}'")


# -----------------------------------------------------------------------
#  Plot observed counts and FRED fit curves for each energy bin.
# -----------------------------------------------------------------------
def plot_all_per_energy_bin(
    result,
    energy_label,
    dev,
    plot_file,
    chart_style,
    fits_base,
    show_rise,
    scale="linear",
):

    if result is None:
        return

    time = result["time"]
    yObserved = result["observed"]
    yErca = result["erca"]
    yBlackhawk = result["blackhawk"]
    yFred = result["fred"]
    yPHB = result["pbh_plus_const"]

    # Plot detected counts
    if chart_style == "step":
        plt.step(
            time,
            yObserved,
            where="mid",
            label=f"Detected GRB Counts",
        )
        if dev is not None:
            plt.errorbar(
                time,
                yObserved,
                yerr=dev,
                fmt="none",
                ecolor="black",
                elinewidth=1,
                capsize=2,
                alpha=0.6,
            )
    else:
        plt.plot(
            time,
            yObserved,
            "o",
            label=f"Detected GRB Counts",
        )

    # Plot FRED, PBH, ERCA, PBH_Const_Tail curves
    plt.plot(
        time,
        yFred,
        linestyle="--",
        alpha=0.7,
        label="FRED Model",
    )

    plt.plot(
        time,
        yBlackhawk,
        linestyle="--",
        alpha=0.7,
        label="PBH Model",
    )

    if show_rise == "Y":
        plt.plot(
            time,
            yErca,
            linestyle=":",
            linewidth=2,
            alpha=0.9,
            label="ERCA Model",
        )
        plt.plot(
            time,
            yPHB,
            linestyle="--",
            alpha=0.7,
            label="PBHCA Model",
        )

    # Axis scaling
    if scale in ["logx", "logy", "loglog"]:
        if scale == "logx":
            plt.xscale("log")
        elif scale == "logy":
            plt.yscale("log")
        else:
            plt.xscale("log")
            plt.yscale("log")

    # Labels and formatting
    plt.xlabel("Time (s)")
    plt.ylabel("Photon Counts / s")
    plt.title(f"{fits_base} Energy Bin {energy_label}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save figure
    base, ext = os.path.splitext(plot_file)
    bin_plot_file = f"{base}_{energy_label}{ext}"

    plt.savefig(bin_plot_file, dpi=300)
    plt.close()
    print(f"✅ Saved plot for energy bin {energy_label} to {bin_plot_file}")
