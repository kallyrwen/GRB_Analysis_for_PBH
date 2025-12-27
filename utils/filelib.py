############################################################
#
#   Python script to read input files and write output files.
#
#   Author: Kally Wen
#           Lynbrook High School
#           kallywen@gmail.com
#
#   Last modification: Dec 15, 2025
#
############################################################

from collections import defaultdict
from decimal import Decimal, getcontext

import numpy as np
from astropy.io import fits

getcontext().prec = 50  # High precision for timestamp processing


# ----------------------------------------------------------
# Generate Effective Area Mapping (for simulation only)
# ----------------------------------------------------------
def read_csv_to_dict(filename):
    result = {}
    with open(filename, "r") as f:
        next(f)  # skip header
        next(f)  # skip first data line if needed
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 2:
                continue
            try:
                key = float(parts[0].strip())  # energy keV
                value = float(parts[1].strip())  # area cm^2
                result[key] = value
            except Exception:
                continue
    return result


# ----------------------------------------------------------
# Get the effective area for a given energy
# ----------------------------------------------------------
def get_area_for_energy(energy, area_map):
    if not area_map:
        return 1.0
    keys = sorted(area_map.keys())
    if energy <= keys[0]:
        return area_map[keys[0]]
    if energy >= keys[-1]:
        return area_map[keys[-1]]
    for i in range(len(keys) - 1):
        e1, e2 = keys[i], keys[i + 1]
        if e1 <= energy <= e2:
            a1, a2 = area_map[e1], area_map[e2]
            return a1 + (a2 - a1) * (energy - e1) / (e2 - e1)
    return 1.0


# ----------------------------------------------------------
# Read SWIFT FIT file to arrays
# ----------------------------------------------------------
def read_swift_file(fits_file, bin_width, energy_bins_keV, start_time, end_time):

    with fits.open(fits_file) as hdul:
        hdul.info()
        data = hdul[1].data
        energy_array = np.array(data["ENERGY"])
        time_array = np.array(data["TIME"])

    # Energy bins
    ebin_edges = np.array(energy_bins_keV)
    energy_nbins = len(ebin_edges) - 1
    ebin_labels = [
        f"{ebin_edges[i]:.0f}-{ebin_edges[i+1]:.0f}" for i in range(energy_nbins)
    ]

    # Time bins
    t_min, t_max = time_array.min(), time_array.max()
    if start_time is not None:
        t_min = max(t_min, start_time)
    if end_time is not None:
        t_max = min(t_max, end_time)
    time_bins = np.arange(t_min, t_max + bin_width, bin_width)
    counts_matrix = np.zeros((len(time_bins) - 1, energy_nbins))

    # Fill counts (without area correction)
    for t, e in zip(time_array, energy_array):
        if t < t_min or t > t_max:
            continue
        time_idx = np.searchsorted(time_bins, t) - 1
        if time_idx < 0 or time_idx >= counts_matrix.shape[0]:
            continue
        for i in range(energy_nbins):
            if ebin_edges[i] <= e < ebin_edges[i + 1]:
                counts_matrix[time_idx, i] += 1
                break
    print(
        f"✅ Read FITS file {fits_file} {energy_nbins} to {len(time_bins)} time bins."
    )
    return (counts_matrix / bin_width), time_bins, ebin_labels


# ----------------------------------------------------------
# Read Fermi FIT file to arrays
# ----------------------------------------------------------
def read_fermi_file_list(file_list_path, bin_width, energy_bins_keV):

    # Read trigger time and FITS file paths
    offset_time = 5
    postpeak_time = 20
    fits_files = []

    with open(file_list_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            # Detect "OFFSET = value" line and "POSTPEAK = value" line
            if line.upper().startswith("OFFSET"):
                try:
                    offset_time = float(line.split("=", 1)[1].strip())
                except Exception:
                    raise ValueError(f"Cannot parse OFFSET line: '{line}'")
                continue
            if line.upper().startswith("POSTPEAK"):
                try:
                    postpeak_time = float(line.split("=", 1)[1].strip())
                except Exception:
                    raise ValueError(f"Cannot parse OFFSET line: '{line}'")
                continue

            # Otherwise treat the line as a FITS file path
            fits_files.append(line)

    # Condition: No FITS files
    if len(fits_files) == 0:
        print("⚠ No FITS file paths found in the input file. Returning None.")
        return None, None, None

    # Extract TRIGTIME from first FIT file
    first_fits = fits_files[0]
    print(f"📄 Reading trigger time from first FITS file: {first_fits}")

    trigger_time = None
    try:
        with fits.open(first_fits) as hdul:
            header = hdul[0].header
            if "TRIGTIME" in header:
                trigger_time = float(header["TRIGTIME"])
                print(f"📌 TRIGTIME found: {trigger_time}")
            else:
                print("ℹ No TRIGTIME found in FITS header. Using full time range.")
    except Exception as e:
        print(f"❌ Failed to read TRIGTIME from FITS header: {e}")
        print("ℹ Using full time range instead.")

    # Set start_time and end_time depending on trigger_time
    if trigger_time is None:
        print("ℹ No trigger time found. Using full FITS time ranges.")
        start_time = None
        end_time = None
    else:
        start_time = trigger_time - offset_time
        end_time = trigger_time + postpeak_time
        print(f"📌 Trigger time = {trigger_time}")
        print(f"⏱  Using ±5s window: {start_time} → {end_time}")

    # Energy bin setup
    ebin_edges = np.array(energy_bins_keV)
    energy_nbins = len(ebin_edges) - 1
    ebin_labels = [
        f"{ebin_edges[i]:.0f}-{ebin_edges[i+1]:.0f}" for i in range(energy_nbins)
    ]

    # Initialize combined collectors
    combined_counts = None
    final_time_bins = None

    # Process each FITS file
    for fits_file in fits_files:

        print(f"🔍 Reading {fits_file}")

        with fits.open(fits_file) as hdul:

            chan_table = hdul[1].data
            e_min = chan_table["E_MIN"]
            e_max = chan_table["E_MAX"]
            energy_map = (e_min + e_max) / 2.0

            data = hdul[2].data
            time_array = data["TIME"]
            pha_array = data["PHA"]
            energy_array = energy_map[pha_array]

        # Clip time to window only if start_time/end_time are defined
        if start_time is not None and end_time is not None:
            mask = (time_array >= start_time) & (time_array <= end_time)
        else:
            mask = np.ones_like(time_array, dtype=bool)

        time_array = time_array[mask]
        energy_array = energy_array[mask]

        if len(time_array) == 0:
            print(f"⚠ No events in selected window for {fits_file}")
            continue

        # Create time bins only once (first valid FITS file)
        if final_time_bins is None:
            local_tmin = time_array.min()
            local_tmax = time_array.max()
            if start_time is None:
                # Use the FITS file's natural time range
                final_tmin = local_tmin
            else:
                # Use overlap between trigger window and file's actual range
                final_tmin = max(local_tmin, start_time)

            if end_time is None:
                # Use the FITS file's time range
                final_tmax = local_tmax
            else:
                final_tmax = min(local_tmax, end_time)

            final_time_bins = np.arange(final_tmin, final_tmax + bin_width, bin_width)
            combined_counts = np.zeros((len(final_time_bins) - 1, energy_nbins))

        # Fill counts for this file
        for t, e in zip(time_array, energy_array):
            time_idx = np.searchsorted(final_time_bins, t) - 1
            if time_idx < 0 or time_idx >= combined_counts.shape[0]:
                continue

            for i in range(energy_nbins):
                if ebin_edges[i] <= e < ebin_edges[i + 1]:
                    combined_counts[time_idx, i] += 1
                    break

        print(f"   ✔ Processed {fits_file}")

    print("✅ All FITS files processed.")

    return (combined_counts / bin_width), final_time_bins, ebin_labels


# ----------------------------------------------------------
#  Update the timestamp of the Simulation file
# ----------------------------------------------------------
def update_sim_file_timestamp(sim_file, dts_file, output_file):
    # Read photon_primary_spectrum.txt
    with open(sim_file, "r") as f_photon:
        # Read and store the first two header lines
        header_lines = [f_photon.readline(), f_photon.readline()]
        # Read the rest of the data lines
        photon_lines = [line.strip() for line in f_photon if line.strip()]

    # Read dst.txt and extract dt values from the second column
    dt_values = []
    with open(dts_file, "r") as f_dts:
        # Skip first two lines (headers)
        for _ in range(2):
            next(f_dts)

        for line in f_dts:
            parts = line.strip().split()
            if len(parts) < 2:
                continue  # Skip incomplete lines
            try:
                dt = Decimal(parts[1])  # dt is the second column
                dt_values.append(dt)
            except Exception:
                continue  # Skip lines that can't be converted

    # Sanity check
    if len(dt_values) != len(photon_lines):
        raise ValueError(
            f"Mismatch: {len(dt_values)} dt values vs {len(photon_lines)} photon data lines."
        )

    # Compute remaining times
    remaining_times = []
    total_time = 0
    for dt in reversed(dt_values):
        remaining_times.append(total_time)
        total_time += dt
    remaining_times.reverse()

    # Write the new output file
    with open(output_file, "w") as f_out:
        # Write original headers
        f_out.writelines(header_lines)

        # Write modified data lines with new time column
        for rem_time, photon_line in zip(remaining_times, photon_lines):
            parts = photon_line.split()
            if not parts:
                continue
            # Format time using E-notation
            time_str = f"{rem_time:.15E}"
            rest_of_line = " ".join(parts[1:])
            f_out.write(f"{time_str} {rest_of_line}\n")

    print(
        f"✅ New simulation file (with timestamp as remain time) written to '{output_file}'"
    )


# -----------------------------
# Simulated photon processing
# -----------------------------
def read_sim_file(
    sim_file,
    energy_bin_edges,
    remain_time,
    area_csv=None,
):

    areas_map = read_csv_to_dict(area_csv)
    with open(sim_file, "r") as f:
        lines = f.readlines()

    # Split and ignore the first token (the "time/energy" header)
    parts = lines[1].strip().split()[1:]

    # Convert to float
    energy_values_GeV = np.array([float(x) for x in parts])
    energy_values_KeV = energy_values_GeV * 1e6  # Convert to KeV
    areas_for_energies = np.array(
        [get_area_for_energy(energy, areas_map) for energy in energy_values_GeV]
    )

    # Skip the first two header lines
    data_lines = lines[2:]
    times = []
    spectra = []

    for line in data_lines:
        parts = line.strip().split()
        if len(parts) < 2:
            continue
        try:
            time = float(Decimal(parts[0]))
            values = [float(x) for x in parts[1:]]
            times.append(time)
            spectra.append(values)
        except Exception:
            continue
    times = np.array(times)
    spectra = np.array(spectra)

    # Filter by remain_time (if provided)
    if remain_time is not None:
        mask = times <= remain_time
        times = times[mask]
        spectra = spectra[mask]

    # Set Up Time and Energy Bins
    max_time = max(times)
    start_time = remain_time if remain_time is not None else max_time
    num_energy_bins = len(energy_bin_edges) - 1

    # Aggregate to Energy Bins
    photon_counts = defaultdict(lambda: np.zeros(num_energy_bins))

    for i in range(len(times)):
        spectrum = spectra[i]

        for j, value in enumerate(spectrum):
            # Assume the j-th entry corresponds to some energy; we bin by index
            energy = energy_values_KeV[j]
            current_area = areas_for_energies[j]
            for k in range(num_energy_bins):
                if energy_bin_edges[k] <= energy < energy_bin_edges[k + 1]:
                    photon_counts[i][k] += (
                        value
                        * (energy_values_GeV[j + 1] - energy_values_GeV[j])
                        * current_area
                    )
                    break
    print(f"✅ Photon counts aggregated to {len(photon_counts)} time bins.")
    return photon_counts, times


# ----------------------------------------------------------
# Write detected photon counts to an output file
# ----------------------------------------------------------
def write_detected_output(
    output_file,
    counts_matrix,
    time_bins,
    bin_labels,
):
    with open(output_file, "w") as f:
        f.write("Time " + " ".join(bin_labels) + "  (keV)\n")
        for t_idx in range(counts_matrix.shape[0]):
            t_val = (time_bins[t_idx] + time_bins[t_idx + 1]) / 2
            f.write(f"{t_val:.6f} " + " ".join(map(str, counts_matrix[t_idx])) + "\n")
    print(f"✅ Detected photon counts written to {output_file}")


# ----------------------------------------------------------
# Write simulated photon counts to an output file
# ----------------------------------------------------------
def write_sim_output(output_file, photon_counts, times, energy_bin_edges):
    num_energy_bins = len(energy_bin_edges) - 1

    # Write Output (Time decreasing)
    with open(output_file, "w") as f_out:
        # Write header
        f_out.write("TimeBinStart ")
        f_out.write(
            " ".join(
                [
                    f"E{energy_bin_edges[i]}_to_{energy_bin_edges[i+1]}"
                    for i in range(num_energy_bins)
                ]
            )
            + "\n"
        )

        # Write binned data in decreasing time order
        for t_idx in sorted(photon_counts.keys()):
            time_start = times[t_idx]
            f_out.write(f"{time_start:.6e} ")
            f_out.write(
                " ".join(
                    [
                        f"{photon_counts[t_idx][e_idx]:.6e}"
                        for e_idx in range(num_energy_bins)
                    ]
                )
                + "\n"
            )
    print(f"✅ Photon counts written to '{output_file}' in decreasing remaining time.")
