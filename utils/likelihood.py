############################################################
#
#   Python script to compute the likelihood of
#   simulation lightcurve and observed lightcurve.
#
#   Author: Kally Wen
#           Lynbrook High School
#           kallyrwen@gmail.com
#
#   Last modification: Dec 15, 2025
#
############################################################

import math
from collections import defaultdict
from itertools import count

import numpy as np


# ----------------------------------------------------------
# Get the effective area for a given energy
# ----------------------------------------------------------
def genBackgroundCounts(
    counts_matrix,
    times,
    energy_bins,
    pre_bg_start,
    pre_bg_end,
    post_bg_start,
    post_bg_end,
):
    if pre_bg_start is None and post_bg_end is None:
        return None, None
    pre_bg_start = 0 if pre_bg_start is None else pre_bg_start
    pre_bg_end = 0 if pre_bg_end is None else pre_bg_end
    post_bg_start = 0 if post_bg_start is None else post_bg_start
    post_bg_end = 0 if post_bg_end is None else post_bg_end

    # Background subtraction
    energy_nbin = len(energy_bins)
    bg_counts = np.zeros(energy_nbin)
    bg_deviation = np.zeros(energy_nbin)
    times = 0.5 * (times[:-1] + times[1:])

    # Find timestamp of peak total photon counts
    total_counts = counts_matrix.sum(axis=1)
    peak_index = np.argmax(total_counts)
    peak_time = times[peak_index]

    # Convert timestamps to relative times (peak = 0)
    relative_times = times - peak_time

    pre_mask = (relative_times >= -pre_bg_start) & (relative_times < -pre_bg_end)
    post_mask = (relative_times >= post_bg_start) & (relative_times < post_bg_end)
    background_mask = pre_mask | post_mask

    if np.any(background_mask):
        bg_counts = counts_matrix[background_mask].mean(axis=0)
        bg_deviation = counts_matrix[background_mask].std(axis=0)
    else:
        print("⚠️ No background data; using zeros.")

    print(f"✅ bg_counts: {bg_counts}")
    print(f"✅ bg_deviation: {bg_deviation}")

    return bg_counts, bg_deviation


# ----------------------------------------------------------
# Align the size of simulated data
# ----------------------------------------------------------
def align_sim_data(counts_array, times, bin_width, start_time=None, end_time=None):
    # Convert to numpy arrays
    times = np.array(times)
    max_time = times.max()
    times = max_time - times

    # Find timestamp of peak total photon counts
    peak_index = np.argmax(counts_array)
    peak_time = times[peak_index]

    # Convert timestamps to relative times (peak = 0)
    relative_times = times - peak_time

    # Determine time range
    min_time = -start_time if start_time is not None else relative_times.min()
    max_time = end_time if end_time is not None else relative_times.max()

    # Define bin edges (regular time points)
    time_bin_edges = np.arange(min_time, max_time, bin_width)

    # Prepare output dictionary
    aligned_counts = []

    for t_bin in time_bin_edges:
        # If exact match exists, use it
        if t_bin in relative_times:
            idx = np.where(relative_times == t_bin)[0][0]
            counts = counts_array[idx]
        else:
            # Find nearest smaller and larger timestamps
            smaller_mask = relative_times < t_bin
            larger_mask = relative_times > t_bin

            if not np.any(smaller_mask) or not np.any(larger_mask):
                # Skip if out of range
                counts = np.zeros_like(counts_array[0])
            else:
                t1 = relative_times[smaller_mask].max()
                t2 = relative_times[larger_mask].min()

                idx1 = np.where(relative_times == t1)[0][0]
                idx2 = np.where(relative_times == t2)[0][0]

                c1 = counts_array[idx1]
                c2 = counts_array[idx2]

                # Linear interpolation
                w = (t_bin - t1) / (t2 - t1)
                counts = c1 + w * (c2 - c1)

        aligned_counts.append(counts)

    # Convert to NumPy array (time_bins × energy_bins)
    aligned_counts = np.array(
        [np.asarray(x).item() if np.ndim(x) == 0 else float(x) for x in aligned_counts]
    )

    # Count bins: before/at peak combined, and after peak
    num_pre_peak = np.sum(time_bin_edges <= 0)
    num_post_peak = np.sum(time_bin_edges > 0)

    print(f"✅ Aligned simulated photon counts data samples: {len(aligned_counts)}")
    print(f"✅ num_pre_peak: {num_pre_peak}")
    print(f"✅ num_post_peak: {num_post_peak}")
    return aligned_counts, time_bin_edges, num_pre_peak, num_post_peak


# ----------------------------------------------------------
# Align the size of detector data
# ----------------------------------------------------------
def align_det_data(
    det_data, det_times, num_pre_peak, num_post_peak, sim_data, sim_times
):
    # Convert to numpy arrays
    det_data = np.asarray(det_data)
    det_times = np.asarray(det_times)

    # Compute midpoints of time bins
    mid_times = (det_times[:-1] + det_times[1:]) / 2

    # Find peak
    peak_index = np.argmax(det_data)
    peak_time = mid_times[peak_index]

    # Convert timestamps to relative times (peak = 0)
    relative_times = mid_times - peak_time

    # Sort by time (if not already sorted)
    sort_idx = np.argsort(relative_times)
    relative_times = relative_times[sort_idx]
    det_data = det_data[sort_idx]

    # Find the index range that matches num_pre_peak and num_post_peak
    start_index = max(0, peak_index - (num_pre_peak - 1))
    end_index = min(len(det_data), peak_index + num_post_peak)

    aligned_counts = det_data[start_index : (end_index + 1)]
    aligned_times = relative_times[start_index : (end_index + 1)]
    if (peak_index - (num_pre_peak - 1)) < 0:
        print(
            f"⚠️ Peak index is less than num_pre_peak. Need to trim the start of the data."
        )
        sim_data = sim_data[num_pre_peak - peak_index - 1 :]
        sim_times = sim_times[num_pre_peak - peak_index - 1 :]
    if (peak_index + num_post_peak) > len(det_data):
        print(
            f"⚠️ Peak index is greater than num_post_peak. Need to trim the end of the data."
        )
        sim_data = sim_data[: -(peak_index + num_post_peak - len(det_data))]

    print(f"✅ Aligned detected photon counts data samples: {len(aligned_counts)}")

    return aligned_counts, aligned_times, sim_data


# ----------------------------------------------------------
# compute scale factor
# ----------------------------------------------------------
def compute_scalefactor(det, sim, background=None, deviation=None):
    if background is None:
        background = np.zeros_like(det)
    det = det - background
    if deviation is None:
        numerator = np.sum(det * sim)
        denominator = np.sum(sim**2)
    else:
        deviation = np.array(deviation)
        if deviation.ndim == 1 and deviation.shape[0] == det.shape[-1]:
            deviation = deviation[np.newaxis, :]
        numerator = np.sum((det * sim) / deviation**2)
        denominator = np.sum(sim**2 / deviation**2)

    scale_factor = numerator / denominator
    print(f"✅ Scale factor computed: {scale_factor:.6e}")
    return scale_factor


# ----------------------------------------------------------
# compute Poisson Log Likelihood
# ----------------------------------------------------------
def compute_likelihood(A, B, scale_factor, background=None):

    if background is None:
        background = np.zeros_like(A)
    B2 = B * scale_factor
    B3 = B2 + background
    A_clipped = np.clip(A, 1e-10, None)
    B_clipped = np.clip(B3, 1e-10, None)

    term1 = A_clipped * np.log(B_clipped)
    term2 = B_clipped
    term3 = np.vectorize(lambda x: math.lgamma(x + 1))(A_clipped)

    K = term1 - term2 - term3
    likelihood = np.sum(K)
    print(f"✅ Likelihood computed: {likelihood:.6f}")
    return likelihood, A_clipped, B_clipped


# ----------------------------------------------------------
# Trim the extra lines
# ----------------------------------------------------------
def trim_extra_lines(sim, sim_time, det, det_time):
    det = np.array(det)
    sim = np.array(sim)

    # Trim sim or det to same length
    if len(sim) > len(det):
        sim = sim[: len(det)]
        sim_time = sim_time[: len(det)]
        print(f"⚠️ Trimmed sim to match det length ({len(det)} elements).")
    elif len(det) > len(sim):
        det = det[: len(sim)]
        det_time = det_time[: len(sim)]
        print(f"⚠️ Trimmed det to match sim length ({len(sim)} elements).")

    return sim, sim_time, det, det_time
