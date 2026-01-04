############################################################
#
#   Python script to compute AIC | BIC of each model.
#
#   Author: Kally Wen
#           Lynbrook High School
#           kallyrwen@gmail.com
#
#   Last modification: Dec 15, 2025
#
############################################################

import numpy as np
from numpy.linalg import det
from scipy.optimize import curve_fit


# ---------------------------------------------------------
#   Norris FRED pulse model.
# ---------------------------------------------------------
def norris_fred_model(t, A, t_s, tau1, tau2):
    # Norris FRED pulse model.
    # I(t) = A * exp[-tau1/(t - t_s) - (t - t_s)/tau2], for t > t_s
    I = np.zeros_like(t)
    mask = t > t_s
    t_shifted = t[mask] - t_s
    I[mask] = A * np.exp(-tau1 / t_shifted - t_shifted / tau2)
    return I


# ---------------------------------------------------------
#   Exponential rise model.
# ---------------------------------------------------------
def exp_rise_model(t, A_r, t0, tau_r):
    # Exponential rise model for pre-peak emission:
    # I(t) = A_r * (exp(-tau_r / (t - t0)))
    # valid for t > t0
    I = np.zeros_like(t)
    mask = t > t0
    I[mask] = A_r * (np.exp(-tau_r / (t[mask] - t0)))
    return I


# ---------------------------------------------------------
#   Fit Norris FRED to each energy bin.
# ---------------------------------------------------------
def fit_fred_per_energy(det_counts, det_times, output_file, energy_bin_label):

    counts = np.clip(np.array(det_counts), 0, None)

    if len(det_times) == len(counts) + 1:
        time_mid = 0.5 * (det_times[:-1] + det_times[1:])
    else:
        time_mid = np.array(det_times)

    if np.all(counts == 0):
        print(f"⚠️ Energy bin {energy_bin_label} has all zero counts, skipping.")
        return None

    # Identify peak and split rising segment
    peak_index = np.argmax(counts)
    peak = counts[peak_index]

    t_rise = time_mid[: peak_index + 1]
    c_rise = counts[: peak_index + 1]
    t_post = time_mid[peak_index + 1 :]
    c_post = det_counts[peak_index + 1 :]

    # Fit exponential rise before peak
    A_r0 = peak
    t00 = t_rise.min()
    tau_r0 = (t_rise[-1] - t_rise[0]) / 2

    try:
        popt_rise, _ = curve_fit(
            exp_rise_model,
            t_rise,
            c_rise,
            p0=[A_r0, t00, tau_r0],
            bounds=([0, t00 - 1, 1e-4], [10 * peak, t_rise.max(), 100]),
            maxfev=10000,
        )
        A_r, t0_r, tau_r = popt_rise
        rise_curve = exp_rise_model(t_rise, A_r, t0_r, tau_r)
        print(f"🔺 Rise Fit: A_r={A_r:.3f}, t0={t0_r:.3f}, tau_r={tau_r:.3f}")
    except:
        print("⚠️ Exponential rise fit failed. Setting rise curve to zero.")
        rise_curve = np.zeros_like(time_mid)
        A_r, t0_r, tau_r = 0, 0, 0

    if len(c_post) > 0:
        const_level = np.mean(c_post)
        const_curve = np.full_like(t_post, const_level)
    else:
        const_level = 0
        const_curve = np.array([])

    combined_curve = np.concatenate([rise_curve, const_curve])

    # Fit Norris FRED (full pulse)
    A = peak
    t_s = time_mid[peak_index] - 0.01
    tau1 = 0.1
    tau2 = 1.0
    p0 = [A, t_s, tau1, tau2]

    lower_bounds = [0, time_mid.min(), 1e-3, 1e-3]
    upper_bounds = [10 * A, time_mid.max(), 100, 100]

    try:
        popt, _ = curve_fit(
            norris_fred_model,
            time_mid,
            counts,
            p0=p0,
            bounds=(lower_bounds, upper_bounds),
            maxfev=10000,
        )
        A, t_s, tau1, tau2 = popt
    except RuntimeError:
        print(f"⚠️ FRED fit did not converge for {energy_bin_label}")

    fred_curve = norris_fred_model(time_mid, A, t_s, tau1, tau2)
    deviation = np.std(counts - fred_curve)

    # Log results
    print(
        f"✅ FRED Fit {energy_bin_label}: A={A:.4f}, t_s={t_s:.4f}, "
        f"tau1={tau1:.4f}, tau2={tau2:.4f}, peak={peak:.4f}, deviation={deviation:.4f}"
    )

    with open(output_file, "a") as f:
        f.write(
            f"Energy bin {energy_bin_label}: FRED(A={A:.4f},t_s={t_s:.4f},tau1={tau1:.4f},"
            f"tau2={tau2:.4f}), Rise(A_r={A_r:.4f},t0={t0_r:.4f},tau_r={tau_r:.4f}), "
            f"Const={const_level:.4f}, peak={peak:.4f}, deviation={deviation:.4f}\n"
        )

    return {
        "A": A,
        "t_s": t_s,
        "tau1": tau1,
        "tau2": tau2,
        "A_r": A_r,
        "t0_r": t0_r,
        "tau_r": tau_r,
        "fred_curve": fred_curve,
        "rise_curve": rise_curve,
        "const_curve": const_curve,
        "const_level": const_level,
        "combined_curve": combined_curve,
        "time": time_mid,
        "counts": counts,
        "det_counts": det_counts,
        "peak": peak,
        "deviation": deviation,
    }


# ---------------------------------------------------------
#  Compute log-likelihood assuming Gaussian errors.
# ---------------------------------------------------------
def compute_log_likelihood(y_obs, y_model, sigma=None):
    y_obs = np.ravel(y_obs)
    y_model = np.ravel(y_model)

    residuals = y_obs - y_model

    if sigma is None:
        sigma2 = np.var(residuals)
    else:
        sigma2 = np.square(sigma)

    # logL = -0.5 * np.sum(residuals**2 / sigma2 + np.log(2 * np.pi * sigma2))
    logL = -0.5 * np.sum(residuals**2 / sigma2)

    return logL


# ---------------------------------------------------------
#  Compute Akaike Information Criterion (AIC).
# ---------------------------------------------------------
def compute_aic(y_obs, y_model, num_params, sigma=None):
    logL = compute_log_likelihood(y_obs, y_model, sigma)
    return 2 * num_params - 2 * logL


# ---------------------------------------------------------
#  Compute Bayesian Information Criterion (BIC).
# ---------------------------------------------------------
def compute_bic(y_obs, y_model, num_params, sigma=None):
    y_obs = np.ravel(y_obs)
    n = len(y_obs)
    logL = compute_log_likelihood(y_obs, y_model, sigma)
    return num_params * np.log(n) - 2 * logL


# ---------------------------------------------------------
#  Compute different models.
# ---------------------------------------------------------
def compare_models(
    result,
    sim_aligned,
    energy_label,
    output_file,
    summary_file,
    fits_base,
    deviation,
):
    num_params_fred = 4  # FRED has 4 fitted parameters
    num_params_blackhawk = 1  # Treated as 1 parameter model
    num_params_combined = 3  # Rise + constant

    with open(output_file, "a") as f:
        f.write("\n")
        f.write("Compute AIC and BIC:\n")

    if result is None:
        print(f"⚠️ Skipping energy bin {energy_label}: no fit available")
        return

    observed_counts = result["det_counts"]
    fred_model = result["fred_curve"]
    blackhawk_model = sim_aligned
    combined_model = result["combined_curve"]  # ✅ your target model
    const_level = float(result["const_level"])
    time = result["time"]

    # Built PBH with constant tail
    peak_idx = int(np.argmax(blackhawk_model))
    pbh_model = blackhawk_model.copy()
    post_const = 0
    if (peak_idx + 1) < len(pbh_model):
        c_post = observed_counts[peak_idx + 1 :]
        post_const = np.mean(c_post)
        pbh_model[peak_idx + 1 :] = post_const

    # Compute AIC/BIC

    aic_fred = compute_aic(observed_counts, fred_model, num_params_fred, deviation)
    bic_fred = compute_bic(observed_counts, fred_model, num_params_fred, deviation)

    aic_blackhawk = compute_aic(
        observed_counts, blackhawk_model, num_params_blackhawk, deviation
    )
    bic_blackhawk = compute_bic(
        observed_counts, blackhawk_model, num_params_blackhawk, deviation
    )

    aic_erca = compute_aic(
        observed_counts, combined_model, num_params_combined, deviation
    )
    bic_erca = compute_bic(
        observed_counts, combined_model, num_params_combined, deviation
    )
    aic_pbh = compute_aic(observed_counts, pbh_model, num_params_blackhawk, deviation)
    bic_pbh = compute_bic(observed_counts, pbh_model, num_params_blackhawk, deviation)

    # Determine best model
    aic_dict = {
        "FRED": aic_fred,
        "PBH": aic_blackhawk,
        "ERCA": aic_erca,
        "PBH2": aic_pbh,
    }
    bic_dict = {
        "FRED": bic_fred,
        "PBH": bic_blackhawk,
        "ERCA": bic_erca,
        "PBH2": bic_pbh,
    }

    best_full_aic = min(aic_dict, key=aic_dict.get)
    best_full_bic = min(bic_dict, key=bic_dict.get)

    # Write detailed output

    with open(output_file, "a") as f:
        f.write(f"   AIC FRED = {aic_fred:.2f}, BIC FRED = {bic_fred:.2f}\n")
        f.write(f"   AIC PBH = {aic_blackhawk:.2f}, BIC PBH = {bic_blackhawk:.2f}\n")
        f.write(f"   AIC ERCA = {aic_erca:.2f}, BIC ERCA = {bic_erca:.2f}\n")
        f.write(
            f"   AIC PBH_Const_Tail = {aic_pbh:.2f}, BIC PBH_Const_Tail = {bic_pbh:.2f}\n"
        )
        f.write(
            f"   Preferred model (FULL): {best_full_aic} (AIC), {best_full_bic} (BIC)\n\n"
        )

    # Write summary file

    with open(summary_file, "a") as f:
        f.write(f"{fits_base}  {energy_label}")
        f.write(
            f"  {result['A']:.2f}  {result['t_s']:.4f}  {result['tau1']:.4f}  {result['tau2']:.4f}"
        )
        f.write(
            f"  {result['A_r']:.2f}  {result['t0_r']:.4f}  {result['tau_r']:.4f}  {result['const_level']:.2f}"
        )
        f.write(f"  {post_const:.2f}")
        f.write(f"  {aic_fred:.2f}  {bic_fred:.2f}")
        f.write(f"  {aic_blackhawk:.2f}  {bic_blackhawk:.2f}")
        f.write(f" {aic_pbh:.2f}  {bic_pbh:.2f}")
        f.write(f"  {aic_erca:.2f}  {bic_erca:.2f}")
        f.write(f"  {best_full_aic}  {best_full_bic}\n")

    # Console summary
    print(
        f"📊 {energy_label} FULL  AIC → Best model: {best_full_aic} | BIC → Best model: {best_full_bic}"
    )

    # Store models for plotting
    curves = {
        "time": time,
        "observed": observed_counts,
        "fred": fred_model,
        "blackhawk": blackhawk_model,
        "pbh_plus_const": pbh_model,
        "erca": combined_model,
    }
    return curves
