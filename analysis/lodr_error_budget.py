import os
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Physical & Orbital Constants ---
MU_EARTH = 3.986004418e14   # m^3 / s^2
R_EARTH = 6.371e6           # m
IFOV_URAD = 31.0            # 6.4"/px = 31.0 urad/px (consistent with ISS 446 px/s)

# --- 2. Phipps et al. (2012), Adv. Space Res. 49, 1283-1300 ---
LAMBDA_M = 1.06e-6          # Nd:YAG wavelength (m)
D_APERTURE = 1.0            # 1 m class ground aperture (m)
SEEING_URAD = 4.85          # 1.0 arcsec atmospheric seeing (urad)
THETA_DIFF_URAD = (1.22 * LAMBDA_M / D_APERTURE) * 1e6  # ~1.29 urad
W_BEAM_URAD = np.hypot(THETA_DIFF_URAD, SEEING_URAD)    # ~5.02 urad effective beam radius

PHI_OPT = 53.0              # kJ/m^2 (Eq. 3, tau = 5 ns on Al)
PHI_TH = 15.0               # kJ/m^2 (plasma ignition cutoff)
PHI_0 = 106.0               # kJ/m^2 (2x Phi_opt on-axis design margin)
CM_UN_S_J = 75.0            # uN*s/J (representative Al baseline)
ETA_C = 0.30                # tumbling/shape coupling efficiency
AREAL_MASS = 10.0           # kg/m^2

# --- 3. Tracker Baselines (latency = lower bound: bin/frame + compute, excl. host I/O) ---
TRACKERS = [
    ("SNN (val, 5.5ms LB)",    2.27,  5.5, "#00d4ff", "o"),
    ("SNN (test, 5.5ms LB)",   3.99,  5.5, "#00ff88", "s"),
    ("200 fps (5.0ms LB)",     1.00,  5.0, "#f59e0b", "^"),
    ("60 fps (16.7ms LB)",     1.00, 16.7, "#fb7185", "D"),
    ("30 fps (33.3ms LB)",     1.00, 33.3, "#ef4444", "X"),
]


def leo_zenith_rate(alt_km):
    h = alt_km * 1e3
    v = np.sqrt(MU_EARTH / (R_EARTH + h))
    omega_rad_s = v / h
    return v, omega_rad_s * 1e3, (omega_rad_s * 1e6) / IFOV_URAD  # m/s, mrad/s, px/s


def delivered_fluence(err_urad, w_urad=W_BEAM_URAD, phi0=PHI_0):
    return phi0 * np.exp(-2.0 * (err_urad / w_urad) ** 2)


def delta_v_mm_s(phi_kj_m2):
    dv = (ETA_C * (CM_UN_S_J * 1e-6) * (phi_kj_m2 * 1e3) / AREAL_MASS) * 1e3
    return np.where(phi_kj_m2 >= PHI_TH, dv, 0.0)


def print_budget_table():
    print(f"Sensor Scale: {IFOV_URAD:.1f} urad/px (6.4\"/px) | Beam Radius w_theta: {W_BEAM_URAD:.2f} urad (diff={THETA_DIFF_URAD:.2f}, seeing={SEEING_URAD:.2f})")
    print("=" * 98)
    print(f"{'Regime':<18} | {'Rate (mrad/s)':>13} | {'px/s':>6} | {'SNN-val':>9} | {'SNN-test':>9} | {'200 fps':>9} | {'60 fps':>9} | {'30 fps':>9}")
    print("-" * 98)
    regimes = [("ISS (533km slant)", 13.86, 446.0)]
    for alt in (450, 800, 1000):
        _, om, px_s = leo_zenith_rate(alt)
        regimes.append((f"LEO {alt} km", om, px_s))
    for name, om, px_s in regimes:
        vals = []
        for _, rmse_px, lat_ms, _, _ in TRACKERS:
            tot_px = np.hypot(rmse_px, px_s * (lat_ms * 1e-3))
            vals.append(f"{tot_px * IFOV_URAD:6.1f} ur")
        print(f"{name:<18} | {om:13.2f} | {px_s:6.1f} | " + " | ".join(f"{v:>9}" for v in vals))
    print("=" * 98)


def make_figure():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.4), dpi=200)
    plt.rcParams.update({"font.size": 10})

    # --- Panel A: Coarse Acquisition / Handoff Budget ---
    alts = np.array([450, 800, 1000])
    for label, rmse_px, lat_ms, col, mk in TRACKERS:
        errs = []
        for alt in alts:
            _, _, px_s = leo_zenith_rate(alt)
            errs.append(np.hypot(rmse_px, px_s * (lat_ms * 1e-3)) * IFOV_URAD)
        ax1.plot(alts, errs, marker=mk, lw=2.2, ms=7, color=col, label=label)

    ax1.axhline(250.0, color="#64748b", ls="--", lw=1.4, label="FSM capture window (±250 µrad)")
    ax1.axhline(W_BEAM_URAD, color="#7c3aed", ls=":", lw=1.6, label=f"1m seeing-limited beam ({W_BEAM_URAD:.1f} µrad)")
    ax1.set_yscale("log")
    ax1.set_xticks(alts)
    ax1.set_xlabel("Circular LEO Altitude (km)")
    ax1.set_ylabel("Uncompensated Pointing Offset (µrad, log)")
    ax1.set_title("A. Coarse Acquisition & FSM Handoff (31.0 µrad/px)")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(fontsize=8, loc="lower left")

    # --- Panel B: Fine-Track Mitigation Coupling Design Curve (800 km LEO) ---
    # Residual error after coarse handoff / Kalman lead-ahead (fine-track regime)
    rmse_grid = np.linspace(0.0, 6.0, 200)     # residual spatial RMSE (urad)
    lat_grid = np.linspace(0.0, 1.0, 200)      # residual uncompensated latency (ms)
    R_G, L_G = np.meshgrid(rmse_grid, lat_grid)
    _, omega_800_mrad, _ = leo_zenith_rate(800)  # 9.32 mrad/s = 9.32 urad/ms
    err_grid = np.hypot(R_G, omega_800_mrad * L_G)
    phi_grid = delivered_fluence(err_grid)

    cf = ax2.contourf(L_G, R_G, phi_grid, levels=np.linspace(0, PHI_0, 22), cmap="viridis")
    cbar = fig.colorbar(cf, ax=ax2, pad=0.02)
    cbar.set_label("Delivered Fluence Φ (kJ/m²)")

    c_opt = ax2.contour(L_G, R_G, phi_grid, levels=[PHI_OPT], colors=["#facc15"], linewidths=2.2, linestyles="-")
    c_th = ax2.contour(L_G, R_G, phi_grid, levels=[PHI_TH], colors=["#ef4444"], linewidths=2.2, linestyles="--")
    ax2.clabel(c_opt, fmt={PHI_OPT: "Φ_opt = 53 kJ/m²"}, fontsize=8.5, inline=True)
    ax2.clabel(c_th, fmt={PHI_TH: "Cutoff Φ_th = 15 kJ/m²"}, fontsize=8.5, inline=True)

    ax2.set_xlabel("Uncompensated / Residual Latency τ_res (ms) @ 800 km")
    ax2.set_ylabel("Residual Spatial Pointing Error σ_pos (µrad)")
    ax2.set_title(f"B. Fine-Track LODR Coupling Design Curve (w_θ = {W_BEAM_URAD:.1f} µrad)")
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    out_path = "results/lodr_design_curve.png"
    plt.savefig(out_path, dpi=200)
    print(f"Saved LODR design curve to {out_path}")


if __name__ == "__main__":
    print_budget_table()
    make_figure()
