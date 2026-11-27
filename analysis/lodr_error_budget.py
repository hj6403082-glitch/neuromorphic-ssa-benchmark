import numpy as np
import matplotlib.pyplot as plt

# --- 1. Orbital & Sensor Constants ---
MU_EARTH = 3.986004418e14   # m^3 / s^2 [Standard]
R_EARTH = 6.371e6           # m [Standard]
IFOV_URAD = 31.0            # 6.4"/px = 31.0 urad/px [Measured from ISS_2_td]
FSM_WINDOW_URAD = 250.0     # ±250 urad fine-steering capture window [Assumed]

# --- 2. Unified 1/e^2 Gaussian Beam Definitions ---
LAMBDA_M = 1.06e-6          # Nd:YAG wavelength [Phipps et al. 2012]
D_APERTURE = 1.0            # 1 m class seeing-limited aperture [Assumed]
# 1/e^2 Gaussian fit to Airy disk: w_diff = 0.82 * lambda / D
W_DIFF_URAD = (0.82 * LAMBDA_M / D_APERTURE) * 1e6  # 0.87 urad
FWHM_TO_1E2 = 1.0 / np.sqrt(2.0 * np.log(2.0))      # 0.8493

# --- 3. Laser-Coupling Constants (Phipps et al., 2012, Adv. Space Res. 49 / arXiv:1110.3835) ---
PHI_OPT = 53.0              # kJ/m^2 (Eq. 3, tau=5 ns on Al, worst case) [Phipps 2012]
PHI_TH = 15.0               # kJ/m^2 (plasma ignition cutoff) [Assumed]
PHI_0 = 106.0               # kJ/m^2 (2x Phi_opt on-axis margin) [Assumed]
CM_UN_S_J = 75.0            # uN*s/J (representative Al baseline, Fig. 3) [Phipps 2012]
ETA_C = 0.30                # tumbling/shape coupling efficiency [Phipps 2012]
AREAL_MASS = 10.0           # kg/m^2 representative debris [Phipps 2012]
PULSES_PER_PASS = 1000      # ~11.2 Hz over ~90 s pass [Phipps 2012]

# --- 4. Measured Trackers (eval_mot_synth.py & baselines/eval_frame_mot.py) ---
# (Label, Measured RMSE_px, Latency lower bound ms, Color, Marker)
TRACKERS = [
    ("SNN (val, 5.5ms LB)",           2.27,  5.5, "#00d4ff", "o"),
    ("SNN (test, 5.5ms LB)",          3.99,  5.5, "#00ff88", "s"),
    ("200 fps (5.0ms LB, 8.5% P)",    5.53,  5.0, "#f59e0b", "^"),
    ("60 fps (16.7ms LB, 31.6% R)",   0.68, 16.7, "#fb7185", "D"),
    ("30 fps (33.3ms LB, 17.1% P)",   5.42, 33.3, "#ef4444", "X"),
]


def leo_zenith_rate(alt_km):
    h = alt_km * 1e3
    v = np.sqrt(MU_EARTH / (R_EARTH + h))
    omega_rad_s = v / h
    return v, omega_rad_s * 1e3, (omega_rad_s * 1e6) / IFOV_URAD


def beam_1e2_urad(seeing_arcsec):
    w_see_urad = (seeing_arcsec * 4.8481368) * FWHM_TO_1E2
    return float(np.hypot(W_DIFF_URAD, w_see_urad))


def pulse_energy_mj(w_urad, alt_km, phi0_kj_m2=PHI_0):
    w_m = (alt_km * 1e3) * (w_urad * 1e-6)
    e_j = 0.5 * np.pi * (w_m ** 2) * (phi0_kj_m2 * 1e3)
    return e_j / 1e6


def delivered_fluence(err_urad, w_urad, phi0=PHI_0):
    return phi0 * np.exp(-2.0 * (err_urad / w_urad) ** 2)


def delta_v_mm_s(phi_kj_m2):
    dv = (ETA_C * (CM_UN_S_J * 1e-6) * (phi_kj_m2 * 1e3) / AREAL_MASS) * 1e3
    return np.where(phi_kj_m2 >= PHI_TH, dv, 0.0)


def print_tables():
    print("=== 1. Coarse Acquisition Pointing Offset (31.0 urad/px, Measured RMSE + Latency LB) ===")
    print(f"{'Regime':<18} | {'Rate(mrad/s)':>12} | {'SNN-val':>9} | {'SNN-test':>9} | {'200 fps':>9} | {'60 fps*':>9} | {'30 fps':>9}")
    print("-" * 88)
    regimes = [("ISS (533km slant)", 13.86, 446.0)]
    for alt in (450, 800, 1000):
        _, om, px_s = leo_zenith_rate(alt)
        regimes.append((f"LEO {alt} km", om, px_s))
    for name, om, px_s in regimes:
        vals = [f"{np.hypot(r, px_s * (l * 1e-3)) * IFOV_URAD:6.1f} ur" for _, r, l, _, _ in TRACKERS]
        print(f"{name:<18} | {om:12.2f} | " + " | ".join(f"{v:>9}" for v in vals))
    print("*60 fps baseline achieves only 31.6% recall on test; 200 fps has 30.3 FP/pass & 156 IDSW.\n")

    print("=== 2. Unified 1/e^2 Beam Radius, Pulse Energy & Delta-v at 800 km LEO ===")
    dv_opt = float(delta_v_mm_s(PHI_OPT))
    dv_0 = float(delta_v_mm_s(PHI_0))
    print(f"Coupling [Phipps 2012]: Phi_opt={PHI_OPT:.0f} kJ/m^2 -> {dv_opt:.1f} mm/s/pulse ({dv_opt*PULSES_PER_PASS/1e3:.1f} m/s per 1000-pulse pass)")
    print(f"On-axis [Assumed 2x]:   Phi_0  ={PHI_0:.0f} kJ/m^2 -> {dv_0:.1f} mm/s/pulse ({dv_0*PULSES_PER_PASS/1e3:.1f} m/s per 1000-pulse pass)")
    print(f"{'Seeing (FWHM)':<15} | {'w_theta (1/e^2)':>15} | {'Spot r @ 800km':>14} | {'E_p @ 53 kJ/m^2':>15} | {'E_p @ 106 kJ/m^2':>16}")
    print("-" * 88)
    for s_arc in (0.5, 1.0, 2.0):
        w_u = beam_1e2_urad(s_arc)
        r_m = 800e3 * (w_u * 1e-6)
        e53 = pulse_energy_mj(w_u, 800, PHI_OPT)
        e106 = pulse_energy_mj(w_u, 800, PHI_0)
        print(f"{s_arc:4.1f}\" seeing    | {w_u:11.2f} urad | {r_m:12.2f} m | {e53:12.2f} MJ | {e106:13.2f} MJ")
    print("Reference [Phipps 2012]: 13 m AO mirror -> 0.155 m spot radius @ 1000 km -> 7.3 kJ/pulse.\n")


def make_figure():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.5), dpi=200)
    plt.rcParams.update({"font.size": 9.5})

    # Panel A: Coarse Acquisition Stage
    alts = np.array([450, 800, 1000])
    for label, rmse_px, lat_ms, col, mk in TRACKERS:
        errs = [np.hypot(rmse_px, leo_zenith_rate(a)[2] * (lat_ms * 1e-3)) * IFOV_URAD for a in alts]
        ax1.plot(alts, errs, marker=mk, lw=2.2, ms=7, color=col, label=label)

    w_1arc = beam_1e2_urad(1.0)
    ax1.axhline(FSM_WINDOW_URAD, color="#64748b", ls="--", lw=1.4, label="FSM capture window (±250 µrad) [Assumed]")
    ax1.axhline(w_1arc, color="#7c3aed", ls=":", lw=1.6, label=f"1m seeing-limited 1/e² beam ({w_1arc:.2f} µrad, 1\" FWHM)")
    ax1.set_yscale("log")
    ax1.set_xticks(alts)
    ax1.set_xlabel("Circular LEO Altitude (km)")
    ax1.set_ylabel("Uncompensated Pointing Offset (µrad, log)")
    ax1.set_title("A. Coarse Acquisition Stage (Measured RMSE + Latency LB)")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(fontsize=7.8, loc="lower left")

    # Panel B: Second-Stage Fine-Track Loop Design Curve (800 km)
    rmse_grid = np.linspace(0.0, 6.0, 200)     # [Assumed fine-track sweep]
    lat_grid = np.linspace(0.0, 1.0, 200)      # [Assumed fine-track sweep]
    R_G, L_G = np.meshgrid(rmse_grid, lat_grid)
    _, omega_800, _ = leo_zenith_rate(800)
    err_grid = np.hypot(R_G, omega_800 * L_G)
    phi_grid = delivered_fluence(err_grid, w_1arc, PHI_0)

    cf = ax2.contourf(L_G, R_G, phi_grid, levels=np.linspace(0, PHI_0, 22), cmap="viridis")
    fig.colorbar(cf, ax=ax2, pad=0.02, label="Delivered Fluence Φ (kJ/m²)")

    # Plot Phi_opt = 53 kJ/m^2 contours across 0.5", 1.0", 2.0" seeing
    for s_arc, col, ls in [(0.5, "#38bdf8", ":"), (1.0, "#facc15", "-"), (2.0, "#fb923c", "-.")]:
        w_s = beam_1e2_urad(s_arc)
        p_s = delivered_fluence(err_grid, w_s, PHI_0)
        ep = pulse_energy_mj(w_s, 800, PHI_0)
        cs = ax2.contour(L_G, R_G, p_s, levels=[PHI_OPT], colors=[col], linewidths=2.0, linestyles=ls)
        ax2.clabel(cs, fmt={PHI_OPT: f"{s_arc}\" ({ep:.1f} MJ)"}, fontsize=8, inline=True)

    c_th = ax2.contour(L_G, R_G, phi_grid, levels=[PHI_TH], colors=["#ef4444"], linewidths=1.8, linestyles="--")
    ax2.clabel(c_th, fmt={PHI_TH: "Cutoff 15 kJ/m² [Assumed]"}, fontsize=8, inline=True)

    ax2.set_xlabel("Fine-Track Residual Latency τ_res (ms) @ 800 km [Assumed sweep]")
    ax2.set_ylabel("Fine-Track Spatial Error σ_pos (µrad) [Assumed sweep]")
    ax2.set_title("B. Second-Stage Fine-Track Coupling (Φ_opt=53 kJ/m², 1/e² seeing sweep)")
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    out_path = "results/lodr_design_curve.png"
    plt.savefig(out_path, dpi=200)
    print(f"Saved updated LODR design curve to {out_path}")


if __name__ == "__main__":
    print_tables()
    make_figure()
