import numpy as np
import matplotlib.pyplot as plt

# --- 1. Orbital & Sensor Constants ---
MU_EARTH = 3.986004418e14   # m^3 / s^2 [Standard]
R_EARTH = 6.371e6           # m [Standard]
IFOV_URAD = 31.0            # 6.4"/px [Assumed: typical SSA optical setup]
FSM_WINDOW_URAD = 250.0     # ±250 urad fine-steering capture window [Assumed]

# --- 2. Phipps et al. (2012, arXiv:1110.3835) Reference AO Design ---
# Reference: 13 m AO mirror -> 0.155 m spot radius @ 1000 km -> w_theta = 0.155 urad, E_p = 7.3 kJ
W_REF_URAD = 0.155          # 1/e^2 beam radius at 1000 km [Phipps 2012, Sec. 4 / Case a]
PHI_OPT = 53.0              # kJ/m^2 (5 ns pulse on Al, worst case) [Phipps 2012, Eq. 3]
PHI_TH = 15.0               # kJ/m^2 (plasma ignition cutoff) [Assumed]
PHI_0 = PHI_OPT             # kJ/m^2 (1x Phi_opt on-axis) [Phipps 2012, Eq. 3]
CM_UN_S_J = 75.0            # uN*s/J (representative Al coupling) [Phipps 2012 — verify Fig. 3]
ETA_C = 0.30                # tumbling/shape efficiency [Phipps 2012 — verify Sec. 3]
AREAL_MASS = 10.0           # kg/m^2 representative debris [Assumed / Phipps 2012 baseline]
PULSES_PER_PASS = 1000      # pulses/pass (~11.2 Hz over ~90 s) [Assumed / Phipps 2012 baseline]

# --- 3. Validation-Tuned, Test-Locked Trackers ---
# (Label, Test RMSE_px, Latency LB ms, R@40%, P@40%, Color, Marker)
TRACKERS = [
    ("SNN v2 (val, 5.5ms LB)",          7.44,  5.5, 60.0, 80.0, "#00d4ff", "o"),
    ("SNN v2 (test60, 5.5ms LB)",       6.50,  5.5, 52.4, 51.0, "#00ff88", "s"),
    ("200 fps v2 (test60, 5.0ms LB)",   7.12,  5.0, 20.6, 47.0, "#f59e0b", "^"),
    ("60 fps v2 (test60, 16.7ms LB)",   6.68, 16.7, 54.0, 24.8, "#fb7185", "D"),
    ("30 fps v2 (test60, 33.3ms LB)",   6.57, 33.3, 42.3, 36.2, "#ef4444", "X"),
]


def leo_zenith_rate(alt_km):
    h = alt_km * 1e3
    v = np.sqrt(MU_EARTH / (R_EARTH + h))
    omega_rad_s = v / h
    return v, omega_rad_s * 1e3, (omega_rad_s * 1e6) / IFOV_URAD


def delivered_fluence(err_urad, w_urad=W_REF_URAD, phi0=PHI_0):
    return phi0 * np.exp(-2.0 * (err_urad / w_urad) ** 2)


def expected_dv_per_pass_m_s(err_urad, w_urad=W_REF_URAD, phi0=PHI_0):
    """Expected Delta-v (m/s) over PULSES_PER_PASS accounting for off-axis fluence drop-off."""
    phi = delivered_fluence(err_urad, w_urad, phi0)
    dv_pulse_m_s = (ETA_C * (CM_UN_S_J * 1e-6) * (phi * 1e3)) / AREAL_MASS
    return np.where(phi >= PHI_TH, dv_pulse_m_s * PULSES_PER_PASS, 0.0)


def print_tables():
    print("=== 1. Coarse Acquisition & Handoff Offset (31.0 urad/px [Assumed], Val-Tuned Proxies) ===")
    print(f"{'Tracker':<26} | {'R@40':>5} | {'P@40':>5} | {'RMSE':>6} | {'ISS 533':>8} | {'450 km':>8} | {'800 km':>8} | {'1000 km':>8}")
    print("-" * 96)
    regimes = [("ISS", 446.0), ("450", leo_zenith_rate(450)[2]),
               ("800", leo_zenith_rate(800)[2]), ("1000", leo_zenith_rate(1000)[2])]
    for name, rmse, lat, r40, p40, _, _ in TRACKERS:
        offs = [f"{np.hypot(rmse, px_s * (lat * 1e-3)) * IFOV_URAD:6.1f} u" for _, px_s in regimes]
        print(f"{name:<26} | {r40:4.1f}% | {p40:4.1f}% | {rmse:4.2f}px | " + " | ".join(f"{o:>8}" for o in offs))
    # Exact Hohmann Delta-v from 800 km circular to 200 km perigee
    r1, r2 = R_EARTH + 800e3, R_EARTH + 200e3
    v_circ = np.sqrt(MU_EARTH / r1)
    v_trans = np.sqrt(MU_EARTH * (2.0 / r1 - 2.0 / (r1 + r2)))
    dv_deorbit = v_circ - v_trans
    print(f"\n=== 2. Reference AO Fine-Pointing Coupling (w_theta = 0.155 urad, Phi_0 = {PHI_0:.1f} kJ/m^2, 1000 pulses/pass) ===")
    print(f"Sanity check: Hohmann Delta-v (800 km -> 200 km perigee) = {dv_deorbit:.1f} m/s | FSM capture window = +/-{FSM_WINDOW_URAD:.0f} urad")
    print(f"Uncompensated latency tolerance for 0.10 urad @ 800 km (9.32 urad/ms): ~10.7 us (requires orbit prediction).")
    for err in (0.0, 0.05, 0.09, 0.12, 0.15, 0.20):
        phi = float(delivered_fluence(err))
        dv = float(expected_dv_per_pass_m_s(err))
        print(f"  Pointing error = {err:4.2f} urad -> Phi = {phi:6.1f} kJ/m^2 -> Expected Delta-v/pass = {dv:6.1f} m/s")


def make_figure():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.8, 5.4), dpi=200)
    plt.rcParams.update({"font.size": 9.5})

    # Panel A: Coarse Acquisition & Handoff (SNN vs. Val-Tuned Event-Binned Frame Proxies)
    alts = np.array([450, 800, 1000])
    for label, rmse_px, lat_ms, r40, p40, col, mk in TRACKERS:
        errs = [np.hypot(rmse_px, leo_zenith_rate(a)[2] * (lat_ms * 1e-3)) * IFOV_URAD for a in alts]
        ax1.plot(alts, errs, marker=mk, lw=2.2, ms=7, color=col,
                 label=f"{label} (R={r40:.0f}%, P={p40:.0f}%)")

    ax1.axhline(FSM_WINDOW_URAD, color="#64748b", ls="--", lw=1.4, label="FSM handoff window (±250 µrad) [Assumed]")
    ax1.set_yscale("log")
    ax1.set_xticks(alts)
    ax1.set_xlabel("Circular LEO Altitude (km)")
    ax1.set_ylabel("Uncompensated Pointing Offset (µrad, log)")
    ax1.set_title("A. Coarse Acquisition & Handoff (31.0 µrad/px [Assumed])")
    ax1.grid(True, which="both", alpha=0.25)
    ax1.legend(fontsize=7.6, loc="lower left")

    # Panel B: Reference 13 m AO Fine-Pointing Loop (Expected Delta-v per pass vs. Pointing Error)
    err_u = np.linspace(0.0, 0.25, 300)
    dv_106 = expected_dv_per_pass_m_s(err_u, W_REF_URAD, PHI_0)
    dv_53 = expected_dv_per_pass_m_s(err_u, W_REF_URAD, PHI_OPT)

    ax2.plot(err_u, dv_106, lw=2.4, color="#00d4ff", label="Φ₀ = 106 kJ/m² (2× margin [Assumed])")
    ax2.plot(err_u, dv_53, lw=2.2, color="#facc15", ls="--", label="Φ₀ = 53 kJ/m² (Φ_opt [Phipps 2012])")
    ax2.axvline(W_REF_URAD, color="#a855f7", ls=":", lw=1.6, label=f"13 m AO 1/e² radius ({W_REF_URAD} µrad)")
    ax2.axhline(165.0, color="#ef4444", ls="-.", lw=1.3, label="~165 m/s (800→200 km perigee drop)")

    ax2.set_xlabel("Fine-Track Pointing Error θ_err (µrad) [Orbit-Predicted AO Loop]")
    ax2.set_ylabel("Expected Δv per 1000-Pulse Pass (m/s)")
    ax2.set_title("B. Reference 13 m AO Loop: Expected Δv vs. Pointing Error")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=8.0, loc="upper right")

    plt.tight_layout()
    out_path = "results/lodr_design_curve.png"
    plt.savefig(out_path, dpi=200)
    print(f"\nSaved reference-design LODR figure to {out_path}")


if __name__ == "__main__":
    print_tables()
    make_figure()
