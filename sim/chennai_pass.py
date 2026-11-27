import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Safe for Wayland/Hyprland without GUI crashes
import matplotlib.pyplot as plt
from skyfield.api import load, wgs84

os.makedirs("data", exist_ok=True)
os.makedirs("docs", exist_ok=True)

ts = load.timescale()

# 1. Fetch live ISS TLE from CelesTrak (cached locally in data/iss.tle)
tle_url = "https://celestrak.org/NORAD/elements/gp.php?CATNR=25544&FORMAT=TLE"
sat = load.tle_file(tle_url, filename="data/iss.tle")[0]
print(f"Satellite : {sat.name}")
print(f"TLE Epoch : {sat.epoch.utc_strftime('%Y-%m-%d %H:%M:%S UTC')}")

# 2. Observer location: Chennai, India
chennai = wgs84.latlon(13.0827, 80.2707)

# 3. Find passes above 20 degrees elevation over the next 48 hours
t0 = ts.now()
t1 = ts.tt_jd(t0.tt + 2.0)
t_events, events = sat.find_events(chennai, t0, t1, altitude_degrees=20.0)

# Pair up complete (rise=0, culminate=1, set=2) passes
passes = []
i = 0
while i + 2 < len(events):
    if events[i] == 0 and events[i + 1] == 1 and events[i + 2] == 2:
        passes.append((t_events[i], t_events[i + 1], t_events[i + 2]))
        i += 3
    else:
        i += 1

if not passes:
    print("No passes above 20 deg in the next 48h. Try lowering altitude_degrees to 10.")
    raise SystemExit(0)

# Pick the highest-elevation pass
best_pass = max(passes, key=lambda p: (sat - chennai).at(p[1]).altaz()[0].degrees)
t_rise, t_culm, t_set = best_pass

# 4. Sample the pass every 1 second
duration_sec = int(round((t_set.tt - t_rise.tt) * 86400))
sec_offsets = np.arange(0, duration_sec + 1)
t_samples = ts.tt_jd(t_rise.tt + sec_offsets / 86400.0)

topo = (sat - chennai).at(t_samples)
alt, az, dist = topo.altaz()

# Inertial unit vectors (measures true apparent angular speed against background stars)
pos_km = topo.position.km  # shape (3, N)
u = pos_km / np.linalg.norm(pos_km, axis=0, keepdims=True)
dot_products = np.clip(np.sum(u[:, :-1] * u[:, 1:], axis=0), -1.0, 1.0)
ang_vel_deg_s = np.degrees(np.arccos(dot_products))
ang_vel_arcsec_s = ang_vel_deg_s * 3600.0

print(f"\nBest pass in next 48h:")
print(f"  Rise (UTC)     : {t_rise.utc_strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Peak (UTC)     : {t_culm.utc_strftime('%Y-%m-%d %H:%M:%S')}")
print(f"  Duration (>20°): {duration_sec} s")
print(f"  Max Elevation  : {alt.degrees.max():.2f} deg")
print(f"  Min Range      : {dist.km.min():.1f} km")
print(f"  Peak Ang Speed : {ang_vel_deg_s.max():.3f} deg/s ({ang_vel_arcsec_s.max():.1f} arcsec/s)")
print(f"  Est. Sensor Speed (at 6.4 arcsec/px): {ang_vel_arcsec_s.max() / 6.4:.1f} px/s")

# 5. Save a 2-panel figure: Sky Track + Apparent Angular Speed
fig = plt.figure(figsize=(11, 4.5))

ax1 = fig.add_subplot(1, 2, 1, projection="polar")
ax1.set_theta_zero_location("N")
ax1.set_theta_direction(-1)
ax1.plot(np.radians(az.degrees), 90.0 - alt.degrees, linewidth=2)
ax1.set_rlim(0, 90)
ax1.set_yticks([0, 30, 60, 90])
ax1.set_yticklabels(["90°", "60°", "30°", "0°"])
ax1.set_title("ISS Sky Track over Chennai")

ax2 = fig.add_subplot(1, 2, 2)
ax2.plot(sec_offsets[:-1], ang_vel_arcsec_s, linewidth=2)
ax2.set_xlabel("Time since rise above 20° (s)")
ax2.set_ylabel("Apparent Angular Speed (arcsec/s)")
ax2.set_title("Apparent Target Speed vs. Background Stars")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out_path = "docs/iss_pass_chennai.png"
plt.savefig(out_path, dpi=150)
print(f"\nSaved plot to {out_path}")
