# Neuromorphic Space Object Tracking: Open Benchmark & Mitigation Pointing Error Budget

**Author:** Krishiykeishav S (B.Tech Mechanical Engineering, VIT Vellore)  
**Repository:** https://github.com/hj6403082-glitch/neuromorphic-ssa-benchmark  
**Date:** November 2026  

---

## 1. Problem Statement & Motivation
Space Situational Awareness (SSA) for Low Earth Orbit (LEO) debris tracking faces a fundamental sensing bottleneck. Conventional frame-based optical sensors (30–60 FPS) suffer from severe motion blur and high readout latency when tracking fast-moving Resident Space Objects (RSOs). 

For example, propagating a real ISS pass over Chennai (13.08° N, 80.27° E; max elevation 49.59°, min range 533.4 km) yields a peak apparent angular velocity of **2,857.9 arcsec/s** against the background star field. On a typical SSA optical setup (6.4 arcsec/pixel), this translates to **446.5 pixels/second**—smearing the target across **~15 pixels per frame** at 30 FPS.

Dynamic Vision Sensors (event cameras) resolve this by asynchronously reporting per-pixel log-intensity changes at microsecond resolution with zero motion blur and low power. However, two gaps limit their adoption in SSA:
1. **Lack of an open, TLE-driven validated benchmark:** Real-sky datasets (e.g., EBSSA) are scarce and difficult to scale across orbital regimes, lighting conditions, and noise profiles.
2. **Unquantified systems-level impact:** Existing neuromorphic tracking papers report pixel-level detection metrics, but do not translate tracking latency and localization error into a downstream **debris-mitigation pointing error budget**.

## 2. Research Objectives & Deliverables
This project takes a software-first, simulation-validated approach across four phases (~20 weeks):

1. **Validated Open Simulation Benchmark (Phase 1):** Build a TLE-driven orbital propagator (`Skyfield`/`SGP4`) coupled to a physical DVS pixel emulator (`v2e`, DAVIS346 346×260) with sidereal star-field drift, background shot/leak noise, and automatic sub-pixel ground truth. Validate synthetic event rates and spatial spread statistically against real-sky **EBSSA** recordings, then release versioned train/val/test splits.
2. **Classical vs. Neuromorphic Baselines (Phase 2):** Benchmark a classical spatiotemporal clustering + Kalman filter tracker against a PyTorch Spiking Neural Network (`snnTorch`) and a quantized **BrainChip Akida (`MetaTF`)** converted model across detection rate, false-alarm rate, track continuity, localization error (arcsec/px), latency, and compute cost.
3. **Mitigation Pointing Error Budget (Phase 3):** Model a closed-loop ground-based laser momentum-transfer pointing system (textbook physics from published literature) driven by tracker outputs. Sweep tracking error and end-to-end latency to produce a design curve comparing frame-based vs. event-based tracking collapse thresholds.
4. **Open Release & Publication (Phase 4):** Publish the open-source codebase, dataset card, reproducible notebooks, and an arXiv preprint / workshop paper.

## 3. Initial Proof-of-Concept (Completed)
- **Orbital Propagation:** Verified live CelesTrak TLE propagation and apparent angular velocity extraction (`sim/chennai_pass.py`, `docs/iss_pass_chennai.png`).
- **Event Pipeline:** Verified GPU-accelerated `v2e` streaming (`--pipeframes` at 5,000 fps) generating **41,975 events** (21,477 ON / 20,498 OFF) over a 0.5 s synthetic ISS pass with synchronized sub-pixel ground truth (`sim/gen_synth_pass.py`, `docs/synth_iss_events.png`).
- **Neuromorphic Toolchain:** Verified `BrainChip MetaTF 2.19.3` software simulator and CUDA-accelerated PyTorch environments.

## 4. Guidance Requested
Seeking a faculty advisor (Aerospace / ECE / AI) for bi-weekly technical feedback, benchmark validation review, and guidance on target publication venues (IEEE / neuromorphic vision or SSA workshops).
