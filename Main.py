# ==============================================================================
# Research Project: Biophysical CRN Modeling of Human Atrial I_Kur Blockade
# Author: Dr. Madaniah Omar Zakari
# Affiliation: Department of Basic Medical Sciences (Physiology), College of Medicine,
#              Taibah University, Al-Madinah Al-Munawwarah, Saudi Arabia
# ==============================================================================
import numpy as np
import matplotlib.pyplot as plt

# 1. PHYSICAL & IONIC CONSTANTS (Courtemanche-Ramirez-Nattel 1998)
R = 8.3143       # J/(mol*K)
T = 310.0        # K (37 C)
F = 96.4867      # C/mmol
RTF = (R * T) / F

Cm = 100.0       # pF
V_cell = 20100.0 # um^3
Vi = V_cell * 0.68
Vup = 0.0552 * V_cell
Vrel = 0.0048 * V_cell

# Fixed Bulk Concentrations (mM)
Nao, Ko, Cao = 140.0, 5.4, 1.8
Nai, Ki = 11.2, 139.0

# Maximal Conductances (nS/pF = mS/uF)
gNa = 7.8
gK1 = 0.09
gto = 0.1652
gKr = 0.0294
gKs = 0.129
gCaL = 0.1238
gbNa = 0.000674
gbCa = 0.00113
INaK_max = 0.60
INaCa_max = 1600.0
IpCa_max = 0.275
Iup_max = 0.005
Ca_up_max = 15.0

def run_crn_simulation(gKur_scale=1.0, bcl=1000.0, n_beats=10, dt=0.01):
    """
    Solves the full Courtemanche-Ramirez-Nattel (1998) human atrial ODE system
    using Forward Euler integration across multiple beats to reach steady state.
    """
    total_time = n_beats * bcl
    n_steps = int(total_time / dt)
    
    # Initial State Variables (Resting Physiological Steady State)
    V = -81.18
    m, h, j = 0.002908, 0.9649, 0.9775
    oa, oi = 0.03043, 0.9992
    ua, ui = 0.004966, 0.9986
    xr, xs = 0.00003296, 0.01869
    d, f, fCa = 0.0001367, 0.9996, 0.7755
    u, v, w = 2.35e-112, 1.0, 0.9992
    Cai = 0.0001013
    Ca_up, Ca_rel = 1.488, 1.488
    
    # Nernst Potentials
    ENa = RTF * np.log(Nao / Nai)
    EK = RTF * np.log(Ko / Ki)
    
    # Storage for the final beat (400 ms window)
    record_window = min(400.0, bcl)
    n_rec = int(record_window / dt)
    t_rec = np.linspace(0, record_window, n_rec)
    V_rec = np.zeros(n_rec)
    
    last_beat_start = (n_beats - 1) * bcl
    rec_idx = 0
    
    for step in range(n_steps):
        t = step * dt
        beat_t = t % bcl
        
        # Stimulus current (-20 pA/pF for 2 ms at t = 20 ms of each beat)
        Istim = -20.0 if (20.0 <= beat_t < 22.0) else 0.0
        
        # 1. Fast Sodium Current (INa)
        am = 0.32 * (V + 47.13) / (1.0 - np.exp(-0.1 * (V + 47.13))) if abs(V + 47.13) > 1e-5 else 3.2
        bm = 0.08 * np.exp(-V / 11.0)
        ah = 0.135 * np.exp(-(V + 80.0) / 6.8) if V < -40.0 else 0.0
        bh = (3.56 * np.exp(0.079 * V) + 3.1e5 * np.exp(0.35 * V)) if V < -40.0 else 1.0 / (0.13 * (1.0 + np.exp(-(V + 10.66) / 11.1)))
        aj = ((-127140.0 * np.exp(0.2444 * V) - 3.474e-5 * np.exp(-0.04391 * V)) * (V + 37.78) / (1.0 + np.exp(0.311 * (V + 79.23)))) if V < -40.0 else 0.0
        bj = (0.1212 * np.exp(-0.01052 * V) / (1.0 + np.exp(-0.1378 * (V + 40.14)))) if V < -40.0 else (0.3 * np.exp(-2.535e-7 * V) / (1.0 + np.exp(-0.1 * (V + 32.0))))
        
        m += dt * (am * (1.0 - m) - bm * m)
        h += dt * (ah * (1.0 - h) - bh * h)
        j += dt * (aj * (1.0 - j) - bj * j)
        INa = gNa * (m**3) * h * j * (V - ENa)
        
        # 2. Time-Independent K+ Current (IK1)
        IK1 = gK1 * (V - EK) / (1.0 + np.exp(0.07 * (V + 80.0)))
        
        # 3. Transient Outward K+ Current (Ito)
        aoa = 0.65 / (np.exp(-(V + 10.0) / 8.5) + np.exp(-(V - 30.0) / 59.0))
        boa = 0.65 / (2.5 + np.exp((V + 82.0) / 17.0))
        tau_oa = 1.0 / (3.0 * (aoa + boa))
        oa_inf = 1.0 / (1.0 + np.exp(-(V + 20.47) / 17.54))
        
        aoi = 1.0 / (18.53 + np.exp((V + 113.7) / 10.95))
        boi = 1.0 / (35.56 + np.exp(-(V + 1.26) / 7.44))
        tau_oi = 1.0 / (3.0 * (aoi + boi))
        oi_inf = 1.0 / (1.0 + np.exp((V + 43.1) / 5.3))
        
        oa += dt * ((oa_inf - oa) / tau_oa)
        oi += dt * ((oi_inf - oi) / tau_oi)
        Ito = gto * (oa**3) * oi * (V - EK)
        
        # 4. Ultra-Rapid Delayed Rectifier K+ Current (IKur - KCNA5 Target)
        gKur_eff = gKur_scale * (0.005 + 0.05 / (1.0 + np.exp(-(V - 15.0) / 13.0)))
        aua = 0.65 / (np.exp(-(V + 10.0) / 8.5) + np.exp(-(V - 30.0) / 59.0))
        bua = 0.65 / (2.5 + np.exp((V + 82.0) / 17.0))
        tau_ua = 1.0 / (3.0 * (aua + bua))
        ua_inf = 1.0 / (1.0 + np.exp(-(V + 30.3) / 9.6))
        
        aui = 1.0 / (21.0 + np.exp(-(V - 185.0) / 28.0))
        bui = np.exp((V - 158.0) / 16.0)
        tau_ui = 1.0 / (3.0 * (aui + bui))
        ui_inf = 1.0 / (1.0 + np.exp((V - 99.45) / 27.48))
        
        ua += dt * ((ua_inf - ua) / tau_ua)
        ui += dt * ((ui_inf - ui) / tau_ui)
        IKur = gKur_eff * (ua**3) * ui * (V - EK)
        
        # 5. Rapid & Slow Delayed Rectifier K+ Currents (IKr, IKs)
        axr = 0.0003 * (V + 14.1) / (1.0 - np.exp(-(V + 14.1) / 5.0)) if abs(V + 14.1) > 1e-5 else 0.0015
        bxr = 0.000073898 * (V - 3.3328) / (np.exp((V - 3.3328) / 5.1237) - 1.0) if abs(V - 3.3328) > 1e-5 else 0.00037836
        tau_xr = 1.0 / (axr + bxr)
        xr_inf = 1.0 / (1.0 + np.exp(-(V + 14.1) / 6.5))
        xr += dt * ((xr_inf - xr) / tau_xr)
        IKr = gKr * xr * (V - EK) / (1.0 + np.exp((V + 15.0) / 22.4))
        
        axs = 0.00004 * (V - 19.9) / (1.0 - np.exp(-(V - 19.9) / 17.0)) if abs(V - 19.9) > 1e-5 else 0.00068
        bxs = 0.000035 * (V - 19.9) / (np.exp((V - 19.9) / 9.0) - 1.0) if abs(V - 19.9) > 1e-5 else 0.000315
        tau_xs = 0.5 / (axs + bxs)
        xs_inf = 1.0 / np.sqrt(1.0 + np.exp(-(V - 19.9) / 12.7))
        xs += dt * ((xs_inf - xs) / tau_xs)
        IKs = gKs * (xs**2) * (V - EK)
        
        # 6. L-Type Ca2+ Current (ICaL)
        tau_d = (1.0 - np.exp(-(V + 10.0) / 6.24)) / (0.035 * (V + 10.0) * (1.0 + np.exp(-(V + 10.0) / 6.24))) if abs(V + 10.0) > 1e-5 else 4.579
        d_inf = 1.0 / (1.0 + np.exp(-(V + 10.0) / 8.0))
        tau_f = 9.0 / (0.0197 * np.exp(-(0.0337 * (V + 10.0))**2) + 0.02)
        f_inf = 1.0 / (1.0 + np.exp((V + 28.0) / 6.9))
        fCa_inf = 1.0 / (1.0 + Cai / 0.00035)
        
        d += dt * ((d_inf - d) / tau_d)
        f += dt * ((f_inf - f) / tau_f)
        fCa += dt * ((fCa_inf - fCa) / 2.0)
        ICaL = gCaL * d * f * fCa * (V - 65.0)
        
        # 7. Pumps, Exchangers & Background Currents
        ECa = 0.5 * RTF * np.log(Cao / Cai)
        IbNa = gbNa * (V - ENa)
        IbCa = gbCa * (V - ECa)
        
        sigma = (np.exp(Nao / 67.3) - 1.0) / 7.0
        fNaK = 1.0 / (1.0 + 0.1245 * np.exp(-0.1 * V / RTF) + 0.0365 * sigma * np.exp(-V / RTF))
        INaK = INaK_max * fNaK * (1.0 / (1.0 + (10.0 / Nai)**1.5)) * (Ko / (Ko + 1.5))
        
        INaCa = (INaCa_max * (np.exp(0.35 * V / RTF) * (Nai**3) * Cao - np.exp(-0.65 * V / RTF) * (Nao**3) * Cai) /
                 ((87.5**3 + Nao**3) * (1.38 + Cao) * (1.0 + 0.1 * np.exp(-0.65 * V / RTF))))
        IpCa = IpCa_max * Cai / (0.0005 + Cai)
        
        # 8. Intracellular Ca2+ Handling (SR Release & Uptake)
        Fn = 1e-12 * Vrel * (0.5 * ICaL - 0.2 * INaCa)
        u_inf = 1.0 / (1.0 + np.exp(-(Fn - 3.4175e-13) / 1.367e-15))
        tau_v = 1.91 + 2.09 / (1.0 + np.exp(-(Fn - 3.4175e-13) / 1.367e-15))
        v_inf = 1.0 - 1.0 / (1.0 + np.exp(-(Fn - 6.835e-14) / 1.367e-15))
        tau_w = 6.0 * (1.0 - np.exp(-(V - 7.9) / 5.0)) / ((1.0 + 0.3 * np.exp(-(V - 7.9) / 5.0)) * (V - 7.9)) if abs(V - 7.9) > 1e-5 else 6.0
        w_inf = 1.0 - 1.0 / (1.0 + np.exp(-(V - 40.0) / 17.0))
        
        u += dt * ((u_inf - u) / 8.0)
        v += dt * ((v_inf - v) / tau_v)
        w += dt * ((w_inf - w) / tau_w)
        
        Irel = 30.0 * (u**2) * v * w * (Ca_rel - Cai)
        Itr = (Ca_up - Ca_rel) / 180.0
        Iup = Iup_max / (1.0 + 0.00092 / Cai)
        Iupleak = Iup_max * Ca_up / Ca_up_max
        
        dCa_up = Iup - Iupleak - Itr * (Vrel / Vup)
        dCa_rel = Itr - Irel
        dCai_total = (2.0 * INaCa - IpCa - ICaL - IbCa) / (2.0 * Vi * F) + (Iupleak - Iup) * (Vup / Vi) + Irel * (Vrel / Vi)
        
        # Buffering factor for cytosolic Ca2+
        b_cmdn = 0.05 * 0.00238 / ((Cai + 0.00238)**2)
        b_trpn = 0.07 * 0.0005 / ((Cai + 0.0005)**2)
        Cai += dt * (dCai_total / (1.0 + b_cmdn + b_trpn))
        Ca_up += dt * dCa_up
        Ca_rel += dt * dCa_rel
        
        # 9. Total Ionic Current & Membrane Potential Update
        Iion = INa + IK1 + Ito + IKur + IKr + IKs + ICaL + IbNa + IbCa + INaK + INaCa + IpCa
        V += dt * (-(Iion + Istim))
        
        if t >= last_beat_start and rec_idx < n_rec:
            V_rec[rec_idx] = V
            rec_idx += 1
            
    return t_rec, V_rec

def extract_ap_metrics(t, V):
    V_rest = np.mean(V[:int(18.0 / 0.01)])
    idx_peak = np.argmax(V)
    V_peak = V[idx_peak]
    APA = V_peak - V_rest
    
    V50 = V_peak - 0.50 * APA
    V90 = V_peak - 0.90 * APA
    
    post_peak_t = t[idx_peak:]
    post_peak_V = V[idx_peak:]
    
    t50 = post_peak_t[np.where(post_peak_V <= V50)[0][0]] - 20.0
    t90 = post_peak_t[np.where(post_peak_V <= V90)[0][0]] - 20.0
    return round(V_rest, 1), round(APA, 1), round(t50, 1), round(t90, 1)

# Run simulations at 1.0 Hz (BCL = 1000 ms) for all 5 blockade tiers
tiers = [
    ("Control (0% Block)", 1.00),
    ("Mild (25% Block)", 0.75),
    ("Moderate (50% Block / IC50)", 0.50),
    ("Severe (75% Block)", 0.25),
    ("Knockout (100% Block)", 0.00),
]

results = {}
print("=== CRN BIOPHYSICAL SIMULATION RESULTS (1.0 Hz Steady State) ===")
for name, scale in tiers:
    t_arr, V_arr = run_crn_simulation(gKur_scale=scale, bcl=1000.0, n_beats=8)
    vm, apa, apd50, apd90 = extract_ap_metrics(t_arr, V_arr)
    results[scale] = (t_arr, V_arr)
    print(f"{name:28s} | Vm = {vm:5.1f} mV | APA = {apa:5.1f} mV | APD50 = {apd50:5.1f} ms | APD90 = {apd90:5.1f} ms")

# Generate Publication-Quality Figures (300 DPI)
t_plot = results[1.00][0]

# FIGURE 1




