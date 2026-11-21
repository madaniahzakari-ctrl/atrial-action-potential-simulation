import numpy as np
import matplotlib.pyplot as plt
from numba import njit
R, T, F = 8.3143, 310.0, 96.4867
RTF = (R * T) / F
Nao, Ko, Cao = 140.0, 5.4, 1.8
Nai, Ki = 11.2, 139.0
gNa, gK1, gto, gKr, gKs, gCaL = 7.8, 0.09, 0.1652, 0.0294, 0.129, 0.1238
gbNa, gbCa, INaK_max, INaCa_max, IpCa_max = 0.000674, 0.00113, 0.60, 1600.0, 0.275
Iup_max, Ca_up_max = 0.005, 15.0
V_cell = 20100.0
Vi, Vup, Vrel = V_cell * 0.68, 0.0552 * V_cell, 0.0048 * V_cell
@njit
def run_crn_plot(gKur_scale=1.0, bcl=1000.0, n_beats=200, dt=0.005):
    steps_per_beat = int(round(bcl / dt))
    n_steps = n_beats * steps_per_beat
    rec_dur = 400.0 if bcl >= 1000.0 else 800.0
    rec_steps = int(round(rec_dur / dt))
    last_beat_step = (n_beats - 1) * steps_per_beat if bcl >= 1000.0 else (n_steps - rec_steps)
    
    V = -81.18
    m, h, j = 0.002908, 0.9649, 0.9775
    oa, oi = 0.03043, 0.9992
    ua, ui = 0.004966, 0.9986
    xr, xs = 0.00003296, 0.01869
    d, f, fCa = 0.0001367, 0.9996, 0.7755
    u, v, w = 2.35e-112, 1.0, 0.9992
    Cai, Ca_up, Ca_rel = 0.0001013, 1.488, 1.488
    
    ENa = RTF * np.log(Nao / Nai)
    EK = RTF * np.log(Ko / Ki)
    
    t_rec = np.arange(rec_steps) * dt
    V_rec = np.zeros(rec_steps)
    
    for step in range(n_steps):
        beat_step = step % steps_per_beat
        beat_t = beat_step * dt
        Istim = -20.0 if (20.0 <= beat_t < 22.0) else 0.0
        
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
        
        IK1 = gK1 * (V - EK) / (1.0 + np.exp(0.07 * (V + 80.0)))
        
        aoa = 0.65 / (np.exp(-(V + 10.0) / 8.5) + np.exp(-(V - 30.0) / 59.0))
        boa = 0.65 / (2.5 + np.exp((V + 82.0) / 17.0))
        oa += dt * (((1.0 / (1.0 + np.exp(-(V + 20.47) / 17.54))) - oa) * 3.0 * (aoa + boa))
        aoi = 1.0 / (18.53 + np.exp((V + 113.7) / 10.95))
        boi = 1.0 / (35.56 + np.exp(-(V + 1.26) / 7.44))
        oi += dt * (((1.0 / (1.0 + np.exp((V + 43.1) / 5.3))) - oi) * 3.0 * (aoi + boi))
        Ito = gto * (oa**3) * oi * (V - EK)
        
        gKur_eff = gKur_scale * (0.005 + 0.05 / (1.0 + np.exp(-(V - 15.0) / 13.0)))
        aua = 0.65 / (np.exp(-(V + 10.0) / 8.5) + np.exp(-(V - 30.0) / 59.0))
        bua = 0.65 / (2.5 + np.exp((V + 82.0) / 17.0))
        ua += dt * (((1.0 / (1.0 + np.exp(-(V + 30.3) / 9.6))) - ua) * 3.0 * (aua + bua))
        aui = 1.0 / (21.0 + np.exp(-(V - 185.0) / 28.0))
        bui = np.exp((V - 158.0) / 16.0)
        ui += dt * (((1.0 / (1.0 + np.exp((V - 99.45) / 27.48))) - ui) * 3.0 * (aui + bui))
        IKur = gKur_eff * (ua**3) * ui * (V - EK)
        
        axr = 0.0003 * (V + 14.1) / (1.0 - np.exp(-(V + 14.1) / 5.0)) if abs(V + 14.1) > 1e-5 else 0.0015
        bxr = 0.000073898 * (V - 3.3328) / (np.exp((V - 3.3328) / 5.1237) - 1.0) if abs(V - 3.3328) > 1e-5 else 0.00037836
        xr += dt * (((1.0 / (1.0 + np.exp(-(V + 14.1) / 6.5))) - xr) * (axr + bxr))
        IKr = gKr * xr * (V - EK) / (1.0 + np.exp((V + 15.0) / 22.4))
        
        axs = 0.00004 * (V - 19.9) / (1.0 - np.exp(-(V - 19.9) / 17.0)) if abs(V - 19.9) > 1e-5 else 0.00068
        bxs = 0.000035 * (V - 19.9) / (np.exp((V - 19.9) / 9.0) - 1.0) if abs(V - 19.9) > 1e-5 else 0.000315
        xs += dt * (((1.0 / np.sqrt(1.0 + np.exp(-(V - 19.9) / 12.7))) - xs) * 2.0 * (axs + bxs))
        IKs = gKs * (xs**2) * (V - EK)
        
        tau_d = (1.0 - np.exp(-(V + 10.0) / 6.24)) / (0.035 * (V + 10.0) * (1.0 + np.exp(-(V + 10.0) / 6.24))) if abs(V + 10.0) > 1e-5 else 4.579
        tau_f = 9.0 / (0.0197 * np.exp(-(0.0337 * (V + 10.0))**2) + 0.02)
        d += dt * (((1.0 / (1.0 + np.exp(-(V + 10.0) / 8.0))) - d) / tau_d)
        f += dt * (((1.0 / (1.0 + np.exp((V + 28.0) / 6.9))) - f) / tau_f)
        fCa += dt * (((1.0 / (1.0 + Cai / 0.00035)) - fCa) / 2.0)
        ICaL = gCaL * d * f * fCa * (V - 65.0)
        
        ECa = 0.5 * RTF * np.log(Cao / Cai)
        IbNa, IbCa = gbNa * (V - ENa), gbCa * (V - ECa)
        sigma = (np.exp(Nao / 67.3) - 1.0) / 7.0
        fNaK = 1.0 / (1.0 + 0.1245 * np.exp(-0.1 * V / RTF) + 0.0365 * sigma * np.exp(-V / RTF))
        INaK = INaK_max * fNaK * (1.0 / (1.0 + (10.0 / Nai)**1.5)) * (Ko / (Ko + 1.5))
        INaCa = (INaCa_max * (np.exp(0.35 * V / RTF) * (Nai**3) * Cao - np.exp(-0.65 * V / RTF) * (Nao**3) * Cai) /
                 ((87.5**3 + Nao**3) * (1.38 + Cao) * (1.0 + 0.1 * np.exp(-0.65 * V / RTF))))
        IpCa = IpCa_max * Cai / (0.0005 + Cai)
        
        Fn = 1e-12 * Vrel * (0.5 * ICaL - 0.2 * INaCa)
        u += dt * (((1.0 / (1.0 + np.exp(-(Fn - 3.4175e-13) / 1.367e-15))) - u) / 8.0)
        tau_v = 1.91 + 2.09 / (1.0 + np.exp(-(Fn - 3.4175e-13) / 1.367e-15))
        v += dt * (((1.0 - 1.0 / (1.0 + np.exp(-(Fn - 6.835e-14) / 1.367e-15))) - v) / tau_v)
        tau_w = 6.0 * (1.0 - np.exp(-(V - 7.9) / 5.0)) / ((1.0 + 0.3 * np.exp(-(V - 7.9) / 5.0)) * (V - 7.9)) if abs(V - 7.9) > 1e-5 else 6.0
        w += dt * (((1.0 - 1.0 / (1.0 + np.exp(-(V - 40.0) / 17.0))) - w) / tau_w)
        
        Irel = 30.0 * (u**2) * v * w * (Ca_rel - Cai)
        Itr = (Ca_up - Ca_rel) / 180.0
        Iup = Iup_max / (1.0 + 0.00092 / Cai)
        Iupleak = Iup_max * Ca_up / Ca_up_max
        
        Ca_up += dt * (Iup - Iupleak - Itr * (Vrel / Vup))
        Ca_rel += dt * (Itr - Irel)
        dCai = (2.0 * INaCa - IpCa - ICaL - IbCa) / (2.0 * Vi * F) + (Iupleak - Iup) * (Vup / Vi) + Irel * (Vrel / Vi)
        b_cmdn = 0.05 * 0.00238 / ((Cai + 0.00238)**2)
        b_trpn = 0.07 * 0.0005 / ((Cai + 0.0005)**2)
        Cai += dt * (dCai / (1.0 + b_cmdn + b_trpn))
        
        Iion = INa + IK1 + Ito + IKur + IKr + IKs + ICaL + IbNa + IbCa + INaK + INaCa + IpCa
        V += dt * (-(Iion + Istim))
        
        if step >= last_beat_step and (step - last_beat_step) < rec_steps:
            V_rec[step - last_beat_step] = V
            
    return t_rec, V_rec

t_p, V_ctrl = run_crn_plot(1.00)
_, V_25 = run_crn_plot(0.75)
_, V_50 = run_crn_plot(0.50)
_, V_75 = run_crn_plot(0.25)
_, V_100 = run_crn_plot(0.00)

print("Min/Max Control V:", round(float(np.min(V_ctrl)), 1), "to", round(float(np.max(V_ctrl)), 1))

# FIGURE 1
plt.figure(figsize=(9, 4.8), dpi=300)
plt.plot(t_p, V_ctrl, color='forestgreen', linewidth=2.5, label='Control (0% Block, 100% G_Kur)')
plt.title('Figure 1: Baseline Human Atrial Action Potential (CRN Biophysical Model)', fontsize=10.5, fontweight='bold')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.ylim(-95, 50)
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(loc='upper right')
plt.savefig('Figure_1_Atrial_Control.png', bbox_inches='tight')
plt.show()

# FIGURE 2
plt.figure(figsize=(9, 4.8), dpi=300)
plt.plot(t_p, V_ctrl, color='forestgreen', linewidth=2.5, label='Control (0% Block)')
plt.plot(t_p, V_25, color='goldenrod', linestyle='-.', linewidth=2.2, label='Mild Blockade (25% Block)')
plt.title('Figure 2: Compensatory Repolarization Adjustments Under Mild I_Kur Blockade', fontsize=10.5, fontweight='bold')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.ylim(-95, 50)
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(loc='upper right')
plt.savefig('Figure_2_Atrial_Mild_Block.png', bbox_inches='tight')
plt.show()

# FIGURE 3
plt.figure(figsize=(9, 4.8), dpi=300)
plt.plot(t_p, V_ctrl, color='gray', linestyle='--', alpha=0.6, linewidth=1.8, label='Control Reference (0% Block)')
plt.plot(t_p, V_50, color='darkorange', linestyle='--', linewidth=2.2, label='Moderate Blockade (50% Block / IC50)')
plt.plot(t_p, V_75, color='crimson', linestyle=':', linewidth=2.2, label='Severe Blockade (75% Block)')
plt.plot(t_p, V_100, color='darkred', linewidth=2.5, label='Complete Knockout (100% Block)')
plt.title('Figure 3: Graded I_Kur Blockade Effects on Atrial Repolarization and Plateau Dome', fontsize=10.5, fontweight='bold')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.ylim(-95, 50)
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(loc='upper right')
plt.savefig('Figure_3_Atrial_Severe_Block.png', bbox_inches='tight')
plt.show()
def get_metrics(t, V, dt=0.005):
    mdp = float(np.min(V[:int(20.0 / dt)]))
    up_end = int(30.0 / dt)
    idx_p0 = int(np.argmax(V[:up_end]))
    v_p0 = float(V[idx_p0])
    apa = v_p0 - mdp
    v50, v90 = v_p0 - 0.5 * apa, v_p0 - 0.9 * apa
    apd50 = float(t[idx_p0:][np.where(V[idx_p0:] <= v50)].item(0) - t[idx_p0])
    apd90 = float(t[idx_p0:][np.where(V[idx_p0:] <= v90)].item(0) - t[idx_p0])
    return round(mdp, 1), round(apa, 1), round(apd50, 1), round(apd90, 1)

for lbl, v_arr in [("Control", V_ctrl), ("Mild 25%", V_25), ("Moderate 50%", V_50), ("Severe 75%", V_75), ("Knockout 100%", V_100)]:
    mdp, apa, apd50, apd90 = get_metrics(t_p, v_arr)
    print(f"{lbl}: MDP={mdp} mV, APA={apa} mV, APD50={apd50} ms, APD90={apd90} ms")
    
t_hf, V_hf50 = run_crn_plot(0.50, bcl=400.0, n_beats=200, dt=0.005)
_, V_hf75 = run_crn_plot(0.25, bcl=400.0, n_beats=200, dt=0.005)
half = int(400.0 / 0.005)

for lbl, v_hf in [("2.5Hz_50%", V_hf50), ("2.5Hz_75%", V_hf75)]:
    b199 = get_metrics(t_hf[:half], v_hf[:half])
    b200 = get_metrics(t_hf[half:] - 400.0, v_hf[half:])
    print(f"{lbl} Beat199 APD90={b199[3]} ms | Beat200 APD90={b200[3]} ms")

plt.figure(figsize=(9, 4.8), dpi=300)
plt.plot(t_hf, V_hf50, color='darkorange', linestyle='--', linewidth=2.2, label='Moderate Blockade (50% Block) at 2.5 Hz')
plt.plot(t_hf, V_hf75, color='crimson', linewidth=2.5, label='Severe Blockade (75% Block) at 2.5 Hz')
plt.title('Figure 4: High-Frequency Pacing (2.5 Hz) Repolarization Instability Across Consecutive Beats', fontsize=10.5, fontweight='bold')
plt.xlabel('Time (ms)')
plt.ylabel('Membrane Potential (mV)')
plt.ylim(-95, 50)
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(loc='upper right')
plt.savefig('Figure_4_Atrial_Rate_Stress.png', bbox_inches='tight')
plt.show()    
        





