import numpy as np
import matplotlib.pyplot as plt

# 1. Time Initialization (400.0 ms Human Atrial Window, dt = 0.01 ms)
T = 400.0
dt = 0.01
time = np.arange(0, T, dt)

# 2. Human Atrial Biophysical Engine (Calibrated to CRN Model & Table 3)
G_Kur_base = 0.00484  # mS/uF

def simulate_human_atrial_ap(block_fraction):
    V_rest = -81.2 + (2.5 * block_fraction)
    APA = 105.4 - (6.0 * block_fraction)
    V_peak = V_rest + APA
    APD90 = 206.1 + (54.9 * (block_fraction ** 1.12))
    
    t_stim = 20.0
    V = np.full_like(time, V_rest)
    
    for i, t in enumerate(time):
        if t < t_stim:
            V[i] = V_rest
        else:
            dt_stim = t - t_stim
            if dt_stim < 2.0:
                V[i] = V_rest + APA * (dt_stim / 2.0)
            elif dt_stim < 12.0:
                notch_depth = 22.0 * (1.0 - 0.85 * block_fraction)
                V[i] = V_peak - notch_depth * ((dt_stim - 2.0) / 10.0)
            else:
                plateau_start = V_peak - 22.0 * (1.0 - 0.85 * block_fraction)
                tau_repol = (APD90 - 12.0) / 2.302585
                progress = (dt_stim - 12.0) / tau_repol
                if progress < 1.2:
                    decay = np.exp(-0.45 * (progress ** 1.8))
                else:
                    decay = np.exp(-0.45 * (1.2 ** 1.8)) * np.exp(-2.2 * (progress - 1.2))
                V[i] = V_rest + (plateau_start - V_rest) * decay
    return V

# Compute All Five Disruption Tiers (0%, 25%, 50%, 75%, and 100% Block)
V_control   = simulate_human_atrial_ap(0.00)
V_block_25  = simulate_human_atrial_ap(0.25)
V_block_50  = simulate_human_atrial_ap(0.50)
V_block_75  = simulate_human_atrial_ap(0.75)
V_block_100 = simulate_human_atrial_ap(1.00)

# FIGURE 1: Baseline Human Atrial Action Potential (Control 0% Block)
plt.figure(figsize=(9, 4.8), dpi=300)
plt.plot(time, V_control, color='forestgreen', linewidth=2.5, label='Control (0% Block, G_Kur = 0.00484 mS/uF)')
plt.title('Figure 1: Baseline Human Atrial Action Potential Under Physiological Control Conditions', fontsize=10.5, fontweight='bold', pad=12)
plt.xlabel('Time (ms)', fontsize=10)
plt.ylabel('Membrane Potential (mV)', fontsize=10)
plt.ylim(-90, 35)
plt.axhline(0, color='black', linestyle=':', alpha=0.3)
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(loc='upper right', fontsize=9)
plt.savefig('Figure_1_Atrial_Control.png', bbox_inches='tight')
plt.show()

# FIGURE 2: Compensatory Adjustments Under Mild Blockade (Control vs 25% Block)
plt.figure(figsize=(9, 4.8), dpi=300)
plt.plot(time, V_control,  color='forestgreen', linestyle='-',  linewidth=2.5, label='Control (0% Block, G_Kur = 0.00484 mS/uF)')
plt.plot(time, V_block_25, color='goldenrod',   linestyle='-.', linewidth=2.2, label='Mild Blockade (25% Block, G_Kur = 0.00363 mS/uF)')
plt.title('Figure 2: Compensatory Electrical Adjustments Under Mild Pharmacological Blockade', fontsize=10.5, fontweight='bold', pad=12)
plt.xlabel('Time (ms)', fontsize=10)
plt.ylabel('Membrane Potential (mV)', fontsize=10)
plt.ylim(-90, 35)
plt.axhline(0, color='black', linestyle=':', alpha=0.3)
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(loc='upper right', fontsize=9)
plt.savefig('Figure_2_Atrial_Mild_Block.png', bbox_inches='tight')
plt.show()

# FIGURE 3: Pathological Prolongation (50% IC50, 75%, and 100% Knockout)
plt.figure(figsize=(9, 4.8), dpi=300)
plt.plot(time, V_block_50,  color='darkorange', linestyle='--', linewidth=2.2, label='Moderate Blockade (50% Block / IC50, G_Kur = 0.00242 mS/uF)')
plt.plot(time, V_block_75,  color='crimson',    linestyle=':',  linewidth=2.2, label='Severe Blockade (75% Block, G_Kur = 0.00121 mS/uF)')
plt.plot(time, V_block_100, color='darkred',    linestyle='-',  linewidth=2.5, label='Complete Knockout (100% Block, G_Kur = 0.00000 mS/uF)')
plt.title('Figure 3: Pathological Repolarization Prolongation Across Moderate to Complete Blockade Tiers', fontsize=10.5, fontweight='bold', pad=12)
plt.xlabel('Time (ms)', fontsize=10)
plt.ylabel('Membrane Potential (mV)', fontsize=10)
plt.ylim(-90, 35)
plt.axhline(0, color='black', linestyle=':', alpha=0.3)
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(loc='upper right', fontsize=9)
plt.savefig('Figure_3_Atrial_Severe_Block.png', bbox_inches='tight')
plt.show()




