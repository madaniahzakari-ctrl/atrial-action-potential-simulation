# ==============================================================================
# Research Project 1: Multi-Figure Computational Modeling of Atrial Excitability
# Targeted Journal: Saudi Journal of Medicine and Medical Sciences (SJMMS)
# Author: Dr. Madaniah Omar Zakari - Department of Medical Physiology
# Repository: atrial-action-potential-simulation
# ==============================================================================

import numpy as np
import matplotlib.pyplot as plt

# 1. TIME INITIALIZATION
T = 50.0       
dt = 0.01      
time = np.arange(0, T, dt)

# 2. CELL MEMBRANE BIOPHYSICAL CONSTANTS
Cm = 1.0       
E_Na = 50.0    
E_K = -77.0    
E_L = -54.4    
g_Na = 120.0   
g_L = 0.3      

# 3. CORE COMPUTATIONAL ENGINE FOR ATRIAL MYOCYTE MODEL
def run_full_atrial_model(g_K_value):
    V = -65.0   
    m = 0.05    
    h = 0.6     
    n = 0.32    
    voltage_profile = []
    
    for t in time:
        I_ext = 10.0 if 5.0 <= t <= 6.0 else 0.0
        
        alpha_m = 0.1 * (V + 40.0) / (1.0 - np.exp(-(V + 40.0) / 10.0)) if V != -40.0 else 1.0
        beta_m = 4.0 * np.exp(-(V + 65.0) / 18.0)
        alpha_h = 0.07 * np.exp(-(V + 65.0) / 20.0)
        beta_h = 1.0 / (1.0 + np.exp(-(V + 35.0) / 10.0))
        alpha_n = 0.01 * (V + 55.0) / (1.0 - np.exp(-(V + 55.0) / 10.0)) if V != -55.0 else 0.1
        beta_n = 0.125 * np.exp(-(V + 65.0) / 80.0)
        
        m += (alpha_m * (1.0 - m) - beta_m * m) * dt
        h += (alpha_h * (1.0 - h) - beta_h * h) * dt
        n += (alpha_n * (1.0 - n) - beta_n * n) * dt
        
        I_Na = g_Na * (m**3) * h * (V - E_Na)
        I_K = g_K_value * (n**4) * (V - E_K)
        I_L = g_L * (V - E_L)
        
        dV = (I_ext - (I_Na + I_K + I_L)) / Cm
        V += dV * dt
        voltage_profile.append(V)
        
    return np.array(voltage_profile)

# Computing the Multi-Tiered Gradients
print("[Systems Biology] Computing separate curves for the multi-figure panel...")
V_control  = run_full_atrial_model(36.0)  # Healthy Control (100% Efficacy)
V_loss_25  = run_full_atrial_model(27.0)  # 25% Loss-of-Function
V_loss_50  = run_full_atrial_model(18.0)  # 50% Loss-of-Function
V_loss_75  = run_full_atrial_model(9.0)   # 75% Loss-of-Function
V_loss_100 = run_full_atrial_model(0.0)   # 100% Complete Knockout

# ------------------------------------------------------------------------------
# FIGURE 1: Baseline Human Atrial Action Potential (Control)
# ------------------------------------------------------------------------------
plt.figure(figsize=(10, 5), dpi=300)
plt.plot(time, V_control, color='forestgreen', linestyle='-', linewidth=2.5, label='Control (100% KCNA5)')
plt.title('Figure 1: Baseline Human Atrial Action Potential Under Physiological Control Conditions', fontsize=11, fontweight='bold')
plt.xlabel('Time (ms)', fontsize=10)
plt.ylabel('Membrane Potential (mV)', fontsize=10)
plt.axhline(0, color='black', linestyle=':', alpha=0.3)
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(loc='upper right', fontsize=9)
plt.savefig('Figure_1_Atrial_Control.png', bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------------------
# FIGURE 2: Early Compensatory Kinetics (Control vs 25% Loss)
# ------------------------------------------------------------------------------
plt.figure(figsize=(10, 5), dpi=300)
plt.plot(time, V_control, color='forestgreen', linestyle='-', linewidth=2.5, label='Control (100% KCNA5)')
plt.plot(time, V_loss_25, color='goldenrod', linestyle='-.', linewidth=2.0, label='25% Loss-of-Function')
plt.title('Figure 2: Compensatory Electrical Adjustments Under Mild Suppressive Gradients', fontsize=11, fontweight='bold')
plt.xlabel('Time (ms)', fontsize=10)
plt.ylabel('Membrane Potential (mV)', fontsize=10)
plt.axhline(0, color='black', linestyle=':', alpha=0.3)
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(loc='upper right', fontsize=9)
plt.savefig('Figure_2_Atrial_Compensatory.png', bbox_inches='tight')
plt.show()

# ------------------------------------------------------------------------------
# FIGURE 3: Pathological APD Prolongation (Advanced Knockout Tiers)
# ------------------------------------------------------------------------------
plt.figure(figsize=(10, 5), dpi=300)
plt.plot(time, V_loss_50, color='darkorange', linestyle='--', linewidth=2.0, label='50% Loss-of-Function')
plt.plot(time, V_loss_75, color='crimson', linestyle=':', linewidth=2.0, label='75% Loss-of-Function')
plt.plot(time, V_loss_100, color='darkred', linestyle='-', linewidth=2.5, label='100% Knockout (Severe AF Substrate)')
plt.title('Figure 3: Pathological Repolarization Breakdown Across Advanced Loss-of-Function Tiers', fontsize=11, fontweight='bold')
plt.xlabel('Time (ms)', fontsize=10)
plt.ylabel('Membrane Potential (mV)', fontsize=10)
plt.axhline(0, color='black', linestyle=':', alpha=0.3)
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(loc='upper right', fontsize=9)
plt.savefig('Figure_3_Atrial_Pathology.png', bbox_inches='tight')
plt.show()

print("\n[Success] All 3 high-resolution independent research figures exported successfully!")

