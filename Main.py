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
 
# 3. MULTI-TIERED GRADIENT SIMULATION ENGINE
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
 
# 4. SIMULATING THE GRADIENT TIERS
print("[Systems Biology] Computing multi-parametric gradient tiers...")
V_control  = run_full_atrial_model(36.0)  # Healthy Control (100% Efficacy)
V_loss_25  = run_full_atrial_model(27.0)  # 25% Loss-of-Function
V_loss_50  = run_full_atrial_model(18.0)  # 50% Loss-of-Function
V_loss_75  = run_full_atrial_model(9.0)   # 75% Loss-of-Function
V_loss_100 = run_full_atrial_model(0.0)   # 100% Complete Knockout
 
# 5. GRAPHICAL COMPILATION FOR FULL RESEARCH PAPER
plt.figure(figsize=(12, 6), dpi=300)
plt.plot(time, V_control,  color='forestgreen', linestyle='-',  linewidth=2.5, label='Control (100% KCNA5)')
plt.plot(time, V_loss_25,  color='goldenrod',   linestyle='-.', linewidth=2.0, label='25% Loss-of-Function')
plt.plot(time, V_loss_50,  color='darkorange',  linestyle='--', linewidth=2.0, label='50% Loss-of-Function')
plt.plot(time, V_loss_75,  color='crimson',     linestyle=':',  linewidth=2.0, label='75% Loss-of-Function')
plt.plot(time, V_loss_100, color='darkred',     linestyle='-',  linewidth=2.5, label='100% Knockout (Severe AF Substrate)')
 
plt.title('Figure 1: Multi-Parametric Gradient Analysis of Atrial Action Potential Repolarization Reserve', fontsize=12, fontweight='bold', pad=15)
plt.xlabel('Time (ms)', fontsize=11, fontweight='bold')
plt.ylabel('Membrane Potential (mV)', fontsize=11, fontweight='bold')
plt.axhline(0, color='black', linestyle=':', alpha=0.3)
plt.grid(True, linestyle=':', alpha=0.5)
plt.legend(loc='upper right', fontsize=10, frameon=True, shadow=True)
 
plt.savefig('Figure_1_Full_Paper_Gradient.png', format='png', bbox_inches='tight')
print("[Success] Expanded Full Paper Figure panel saved successfully!")
plt.show()
