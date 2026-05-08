import os
import sys

import numpy as np

# Add path to the main project folder where data_generation is located
main_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(main_folder)

import data_generation.helper_functions as help

# Parsimonious models
def parsimonious_model(t_stim=[50], d_stim=[2], a_stim=[-25], T=400):
    """Numerical solver for Parsimonious Rabbit model.

    Args:
        t_stim (list): Starttimes for rectangular stimulation pulses.
        d_stim (list): Duration of each stimulation pulse.
        a_stim (list): Intensity of each stimulation pulse.
        T (int): Maximum time up to which all functions should be 
            evaluated.

    Returns:
        t (arr): Time in ms.
        v (arr): Transmembrane potential in mV.
        m (arr): Fast inactivation gating variable, unitless.
        h (arr): Fast activation gating variable, unitless.
        I_Na (arr): Sodium current in uA/cm^2.
        I_K (arr): Potassium current in uA/cm^2.
        I_stim (arr): Stimulation current in uA/cm^2.
    """
    # Set up parameters
    Cm = 1         # uF/cm^2
    g_Na = 11      # mS/cm^2
    g_K = 0.3      # mS/cm^2
    v_Na = 65      # mV
    v_K = -83      # mV
    b = 0.047      # 1/mV
    Em, km = -41, -4    
    Eh, kh, tau_h_0, delta_h = -74.9, 4.4, 6.8, 0.8

    # Define currents
    I_Na = lambda v, m, h: g_Na*(m**3)*h*(v-v_Na)
    I_K = lambda v: g_K*np.exp(-b*(v-v_K))*(v-v_K)
    I_stim = lambda t: sum(
        a_stim_ * (t >= t_stim_) * (t <= t_stim_ + d_stim_) 
        for t_stim_, d_stim_, a_stim_ in zip(t_stim, d_stim, a_stim)
    )

    # Define rate constants
    m_inf = lambda v: 1/(1+np.exp((v-Em)/km))
    tau_m = lambda v: 0.12
    h_inf = lambda v: 1/(1+np.exp((v-Eh)/kh))
    tau_h = lambda v: 2*tau_h_0*np.exp(delta_h*(v-Eh)/kh)/(1+np.exp((v-Eh)/kh))

    # Set up discrerization
    T = T                               # Total simulation time (in ms)
    dt = 0.005                          # Time step (in ms)
    N = round(T/dt)                     # Number of time steps
    t = np.arange(0, T+dt, dt)[:, None] # Time vector

    # Set up solution vectors
    v = np.zeros((N+1, 1))
    m = np.zeros((N+1, 1))
    h = np.zeros((N+1, 1))

    # Define initial conditions
    v[0] = -83
    m[0] = 0
    h[0] = 0.86 # 0.9

    # Explicit numerical scheme
    for n in range(N):
        v[n+1] = v[n] - (dt/Cm)*(I_Na(v[n],m[n],h[n]) + I_K(v[n]) + I_stim(t[n]))
        m[n+1] = m[n] + dt*((m_inf(v[n])-m[n])/tau_m(v[n]))
        h[n+1] = max(h[n] + dt*((h_inf(v[n])-h[n])/tau_h(v[n])), 0)

    return t, v, m, h, I_Na(v,m,h), I_K(v), I_stim(t)

# if __name__ == "__main__":
#     # Test the solver
#     t, v, m, h, I_Na, I_K, I_stim = parsimonious_model(
#         t_stim=[50], d_stim=[2], a_stim=[-25], T=2000
#     )

#     help.plotting_impluse_action_potential(t, v, I_stim, T=1000, n_I=1)
