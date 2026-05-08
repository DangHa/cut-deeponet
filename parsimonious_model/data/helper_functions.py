import os
import sys

import matplotlib.pyplot as plt

# Add path to the main project folder where data_generation is located
main_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(main_folder)

# Show v vs t plot
def plotting_impluse_action_potential(t, v, I_stim, T, n_I):
    # Filepath for saving plot
    output_dir = "data_and_results/plotting"
    os.makedirs(output_dir, exist_ok=True)

    # Plot
    fig, ax = plt.subplots(3,1, figsize=(10, 10))
    ax[0].plot(t, v)
    ax[0].set_xlabel("t [ms]")
    ax[0].set_ylabel("v [mV]")
    ax[0].title.set_text(f"Plot for T={T}ms and {n_I} pulses")
    ax[1].plot(t, I_stim)
    ax[1].set_xlabel("t [ms]")
    ax[1].set_ylabel("I_stim [uA/cm^2]")
    ax[2].plot(t[:12500], I_stim[:12500])
    ax[2].set_xlabel("t [ms]")
    ax[2].set_ylabel("I_stim [uA/cm^2] (zoom)")
    plt.grid()
    fig.tight_layout(pad=2.0)
    plt.savefig(f"data_and_results/plotting/PRM_T_{T}_n_I_{n_I}.png")
    plt.close()