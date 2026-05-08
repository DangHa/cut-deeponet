import os
import sys

import numpy as np
import json
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from multiprocessing import Pool

# Add path to the main project folder where data_generation is located
main_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(main_folder)

from data_generation.parsimonious_model import parsimonious_model as PRM
import data_generation.helper_functions as help

# Wrapper function for multiprocessing
def compute_and_save(params):
    """Compute and save PR model data."""
    # Parameters
    d_stim, a_stim = params

    # Filepath for saving data
    output_dir = "data_and_results/data_generation/boundary_data"
    os.makedirs(output_dir, exist_ok=True)
    filename = f"stim_d_{d_stim:.2f}_a_{a_stim:.2f}.npz"
    filepath = os.path.join(output_dir, filename)

    # Run the model and save the data
    t, v, _, _, _, _, _ = PRM(
        t_stim=[25], 
        d_stim=[d_stim], 
        a_stim=[a_stim],
        T=550
    )
    np.savez_compressed(filepath, 
                        t=t, 
                        v=v, 
                        d_stim=d_stim, 
                        a_stim=a_stim)
    return 


def generate_data_to_choose_a_stim():
    """Generate data to later choose range for a_stim for each d_stim."""
    # Parameters for testing
    d_stim_vals = np.arange(0, 5+0.1, 0.1)
    a_stim_vals = np.arange(-100, 0+0.5, 0.5)
    grid_d, grid_a = np.meshgrid(d_stim_vals, a_stim_vals, indexing="ij")
    pairs = np.column_stack([grid_d.ravel(), grid_a.ravel()])

    # Directory for storing data
    output_dir = "data_and_results/data_generation/boundary_data"
    os.makedirs(output_dir, exist_ok=True) 

    # Create a multiprocessing Pool
    num_cores = os.cpu_count()
    if num_cores > 2:
        with Pool(num_cores-2) as pool:
            # Filter pairs to exclude already existing files
            params_to_compute = [
                pair for pair in pairs
                if not os.path.exists(
                    os.path.join(
                        output_dir, 
                        f"stim_d_{pair[0]:.2f}_a_{pair[1]:.2f}.npz"
                    )
                )
            ]

            # Only process pairs that don't already have data
            pool.map(compute_and_save, params_to_compute)
    else:
        with Pool(num_cores) as pool:
            # Filter pairs to exclude already existing files
            params_to_compute = [
                pair for pair in pairs
                if not os.path.exists(
                    os.path.join(
                        output_dir, 
                        f"stim_d_{pair[0]:.2f}_a_{pair[1]:.2f}.npz"
                    )
                )
            ]

            # Only process pairs that don't already have data
            pool.map(compute_and_save, params_to_compute)

    print(f"Simulation completed.")


def calculate_valid_a_stim_and_APD():
    """Calculate valid a_stim values and their APD for each d_stim.
    
    For each d_stim value (range [0, 5], stepsize=0.1) look at all 
    a_stim values (range [-100, 0], stepsize=0.5) and check, if v peaks
    over 0. If so calculate AP duration by extracting the two indices
    where v first raises above -82 and then falls back below and 
    subtracting the corresponding times. Also check, if APD decreases 
    with increasing a_stim (which is negativ). Save all valid a_stim 
    values with their corresponding APD in a .json file.
    """
    # Directory containing the data
    data_dir = "data_and_results/data_generation/boundary_data"

    # Threshold
    v_th = -82

    # Results
    results = {}

    # Loop through files
    for d_stim in tqdm(np.arange(0.00, 5.10, 0.10)): 

        # Lists for valid a_stim and APD values
        a_stim_vals = []
        APD_vals = []
        
        for a_stim in np.arange(-100.00, 0.50, 0.50):
            # File path
            file_path = os.path.join(
                data_dir, 
                f"stim_d_{d_stim:.2f}_a_{a_stim:.2f}.npz"
            )
            if not os.path.exists(file_path):
                print(
                    f"File for d={d_stim:.2f} and a={a_stim:.2f} "
                    f"does not exists."
                )
                continue
            
            # Load data
            data = np.load(file_path)
            t, v = data["t"], data["v"]

            # Check if actual AP exists
            if np.max(v) > 0:
                # Start of AP
                above_v_th_inds = np.where(v > v_th)[0]
                start_ind = above_v_th_inds[0]
                t_start = t[start_ind]

                # End of AP
                below_v_th_inds = np.where(v[start_ind:] < v_th)[0]
                if len(below_v_th_inds) > 0:
                    end_ind = below_v_th_inds[0] + start_ind
                    t_end = t[end_ind]

                    # APD
                    APD = t_end - t_start
                    a_stim_vals.append(a_stim)
                    APD_vals.append(APD[0])

                    # Make sure APD values decrease with increasing a_stim
                    # Small tolerance of 1ms due to numerical calculations
                    if len(APD_vals) > 1 and APD_vals[-2]+1 < APD_vals[-1]:
                        print(
                            "Wrong order in APD\n"
                            f"Previous: a={a_stim_vals[-2]} with " 
                            f"APD={APD_vals[-2]:.2f}\n"
                            f"Now: a={a_stim_vals[-1]} with "
                            f"APD={APD_vals[-1]:.2f}"
                        )

        # Check for valid a_stim values
        if len(a_stim_vals) > 0:
            key = f"{d_stim:.2f}"
            results[key] = [a_stim_vals, APD_vals]

    # Save data
    with open("data_generation/Boundaries.json", "w") as f:
        json.dump(results, f)

def plot_APD_dependence():
    """Plot APD against a_stim for different d_stim."""
    # Check / Create saving directory
    save_dir = "data_and_results/plotting"
    os.makedirs(save_dir, exist_ok=True) 

    # Load dictionary for a_stim ranges
    with open("data_generation/Boundaries.json", "r") as file:
        bounds = json.load(file)

    # Different d_stim values
    d_stim = np.arange(0.5, 4.5+0.5, 0.5)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    for d in d_stim:
        key = f"{d:.2f}"
        data = bounds[key]
        a_stim, APD = np.array(data[0]), np.array(data[1])
        ax.plot(
            a_stim, APD, 
            marker="o", markersize=2, linewidth=0.2, 
            label=f"d_stim = {d:.1f}"
        )
    ax.set_xlabel("a_stim [uA/cm^2]")
    ax.set_ylabel("APD [ms]")
    ax.title.set_text("APD as a function of a_stim for different d_stim")
    plt.grid()
    plt.legend()
    fig.tight_layout(pad=2.0)
    plt.savefig(f"data_and_results/plotting/APD_dependence.png")
    plt.close()


# Generate n_I pulses in time T
def main(T, n_I):
    """Generate data up to time T with n_I pulses."""
    # Load dictionary for a_stim ranges
    with open("data_generation/Boundaries.json", "r") as file:
        bounds = json.load(file)

    # Set up t_stim
    APD_max = (T-30) // n_I # Upper bound for each APD
    APD_min = 200           # Lower bound for each APD
    t_stim = list(np.arange(15, 15 + n_I * APD_max, APD_max))

    # Generate d_stim values with one decimal
    rng = np.random.default_rng() 
    d_stim = np.round(rng.uniform(0.5, 4.5, n_I), 1)

    # Generate a_stim values
    a_stim = []
    for d in d_stim:
        key = f"{d:.2f}"
        data = bounds[key]
        a_stim_, APD = np.array(data[0]), np.array(data[1])
        # Filter out bounds for a_stim values 
        # such that APD lies in above defined range
        mask = (APD >= APD_min) & (APD <= APD_max)
        a_stim.append(rng.uniform(
            np.min(a_stim_[mask]), 
            np.max(a_stim_[mask]), 
            1))

    t, v, _, _, _, _, I_stim = PRM(t_stim, list(d_stim), a_stim, T)
    help.plotting_impluse_action_potential(t, v, I_stim, T, n_I)


if __name__ == "__main__":
    generate_data_to_choose_a_stim()
    calculate_valid_a_stim_and_APD()
    plot_APD_dependence()

    configurations = [
        [6000, 15],
        [6000, 10],
        [3000, 10],
        [3000, 5],
        [1000, 4],
        [1000, 3],
        [1000, 2]
    ]

    for config in tqdm(configurations):
        main(config[0], config[1])