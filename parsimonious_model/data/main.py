import os
import sys

from multiprocessing import Pool
import numpy as np
import json
import timeit

# Add path to the main project folder where data_generation is located
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from data_generation.parsimonious_model import parsimonious_model as PRM


# Wrapper function for multiprocessing
def run_pr_model(params):
    """
    Wrapper for running the parsimonious model.

    Args:
        params (list): Model parameters
                      [stim_times, stim_durations, stim_amplitudes, duration]

    Returns:
        tuple: Time, voltage, gating variables (m, h), ionic currents 
               (I_Na, I_K), and stimulus current (I_stim)
    """
    stim_times, stim_durations, stim_amplitudes, total_duration = params

    # print(
    #     (
    #     f"Running PR model with stim_times={stim_times}, "
    #      f"stim_durations={stim_durations}, "
    #      f"stim_amplitudes={stim_amplitudes}, "
    #      f"T={total_duration}"
    #      )
    # )
    return PRM(
        t_stim=stim_times,
        d_stim=stim_durations,
        a_stim=stim_amplitudes,
        T=total_duration
    )


def generate_unique_params(rng, bounds, n_data, n_stim, total_duration,
                           apd_min, apd_max, used_params_set, physical):
    """
    Generate unique stimulation parameters within the given constraints.

    Args:
        rng (Generator): Random number generator instance.
        bounds (dict): Boundaries for amplitude and APD values.
        n_data (int): Number of parameter sets to generate.
        n_stim (int): Number of stimulations.
        total_duration (int): Total simulation duration.
        apd_min (int): Minimum allowable APD.
        apd_max (int): Maximum allowable APD.
        used_params_set (set): Set of already used parameter tuples.
        physical (bool): Boolean wether to generate only pyhsical ('True') 
                         or also unphysical ('False') data.

    Returns:
        list: List of unique parameter sets.
    """
    unique_params = []
    for _ in range(n_data):
        while True:
            stim_durations = np.round(rng.uniform(0.5, 4.5, n_stim), 1)
            stim_amplitudes = []

            for duration in stim_durations:
                key = f"{duration:.2f}"
                amp_values, apd_values = map(np.array, bounds[key])
                valid_indices = (apd_values >= apd_min) & (apd_values <= apd_max)
                if not np.any(valid_indices):
                    break
                amp_range = amp_values[valid_indices]
                if physical:
                    stim_amplitudes.append(
                        rng.uniform(amp_range.min(), amp_range.max())
                    )
                else:
                    stim_amplitudes.append(
                        rng.uniform(amp_range.min(), 0)
                    )

            if len(stim_amplitudes) == n_stim:
                stim_tuple = (tuple(stim_durations), tuple(stim_amplitudes))
                if stim_tuple not in used_params_set:
                    used_params_set.add(stim_tuple)
                    stim_times = list(
                        np.arange(15, 15 + n_stim * apd_max, apd_max + 2)
                    )
                    unique_params.append(
                        [stim_times, list(stim_durations), list(stim_amplitudes),
                         total_duration]
                    )
                    break

    return unique_params


def save_generated_data(file_path, data, params):
    """
    Save generated simulation data to a compressed file.

    Args:
        file_path (str): Path to save the file.
        data (list): Simulation results.
        params (list): Parameter sets used for the simulations.
    """
    t, v, m, h, I_Na, I_K, I_stim = zip(*data)

    np.savez_compressed(
        file_path,
        t=np.array(t),
        v=np.array(v),
        m=np.array(m),
        h=np.array(h),
        I_Na=np.array(I_Na),
        I_K=np.array(I_K),
        I_stim=np.array(I_stim),
        params=np.array(params)
    )


def generate_pr_data(dataset_name, n_data, total_duration, n_stim, physical,
                     used_train_params=None):
    """
    Generate data using the parsimonious model and save it.

    Args:
        dataset_name (str): Name of the dataset (e.g., 'train' or 'test').
        n_data (int): Number of data samples to generate.
        total_duration (int): Total simulation duration in ms.
        n_stim (int): Number of stimulations.
        physical (bool): Boolean wether to generate only pyhsical ('True') 
                         or also unphysical ('False') data.
        used_train_params (list): Optional list of previously used 
                                  parameters to avoid duplication.

    Returns:
        list: List of used parameters for the dataset.
    """
    save_dir = "data_and_results/data_generation"
    os.makedirs(save_dir, exist_ok=True)
    if physical:
        save_file = (f"{dataset_name}_n_data_{n_data}_T_{total_duration}_"
                    f"n_stim_{n_stim}_physical.npz")
    else:
        save_file = (f"{dataset_name}_n_data_{n_data}_T_{total_duration}_"
                    f"n_stim_{n_stim}_unphysical.npz")
    file_path = os.path.join(save_dir, save_file)

    if os.path.exists(file_path):
        print(f"File {file_path} already exists.")
        data = np.load(file_path)
        return data["params"].tolist()

    with open("data_generation/Boundaries.json", "r") as file:
        bounds = json.load(file)

    apd_max = (total_duration - 30) // n_stim
    apd_min = 0
    rng = np.random.default_rng()
    # print(f"APD range: {apd_min} - {apd_max}")

    used_params_set = set(
        (tuple(p[1]), tuple(p[2]))  # Only include stim_durations and stim_amplitudes
        for p in (used_train_params or [])
    )

    params = generate_unique_params(
        rng, bounds, n_data, n_stim, total_duration, apd_min, apd_max, 
        used_params_set, physical
    )
    # print("PARAMS: ", params)

    num_cores = os.cpu_count()
    with Pool(num_cores) as pool:
            data = pool.map(run_pr_model, params)

    # Save the results
    stripped_params = [[t, d, a] for t, d, a, _ in params]  # Remove total_duration
    save_generated_data(file_path, data, stripped_params)
    return stripped_params


if __name__ == "__main__":
    # Simulation parameters
    n_data_train = 400
    n_data_val = 100
    n_data_test = 100
    total_duration = 400
    n_stim = 1
    physical = False

    # Generate training and test datasets
    start_time = timeit.default_timer()
    train_params = generate_pr_data("train", n_data_train, total_duration, n_stim, 
                                    physical)
    train_time = timeit.default_timer() - start_time
    print(f"Generating train data took {train_time:.2f}s.")

    start_time = timeit.default_timer()
    val_params = generate_pr_data("val", n_data_val, total_duration, n_stim, 
                                    physical, train_params)
    val_time = timeit.default_timer() - start_time
    print(f"Generating validation data took {val_time:.2f}s.")

    start_time = timeit.default_timer()
    test_params = generate_pr_data("test", n_data_test, total_duration, n_stim, 
                                   physical, train_params+val_params)
    test_time = timeit.default_timer() - start_time
    print(f"Generating test data took {test_time:.2f}s.")