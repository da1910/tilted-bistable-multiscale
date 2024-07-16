import copy
import datetime
import json
import math
import multiprocessing
import os
from typing import Dict, List

import numpy as np
from numpy.random import normal, uniform
import numexpr as ne

RANDOM_SEEDINT = 131263
n_bins = 160
bin_edges = np.linspace(-2.0, 2.0, n_bins + 1)

def get_histogram(data: np.ndarray, bin_edges: np.ndarray):
    bin_index = 0
    num_bins = np.size(bin_edges) - 1
    bin_right = bin_edges[bin_index + 1]
    data_index = 0
    bin_values = np.zeros(shape=num_bins)
    num_values = np.size(data)
    while data_index < num_values:
        current_data = data[data_index]
        if np.isnan(current_data):
            break
        elif current_data <= bin_right:
            bin_values[bin_index] += 1
        else:
            while bin_index < num_bins - 1:
                bin_index += 1
                bin_right = bin_edges[bin_index + 1]
                if bin_right > current_data:
                    bin_values[bin_index] += 1
                    break
            if bin_index == num_bins - 1:
                bin_right = np.infty
        data_index += 1
    return bin_values

def get_distributions(run_id: int, noise_prefactor: float, alpha: float, epsil: float, eta: float, run_width: int, t_end: float, t_threshold: float, dt: float):
    print(f"{datetime.datetime.now().isoformat()} - {os.getpid()} - Starting run {run_id}")
    sqrt_dt = math.sqrt(dt)
    np.random.seed(run_id * RANDOM_SEEDINT)
    # x_init = uniform(low=-1., high=1., size=run_width)
    x_init = np.zeros(run_width)
    pi_over_epsil = np.pi / epsil
    t_step = 0.
    t_report = 0.01
    step_no = 0
    x = copy.copy(x_init)
    xs = np.zeros((run_width, int((t_end - t_threshold)/dt) + 1))
    while t_step < t_threshold:
        if t_step > t_report:
            print(t_report)
            t_report += 0.01
        tpe_x = 2. * pi_over_epsil * x
        noise_term = noise_prefactor * normal(loc=0, scale=sqrt_dt, size=np.size(x_init))
        x = ne.evaluate("x + (x*(x*(x - pi_over_epsil * cos(tpe_x)) - alpha - sin(tpe_x)) + eta) * dt + noise_term")
        t_step += dt
    print(f"{datetime.datetime.now().isoformat()} - {os.getpid()} - Reached threshold time: {t_threshold}")
    if any(np.isnan(x)):
        print(f"{datetime.datetime.now().isoformat()} - {os.getpid()} - Some nans: {sum(np.isnan(x))}")
    while t_step <= t_end:
        if t_step > t_report:
            print(t_report)
            t_report += 0.01
        tpe_x = 2. * pi_over_epsil * x
        noise_term = noise_prefactor * normal(loc=0, scale=sqrt_dt, size=np.size(x_init))
        x = ne.evaluate("x + (x*(x*(x - pi_over_epsil * cos(tpe_x)) - alpha - sin(tpe_x)) + eta) * dt + noise_term")
        t_step += dt
        xs[:, step_no] = x
        step_no += 1
    print(f"{datetime.datetime.now().isoformat()} - {os.getpid()} - Finished run {run_id}")
    print(f"{datetime.datetime.now().isoformat()} - {os.getpid()} - Computing counts")
    flattened_data = np.ravel(xs)
    flattened_data.sort()
    nan_mask = np.isnan(flattened_data)
    if not any(nan_mask):
        data = flattened_data
    else:
        index = np.where(nan_mask)[0][0]
        data = flattened_data[:index]
    hist = np.histogram(data, bin_edges)[0]
    print(f"{datetime.datetime.now().isoformat()} - {os.getpid()} - Done")
    return hist


def run_simulation(setup):
    epsil = 0.1
    noise_prefactor = np.sqrt(2. / setup["beta"])
    t_threshold = 10.

    ne.use_vml = True
    ne.set_vml_num_threads(8)

    t_end = 10010.
    dt = 5e-5

    n_runs = 5
    run_width = 20

    data = np.zeros(n_bins)
    for run in range(0, n_runs):
        print(f"Run number {run}")
        data = data + get_distributions(run_id=run, noise_prefactor=noise_prefactor, alpha=setup["alpha"], epsil=epsil, eta=setup["eta"], run_width=run_width, t_end=t_end, t_threshold=t_threshold, dt=dt)
    output = copy.copy(setup)
    output["bin_edges"] = bin_edges.tolist()
    output["counts"] = data.tolist()
    return output

run_data = [
    {
        "alpha": -0.5,
        "beta": 10.,
        "eta": 0.0
    },
    {
        "alpha": -0.25,
        "beta": 10.,
        "eta": 0.0
    },
    {
        "alpha": 0.25,
        "beta": 10.,
        "eta": 0.0
    },
    # {
    #     "alpha": -0.5,
    #     "beta": 10.,
    #     "eta": 0.1
    # },
    # {
    #     "alpha": -0.25,
    #     "beta": 10.,
    #     "eta": 0.1
    # },
    # {
    #     "alpha": 0.25,
    #     "beta": 10.,
    #     "eta": 0.1
    # }
]  # type: List[Dict[str, float]]

if __name__ == '__main__':
    results_queue = multiprocessing.Queue()
    with multiprocessing.Pool(processes=3) as pool:
        results = pool.map(run_simulation, run_data)

    with open("./pdf_data_mkl_1e-1_20.json", "w", encoding="utf8") as fp:
        json.dump(results, fp)
