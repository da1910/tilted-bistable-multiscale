import copy
import json
from typing import Callable, Dict, List

import numpy as np
from numpy.random import normal, uniform

RANDOM_SEEDINT = 131263
n_bins = 1000
bin_edges = np.linspace(-1.6, 1.6, n_bins + 1)


def df(x: float, alpha: float, pe: float, eta: float) -> float:
    tpe_x = 2. * pe * x
    return x*(x*(x - pe * np.cos(tpe_x)) + alpha - 2 * np.sin(tpe_x)) + eta


def get_distributions(run_id: int, noise_prefactor: float, dfx: Callable):
    print(f"Starting run {run_id}")
    np.random.seed(run_id * RANDOM_SEEDINT)
    x_init = uniform(low=-2., high=2., size=run_width)
    t_step = 0.
    step_no = 0
    x = copy.copy(x_init)
    xs = np.zeros((run_width, int((t_end - t_threshold)/dt)))
    while t_step < t_threshold:
        x = x + dfx(x) * dt + noise_prefactor * normal(loc=0, scale=sqrt_dt, size=np.size(x_init))
        t_step += dt
    print(f"Reached threshold time: {t_threshold}")
    while t_step <= t_end:
        x = x + dfx(x) * dt + noise_prefactor * normal(loc=0, scale=sqrt_dt, size=np.size(x_init))
        t_step += dt
        xs[:, step_no] = x
        step_no += 1
    print(f"Finished run {run_id}")
    print("Computing counts")
    flattened_data = xs.flatten()
    filtered_data = flattened_data[~np.isnan(flattened_data)]
    return np.histogram(filtered_data, bin_edges)[0]

run_data = [
    {
        "alpha": -1.0,
        "beta": 20.,
        "eta": 0.1
    },
    {
        "alpha": 1.0,
        "beta": 20.,
        "eta": 0.1
    },
    {
        "alpha": -0.8,
        "beta": 20.,
        "eta": 0.1
    },
    {
        "alpha": -1.0,
        "beta": 4.,
        "eta": 0.0
    },
    {
        "alpha": 1.0,
        "beta": 4.,
        "eta": 0.0
    },
    {
        "alpha": -0.4,
        "beta": 4.,
        "eta": 0.0
    },
]  # type: List[Dict[str, float]]

output_data = []

for setup in run_data:
    epsil = 0.01
    noise_prefactor = np.sqrt(2. / setup["beta"])
    t_threshold = 10

    pi_over_epsil = np.pi / epsil

    t_end = 100
    dt = 1e-5
    sqrt_dt = np.sqrt(dt)

    n_runs = 4
    run_width = 100

    curried_fun = lambda x: df(x, alpha=setup["alpha"], pe=pi_over_epsil, eta=setup["eta"])
    data = np.zeros(n_bins)
    for run in range(0, n_runs):
        print(f"Run number {run}")
        data = data + get_distributions(run_id=run, noise_prefactor=noise_prefactor, dfx=curried_fun)
    output = copy.copy(setup)
    output["bin_edges"] = bin_edges.tolist()
    output["counts"] = data.tolist()
    output_data.append(output)

with open("./pdf_data_16.json", "w", encoding="utf8") as fp:
    json.dump(output_data, fp)