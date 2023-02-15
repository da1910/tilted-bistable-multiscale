import copy
import json
import multiprocessing
from functools import partial

import numpy as np
from numpy.random import normal, uniform
import matplotlib.pyplot as plt

RANDOM_SEEDINT = 131263

def df(x: float, alpha: float, pe: float, eta: float) -> float:
    tpe_x = 2. * pe * x
    return x*(x*(x - pe * np.cos(tpe_x)) + alpha - 2 * np.sin(tpe_x)) + eta

def get_final_position(run_id: int, noise_prefactor: float):
    print(f"Starting run {run_id}")
    np.random.seed(run_id * RANDOM_SEEDINT)
    x_init = uniform(low=-2., high=2., size=run_width)
    t_step = 0.
    step_no = 0
    x = copy.copy(x_init)
    xs = np.zeros((run_width, int((t_end - t_threshold)/dt)))
    while t_step < t_threshold:
        x = x + curried_fun(x) * dt + noise_prefactor * normal(loc=0, scale=sqrt_dt, size=np.size(x_init))
        t_step += dt
    print(f"Reached threshold time: {t_threshold}")
    while t_step <= t_end:
        x = x + curried_fun(x) * dt + noise_prefactor * normal(loc=0, scale=sqrt_dt, size=np.size(x_init))
        t_step += dt
        xs[:, step_no] = x
        step_no += 1
    print(f"Finished run {run_id}")
    return xs

alpha = -1.
beta = 20.
epsil = 0.01
eta = 0.1
noise_prefactor = np.sqrt(2. / beta)
t_threshold = 50

pi_over_epsil = np.pi / epsil

# sample after a threshold time rather than just the final time
# try with tilt = 0
t_end = 250
dt = 1e-4
sqrt_dt = np.sqrt(dt)

n_runs = 4
run_width = 100

curried_fun = lambda x: df(x, alpha=alpha, pe=pi_over_epsil, eta=eta)
data = []
for run in range(0, n_runs):
    print(f"Run number {run}")
    data.append(get_final_position(run_id=run, noise_prefactor=noise_prefactor))

combined_data = np.hstack(data).flatten()
filtered_data = combined_data[~np.isnan(combined_data)]


counts, edges = np.histogram(filtered_data, np.linspace(-1.0, 1.0, 1001))
plt.stairs(counts, edges)
plt.show()