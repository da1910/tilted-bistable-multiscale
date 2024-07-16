import copy

import numpy as np
from matplotlib import pyplot as plt

RANDOM_SEEDINT = 131071


def df(x: float, alpha: float, pe: float, eta: float) -> float:
    tpe_x = 2. * pe * x
    return x*(x*(x - pe * np.cos(tpe_x)) + alpha - 2 * np.sin(tpe_x)) + eta

def get_final_position(run_id: int, noise_prefactor: float):
    np.random.seed(run_id * RANDOM_SEEDINT)
    x_init = np.random.uniform(low=-2., high=2., size=run_width)
    t_step = 0.
    x = copy.copy(x_init)
    while t_step <= t_end:
        x = x + curried_fun(x) * dt + noise_prefactor * np.random.normal(loc=0, scale=sqrt_dt, size=np.size(x_init))
        t_step += dt
    return x_init, x

alpha = -1.
beta = 20.
epsil = 0.01
eta = 0.1
noise_prefactor = np.sqrt(2. / beta)

pi_over_epsil = np.pi / epsil

dt = 1e-5
sqrt_dt = np.sqrt(dt)

n_runs = 4
run_width = 100
num_runs = 10

curried_fun = lambda x: df(x, alpha=alpha, pe=pi_over_epsil, eta=eta)

run_lengths = []
corr_coefs = []

for log_t_end in range(6, 8):
    t_end = 2 ** log_t_end
    print(f"Running with t < {t_end}")
    total_data = np.empty((run_width * num_runs, 2))
    for run_id in range (0, num_runs):
        print(f"\tsample set {run_id}")
        starting_values, data = get_final_position(run_id, noise_prefactor=noise_prefactor)
        total_data[run_id*run_width:(run_id+1)*run_width, :] = np.vstack((starting_values, data)).T
    total_data = total_data[~np.isnan(total_data[:,1])]
    run_lengths.append(t_end)
    corr_coefs.append(np.corrcoef(total_data))

fig, ax = plt.subplots()
ax.loglog(run_lengths, corr_coefs)
plt.show()