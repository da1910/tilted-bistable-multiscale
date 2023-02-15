import time

import datetime
import numpy as np
from matplotlib import pyplot as plt


def generate_random(n_samples) -> float:
    np.random.seed(datetime.datetime.now().microsecond)
    start = time.time()
    _ = np.random.uniform(low=-2., high=2., size=n_samples)
    end = time.time()
    return end - start

sample_counts = []
average_times = []
for n in range(0, 24):
    sample_count = 2 ** n
    print(f"Running with {sample_count} samples per invocation...")
    length_results = []
    for run_num in range(0, 100):
        length_results.append(generate_random(sample_count))
    sample_counts.append(sample_count)
    average_times.append(np.mean(length_results))

fig, ax = plt.subplots()
ax.loglog(sample_counts, average_times)

plt.show()