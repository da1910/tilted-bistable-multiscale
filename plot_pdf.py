import json

from matplotlib import pyplot as plt
import numpy as np

with open("samples_a_-1.json", "r", encoding="utf-8") as fp:
    data = json.load(fp)

data = np.array(data)
data = data[~np.isnan(data)]

counts, bins = np.histogram(data, 800, range=(-1.5, 1.5), density=True)

fig, ax = plt.subplots()
ax.stairs(counts, bins)
plt.show()