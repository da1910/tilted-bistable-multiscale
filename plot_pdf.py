import json

from matplotlib import pyplot as plt
import numpy as np

with open("pdf_data_16.json", "r", encoding="utf-8") as fp:
    data = json.load(fp)

data = np.array(data, dtype="double")
data = data[~np.isnan(data)]

counts, bins = np.histogram(data, 800, range=(-1.5, 1.5), density=True)

fig, ax = plt.subplots()
ax.stairs(counts, bins)
plt.show()