import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
import os
import json

input_files = os.listdir('./processed_output')
sorted_files = sorted(input_files, reverse=True)
selected_data = './processed_output/{}'.format(sorted_files[0])
viridis = cm.get_cmap('viridis')

with open(os.path.join(selected_data, 'results.json'), encoding='utf8') as f:
    data = json.load(f)

with open(os.path.join(selected_data, 'crit_data.json'), encoding='utf8') as f:
    crit_data = json.load(f)

eta_dict = {value: np.log10(float(value)) for value in data.keys()}
min_eta = min(eta_dict.values())
eta_range = max(eta_dict.values()) - min_eta

for k, v in eta_dict.items():
    eta_dict[k] = (v - min_eta)/eta_range

figure_1, ax_1 = plt.subplots()
ax_1.set_xlabel(r"$\log\left(\sigma_{c}-\sigma\right)/\sigma_{c}$")
ax_1.set_ylabel(r"$\log\left|\alpha_{c}\right|/N$")

for eta, current_data in data.items():
    current_approach = current_data['approach']
    current_d_lambda = current_data['d_lambda']
    ax_1.scatter(current_approach, current_d_lambda,
                 40, np.tile(viridis(eta_dict[eta]), (100, 1)), marker='o')
    ax_1.set_xscale('log')
    ax_1.set_yscale('log')

figure_1.show()
figure_1.savefig('figure_1.svg')

figure_2, ax_2 = plt.subplots()
ax_2.set_xlabel(r'$\left(\sigma_{c}-\sigma\right)/\sigma_{c}$')
ax_2.set_ylabel(r'$\left|\alpha_{c}\right|/N$')

for eta, current_data in data.items():
    current_approach = current_data['approach']
    current_d_lambda = current_data['d_lambda']
    ax_2.scatter(current_approach, current_d_lambda,
                 9, np.tile(viridis(eta_dict[eta]), (100, 1)))
figure_2.savefig('figure_2.svg')

figure_3, ax_3 = plt.subplots()
etas = []
fits = []
for eta, current_data in data.items():
    fitted_polynomial = current_data['fit']
    etas.append(eta)
    fits.append(fitted_polynomial)
ax_3.scatter([float(eta) for eta in etas], [x[0] for x in fits], 36, [viridis(eta_dict[eta]) for eta in etas])
ax_3.set_xlabel(r'$\eta$')
ax_3.set_ylabel(r'$\gamma - Exponent in critical approach$')
figure_3.savefig('figure_3.svg')


crit_data = np.array(crit_data)

figure_4, ax_4 = plt.subplots()
ax_4.scatter([float(eta) for eta in etas], np.divide(1., crit_data[:, 2]), 25)
ax_4.set_xlabel(r'$\eta$')
ax_4.set_ylabel(r'$\sigma_{c} - Critical \sigma value$')
figure_4.savefig('figure_4.svg')

figure_5, ax_5 = plt.subplots()
series = ax_5.scatter(crit_data[:, 0], crit_data[:, 1], 36, np.log10([float(eta) for eta in etas]))
ax_5.set_xlabel(r'$\lambda_{c}$')
ax_5.set_ylabel(r'$x_{c}$')
c = plt.colorbar(series, ax=ax_5)
c.set_label(r'$\log \eta$')
figure_5.savefig('figure_5.svg')

plt.show()
print('Done')