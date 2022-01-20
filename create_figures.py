import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors
from mpl_toolkits import mplot3d
import os
import json

input_files = os.listdir('./processed_output')
sorted_files = sorted(input_files, reverse=True)
selected_data = f'./processed_output/{sorted_files[0]}'
viridis = cm.get_cmap('viridis')

with open(os.path.join(selected_data, 'extra_data.json'), encoding='utf8') as f:
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
ax_3.set_ylim([1.2, 1.8])
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


figure_6 = plt.figure(figsize=plt.figaspect(1), dpi=100)
cmap = plt.get_cmap('viridis', len(etas))
log_etas = list(map(lambda x: np.log10(float(x)), etas))
norm = colors.Normalize(vmax=max(log_etas), vmin=min(log_etas))

ax00 = figure_6.add_subplot(2, 2, 1)
ax01 = figure_6.add_subplot(2, 2, 2)
ax10 = figure_6.add_subplot(2, 2, 3)
ax11 = figure_6.add_subplot(2, 2, 4, projection='3d')

ax00.set_xlabel(r'$\lambda$')
ax00.set_ylabel(r'$\sigma$')
ax00.set_xlim([-0.5, 0])
ax00.set_ylim([0.04, 0.24])

ax01.set_xlabel(r'$\sigma$')
ax01.set_ylabel(r'$x$')
ax01.set_xlim([0.05, 0.25])
ax01.set_ylim([-0.6, 0])

ax10.set_xlabel(r'$x$')
ax10.set_ylabel(r'$\lambda$')
ax10.set_xlim([-0.7, 0])
ax10.set_ylim([-0.9, 0])
ax11.set_xlabel(r'$x$')
ax11.set_ylabel(r'$\lambda$')
ax11.set_zlabel(r'$\sigma$')
ax11.set_xlim([-0.8, 0])
ax11.set_ylim([-1, 0])
ax11.set_zlim([0, 0.25])
ax11.xaxis.set_pane_color((0, 0, 0, 0))
ax11.yaxis.set_pane_color((0, 0, 0, 0))
ax11.zaxis.set_pane_color((0, 0, 0, 0))
ax11.grid(False)
ax11.azim = -45
ax11.elev = 22.5

figure_6.set_figwidth(9.6)
figure_6.set_figheight(7.2)

for eta, current_data in data.items():
    eta_log = np.log10(float(eta))
    color = cmap(norm(eta_log))
    np_data = np.array(current_data['raw'])
    ax00.plot(np_data[:, 0], np.divide(1., np_data[:, 2]), color=color)
    ax01.plot(np.divide(1., np_data[:, 2]), np_data[:, 1], color=color)
    ax10.plot(np_data[:, 1], np_data[:, 0], color=color)
    ax11.plot(np_data[:, 1], np_data[:, 0], np.divide(1., np_data[:, 2]), color=color)

figure_6.savefig('figure_6.svg')

plt.show()
print('Done')
