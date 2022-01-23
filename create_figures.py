from typing import Dict, Tuple, List

import matplotlib.axes
import matplotlib.figure
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib import colors
from mpl_toolkits import mplot3d
from mpl_toolkits.axes_grid.inset_locator import inset_axes, InsetPosition, mark_inset
import os
import json


def generate_figure_one(axis: matplotlib.axes.Axes, data: Dict, eta_dict: Dict) -> None:
    axis.set_xlabel(r"$\log\left(\sigma_{c}-\sigma\right)/\sigma_{c}$")
    axis.set_ylabel(r"$\log\left|\alpha_{c}\right|/N$")

    for eta, current_data in data.items():
        current_approach = current_data['approach']
        current_d_lambda = current_data['d_lambda']
        axis.scatter(current_approach, current_d_lambda,
                     40, np.tile(viridis(eta_dict[eta]), (100, 1)), marker='o')
        axis.set_xscale('log')
        axis.set_yscale('log')


def generate_figure_two(axis: matplotlib.axes.Axes, data: Dict, eta_dict: Dict) -> None:
    axis.set_xlabel(r'$\left(\sigma_{c}-\sigma\right)/\sigma_{c}$')
    axis.set_ylabel(r'$\left|\alpha_{c}\right|/N$')

    for eta, current_data in data.items():
        current_approach = current_data['approach']
        current_d_lambda = current_data['d_lambda']
        axis.scatter(current_approach, current_d_lambda,
                     9, np.tile(viridis(eta_dict[eta]), (100, 1)))


def generate_figure_three(axis: matplotlib.axes.Axes, etas: List[float], fits: List[Tuple[float, float]], eta_dict: Dict) -> None:
    axis.scatter([float(eta) for eta in etas], [x[0] for x in fits], 36, [viridis(eta_dict[eta]) for eta in etas])
    axis.set_ylim(bottom=1.2, top=1.8)
    axis.set_xlabel(r'$\eta$')
    axis.set_ylabel(r'$\gamma - Exponent in critical approach$')


def generate_figure_four(axis: matplotlib.axes.Axes, etas: List[float], crit_data: np.ndarray) -> None:
    axis.scatter([float(eta) for eta in etas], np.divide(1., crit_data[:, 2]), 25)
    axis.set_xlabel(r'$\eta$')
    axis.set_ylabel(r'$\sigma_{c} - Critical \sigma value$')


def generate_figure_five(axis: matplotlib.axes.Axes, etas: List[float], crit_data: np.ndarray) -> None:
    series = ax_5.scatter(crit_data[:, 0], crit_data[:, 1], 36, np.log10([float(eta) for eta in etas]))
    axis.set_xlabel(r'$\lambda_{c}$')
    axis.set_ylabel(r'$x_{c}$')
    c = plt.colorbar(series, ax=axis)
    c.set_label(r'$\log \eta$')


def generate_figure_six(fig: matplotlib.figure.Figure, etas: List[float], data: Dict) -> None:
    cmap = plt.get_cmap('viridis', len(etas))
    log_etas = list(map(lambda x: np.log10(float(x)), etas))
    norm = colors.Normalize(vmax=max(log_etas), vmin=min(log_etas))

    ax00 = fig.add_subplot(2, 2, 1)
    ax01 = fig.add_subplot(2, 2, 2)
    ax10 = fig.add_subplot(2, 2, 3)
    ax11 = fig.add_subplot(2, 2, 4, projection='3d')

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

    fig.set_figwidth(9.6)
    fig.set_figheight(7.2)

    for eta, current_data in data.items():
        eta_log = np.log10(float(eta))
        color = cmap(norm(eta_log))
        np_data = np.array(current_data['raw'])
        ax00.plot(np_data[:, 0], np.divide(1., np_data[:, 2]), color=color)
        ax01.plot(np.divide(1., np_data[:, 2]), np_data[:, 1], color=color)
        ax10.plot(np_data[:, 1], np_data[:, 0], color=color)
        ax11.plot(np_data[:, 1], np_data[:, 0], np.divide(1., np_data[:, 2]), color=color)


def compute_critical_approach_fit(data: Dict) -> Tuple[List[float], List[Tuple[float, float]]]:
    etas = []
    fits = []
    for eta, current_data in data.items():
        fitted_polynomial = current_data['fit']
        etas.append(eta)
        fits.append(fitted_polynomial)
    return etas, fits


input_files = os.listdir('./processed_output')
sorted_files = sorted(input_files, reverse=True)
selected_data = f'./processed_output/{sorted_files[0]}'
viridis = cm.get_cmap('viridis')

with open(os.path.join(selected_data, 'extra_data.json'), encoding='utf8') as f:
    data = json.load(f)

with open(os.path.join(selected_data, 'crit_data.json'), encoding='utf8') as f:
    crit_data = np.array(json.load(f))

eta_dict = {value: np.log10(float(value)) for value in data.keys()}
min_eta = min(eta_dict.values())
eta_range = max(eta_dict.values()) - min_eta

for k, v in eta_dict.items():
    eta_dict[k] = (v - min_eta)/eta_range

etas, fits = compute_critical_approach_fit(data)


figure_1, ax_1 = plt.subplots()
generate_figure_one(ax_1, data, eta_dict)
figure_1.show()
figure_1.savefig('figure_1.svg')

figure_2, ax_2 = plt.subplots()
generate_figure_one(ax_2, data, eta_dict)
figure_2.savefig('figure_2.svg')

figure_3, ax_3 = plt.subplots()
generate_figure_three(ax_3, etas, fits, eta_dict)
figure_3.savefig('figure_3.svg')

figure_4, ax_4 = plt.subplots()
generate_figure_four(ax_4, etas, crit_data)
figure_4.savefig('figure_4.svg')

figure_5, ax_5 = plt.subplots()
generate_figure_five(ax_5, etas, crit_data)
figure_5.savefig('figure_5.svg')

figure_6 = plt.figure(figsize=plt.figaspect(1), dpi=100)
generate_figure_six(figure_6, etas, data)
figure_6.savefig('figure_6.svg')

plt.show()
print('Done')
