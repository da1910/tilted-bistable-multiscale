from typing import Dict, Tuple, List, Union, Iterable

import matplotlib.axes
import matplotlib.figure
import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from matplotlib import colormaps as cm
from matplotlib import colors
import os
import json


def generate_figure_one(axis: matplotlib.axes.Axes, data: Dict, eta_dict: Dict) -> None:
    axis.set_xlabel(r"$\left(\sigma_{c}-\sigma\right)/\sigma_{c}$")
    axis.set_ylabel(r"$\left|\alpha_{c}\right|/N$")

    for eta, current_data in data.items():
        current_approach = current_data["approach"]
        current_d_lambda = current_data["d_lambda"]
        axis.scatter(
            current_approach,
            current_d_lambda,
            40,
            np.tile(viridis(eta_dict[eta]), (100, 1)),
            marker="o",
        )
        axis.set_xscale("log")
        axis.set_yscale("log")


def generate_figure_two(axis: matplotlib.axes.Axes, data: Dict, eta_dict: Dict) -> None:
    axis.set_xlabel(r"$\left(\sigma_{c}-\sigma\right)/\sigma_{c}$")
    axis.set_ylabel(r"$\left|\alpha_{c}\right|/N$")

    for eta, current_data in data.items():
        current_approach = current_data["approach"]
        current_d_lambda = current_data["d_lambda"]
        axis.scatter(
            current_approach,
            current_d_lambda,
            9,
            np.tile(viridis(eta_dict[eta]), (100, 1)),
        )
    inset_axis = axis.inset_axes([0.15, 0.6, 0.4, 0.3])
    generate_figure_one(inset_axis, data, eta_dict)


def generate_figure_three(
    axis: matplotlib.axes.Axes,
    etas: List[float],
    fits: List[Tuple[float, float]],
    eta_dict: Dict,
) -> None:
    axis.scatter(
        [float(eta) for eta in etas],
        [x[0] for x in fits],
        36,
        [viridis(eta_dict[eta]) for eta in etas],
    )
    axis.set_ylim(bottom=1.2, top=1.8)
    axis.set_xlabel(r"$\eta$")
    axis.set_ylabel(r"$\gamma$ - Exponent in critical approach")


def generate_figure_four(
    axis: matplotlib.axes.Axes, etas: List[float], crit_data: np.ndarray
) -> None:
    axis.scatter([float(eta) for eta in etas], np.divide(1.0, crit_data[:, 2]), 25)
    axis.set_xlabel(r"$\eta$")
    axis.set_ylabel(r"$\sigma_{c}$ - Critical $\sigma$ value")


def generate_figure_five(
    axis: matplotlib.axes.Axes, etas: List[float], crit_data: np.ndarray
) -> None:
    series = axis.scatter(
        crit_data[:, 0], crit_data[:, 1], 36, np.log10([float(eta) for eta in etas])
    )
    axis.set_xlabel(r"$\lambda_{c}$")
    axis.set_ylabel(r"$x_{c}$")
    c = plt.colorbar(series, ax=axis)
    c.set_label(r"$\log \eta$")


def generate_figure_six(
    fig: matplotlib.figure.Figure, etas: List[float], data: Dict
) -> None:
    cmap = plt.get_cmap("viridis", len(etas))
    log_etas = list(map(lambda x: np.log10(float(x)), etas))
    norm = colors.Normalize(vmax=max(log_etas), vmin=min(log_etas))

    ax00 = fig.add_subplot(2, 2, 1)
    ax01 = fig.add_subplot(2, 2, 2)
    ax10 = fig.add_subplot(2, 2, 3)
    ax11 = fig.add_subplot(2, 2, 4, projection="3d")

    ax00.set_xlabel(r"$\lambda$")
    ax00.set_ylabel(r"$\sigma$")
    ax00.set_xlim([-0.5, 0])
    ax00.set_ylim([0.04, 0.24])

    ax01.set_xlabel(r"$\sigma$")
    ax01.set_ylabel(r"$x$")
    ax01.set_xlim([0.05, 0.25])
    ax01.set_xticks(np.linspace(0.05, 0.25, 5))
    ax01.set_ylim([-0.6, 0])

    ax10.set_xlabel(r"$x$")
    ax10.set_ylabel(r"$\lambda$")
    ax10.set_xlim([-0.7, 0])
    ax10.set_ylim([-0.9, 0])
    ax11.set_xlabel(r"$x$")
    ax11.set_ylabel(r"$\lambda$")
    ax11.set_zlabel(r"$\sigma$")
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
        np_data = np.array(current_data["raw"])
        ax00.plot(np_data[:, 0], np.divide(1.0, np_data[:, 2]), color=color)
        ax01.plot(np.divide(1.0, np_data[:, 2]), np_data[:, 1], color=color)
        ax10.plot(np_data[:, 1], np_data[:, 0], color=color)
        ax11.plot(
            np_data[:, 1], np_data[:, 0], np.divide(1.0, np_data[:, 2]), color=color
        )


def generate_figure_seven(fig: matplotlib.figure.Figure, snapshot: List) -> None:
    fig.set_layout_engine("tight")

    ax00 = fig.add_subplot(1, 3, 1)
    ax01 = fig.add_subplot(1, 3, 2)
    ax10 = fig.add_subplot(1, 3, 3)

    ax00.set_xlabel(r"$\lambda$")
    ax00.set_ylabel(r"$x$")
    ax00.set_xlim([-2, 2])
    ax00.set_ylim([-2, 2])

    ax01.set_xlabel(r"$\lambda$")
    ax01.set_ylabel(r"$x$")
    ax01.set_xlim([-2, 2])
    ax01.set_ylim([-2, 2])

    ax10.set_xlabel(r"$\lambda$")
    ax10.set_ylabel(r"$x$")
    ax10.set_xlim([-2, 2])
    ax10.set_ylim([-2, 2])

    axs = [ax00, ax01, ax10]

    relevant_snapshots = [item for item in snapshot if item["eta"] > 0]
    betas = [40.0, 7.6, 1.0]
    sorted_data = []

    for beta in betas:
        sorted_data.append(next(item for item in relevant_snapshots if item["beta"] == beta))

    for index, ax in enumerate(axs):
        current_item = sorted_data[index]
        plot_snapshot(ax, current_item)
        ax.set_aspect('equal')

def plot_pdf(ax: matplotlib.axes.Axes, runs: Iterable[Dict]):
    bin_widths = {item["bin_widths"] for item in runs}
    sorted_bin_widths = sorted(bin_widths)

    for bin_width in sorted_bin_widths:
        pdf_data = next(run for run in runs if run["bin_widths"] == bin_width)
        n_samples = sum(pdf_data["counts"])
        bin_widths = np.array([x - y for x, y in zip(pdf_data["bin_edges"][1:], pdf_data["bin_edges"][:-1])])
        normalized_data = np.array(pdf_data["counts"]) / (n_samples * bin_widths)
        bin_centres = np.array([(x + y) / 2. for x, y in zip(pdf_data["bin_edges"][1:], pdf_data["bin_edges"][:-1])])
        ax.plot(bin_centres, normalized_data, label="Simulated PDF")

    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(0, 3)

def plot_homogenized_pdf(ax: matplotlib.axes.Axes, alpha: float, beta: float, eta: float):
    x_vals = np.linspace(-10, 10, 4001)
    i_0 = sp.special.i0(beta * x_vals ** 2 / 2)
    y_vals = np.exp(-beta*(x_vals ** 4 / 4 - (alpha * x_vals ** 2 / 2) + x_vals * eta)) * i_0

    area = np.trapz(y_vals, x_vals)

    ax.plot(x_vals, y_vals / area, linestyle=(0, (0.8, 0.8)), color="green", label="Homogenized PDF")

def generate_figure_eight(fig: matplotlib.figure.Figure, relevant_snapshot: Dict, pdf_data: Dict) -> None:
    fig.set_layout_engine(layout="tight")
    fig.set_size_inches(12, 5)
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1])

    ax_top = fig.add_subplot(gs[0, :])
    ax_top.set_aspect(0.5)
    plot_snapshot(ax_top, relevant_snapshot)
    ax_top.set_xlabel(r"$\lambda$")
    ax_top.set_ylabel(r"$x$")

    ax00 = fig.add_subplot(gs[1, 0])
    ax01 = fig.add_subplot(gs[1, 1])
    ax10 = fig.add_subplot(gs[1, 2])

    ax00.set_xlabel(r"$x$")
    ax00.set_ylabel(r"$f\left(x\right)$")
    ax00.set_xlim([-2, 2])
    ax00.set_ylim([-2, 2])

    ax01.set_xlabel(r"$x$")
    ax01.set_ylabel(r"$f\left(x\right)$")
    ax01.set_xlim([-2, 2])
    ax01.set_ylim([-2, 2])

    ax10.set_xlabel(r"$x$")
    ax10.set_ylabel(r"$f\left(x\right)$")
    ax10.set_xlim([-2, 2])
    ax10.set_ylim([-2, 2])

    tilted_data = [run for run in pdf_data if run["eta"] > 0]

    alphas = {value["alpha"] for value in tilted_data}
    sorted_alphas = sorted(alphas)
    axs = [ax00, ax01, ax10]

    for index, ax in enumerate(axs):
        current_data = [results for results in tilted_data if results["alpha"] == sorted_alphas[index]]
        plot_pdf(ax, current_data)
        alpha = current_data[0]["alpha"]
        beta = current_data[0]["beta"]
        eta = current_data[0]["eta"]
        plot_homogenized_pdf(ax, alpha, beta, eta)
        ax_top.axvline(alpha, color="k", linestyle="-.")

    lines, labels = axs[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc=(0.03, 0.4))

    ax_top.annotate("a.", xy=(-0.7, 1.7))
    ax_top.annotate("b.", xy=(-0.15, 1.7))
    ax_top.annotate("c.", xy=(0.6, 1.7))

    axs[0].annotate(r"a) $\lambda = -0.5$", xy=(0.5, 2.2), annotation_clip=False)
    axs[1].annotate(r"b) $\lambda = -0.25$", xy=(0.5, 2.2), annotation_clip=False)
    axs[2].annotate(r"c) $\lambda = 0.5$", xy=(0.5, 2.2), annotation_clip=False)


def generate_figure_eight_a(fig: matplotlib.figure.Figure, relevant_snapshot: Dict, pdf_data: Dict) -> None:
    fig.set_layout_engine(layout="tight")
    fig.set_size_inches(12, 5)
    gs = fig.add_gridspec(2, 3, height_ratios=[2, 1])

    ax_top = fig.add_subplot(gs[0, :])
    ax_top.set_aspect(0.5)
    plot_snapshot(ax_top, relevant_snapshot)

    unstable_branch = [[-0.7, 0], [0, 0]]
    ax_top.plot(*unstable_branch, color="black", linestyle="--", linewidth=2)
    ax_top.set_xlabel(r"$\lambda$")
    ax_top.set_ylabel(r"$x$")

    ax00 = fig.add_subplot(gs[1, 0])
    ax01 = fig.add_subplot(gs[1, 1])
    ax10 = fig.add_subplot(gs[1, 2])

    ax00.set_xlabel(r"$x$")
    ax00.set_ylabel(r"$f\left(x\right)$")
    ax00.set_xlim([-2, 2])
    ax00.set_ylim([-2, 2])

    ax01.set_xlabel(r"$x$")
    ax01.set_ylabel(r"$f\left(x\right)$")
    ax01.set_xlim([-2, 2])
    ax01.set_ylim([-2, 2])

    ax10.set_xlabel(r"$x$")
    ax10.set_ylabel(r"$f\left(x\right)$")
    ax10.set_xlim([-2, 2])
    ax10.set_ylim([-2, 2])

    tilted_data = [run for run in pdf_data if run["eta"] == 0]

    alphas = {value["alpha"] for value in tilted_data}
    sorted_alphas = sorted(alphas)
    axs = [ax00, ax01, ax10]

    for index, ax in enumerate(axs):
        current_data = [results for results in tilted_data if results["alpha"] == sorted_alphas[index]]
        plot_pdf(ax, current_data)
        alpha = current_data[0]["alpha"]
        beta = current_data[0]["beta"]
        eta = current_data[0]["eta"]
        plot_homogenized_pdf(ax, alpha, beta, eta)
        ax_top.axvline(alpha, color="k", linestyle="-.")

    lines, labels = axs[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc=(0.03, 0.4))

    ax_top.annotate("a.", xy=(-0.47, 1.7))
    ax_top.annotate("b.", xy=(-0.15, 1.7))
    ax_top.annotate("c.", xy=(0.6, 1.7))

    axs[0].annotate(r"a) $\lambda = -0.5$", xy=(0.5, 2.2), annotation_clip=False)
    axs[1].annotate(r"b) $\lambda = -0.25$", xy=(0.5, 2.2), annotation_clip=False)
    axs[2].annotate(r"c) $\lambda = 0.5$", xy=(0, 2.2), annotation_clip=False)

def plot_snapshot(ax: matplotlib.axes.Axes, data: Dict):
    b1x = np.array([row[0] for row in data["b1"]], dtype=float)
    b1y = np.array([row[2] for row in data["b1"]], dtype=float)
    db1 = np.gradient(b1y, b1x) > 0
    b2x = np.array([row[0] for row in data["b2"]], dtype=float)
    b2y = np.array([row[2] for row in data["b2"]], dtype=float)
    db2 = np.gradient(b2y, b2x) > 0
    b1_chunks = split_array_on_sign(b1x, b1y, db1)
    b2_chunks = split_array_on_sign(b2x, b2y, db2)

    for chunk in b1_chunks:
        color = "black"
        if not chunk["sign"]:
            line_style = "--"
        else:
            line_style = "-"
        ax.plot(chunk["data"][:, 0], chunk["data"][:, 1], color=color, linestyle=line_style)

    for chunk in b2_chunks:
        color = "black"
        if chunk["sign"]:
            line_style = "--"
        else:
            line_style = "-"
        ax.plot(chunk["data"][:, 0], chunk["data"][:, 1], color=color, linestyle=line_style)


def split_array_on_sign(xdata: np.ndarray, ydata: np.ndarray, flags: np.ndarray) -> List[Dict[str, Union[float, np.ndarray]]]:
    output = []
    current_chunk = []
    current_sign = None
    for xval, yval, sign in zip(xdata, ydata, flags):
        if current_sign is None:
            current_sign = sign
        current_chunk.append([xval, yval])
        if sign != current_sign:
            output.append({"sign": current_sign, "data": np.array(current_chunk, dtype=float)})
            current_sign = sign
            current_chunk = [[xval, yval]]
    output.append({"sign": current_sign, "data": np.array(current_chunk, dtype=float)})
    return output


def compute_critical_approach_fit(
    data: Dict,
) -> Tuple[List[float], List[Tuple[float, float]]]:
    etas = []
    fits = []
    for eta, current_data in data.items():
        fitted_polynomial = current_data["fit"]
        etas.append(eta)
        fits.append(fitted_polynomial)
    return etas, fits

plt.rcParams.update({
    "text.usetex": True,
})

input_files = os.listdir("./processed_output")
sorted_files = sorted(input_files, reverse=True)
main_data = f"./processed_output/{sorted_files[0]}"

input_files = os.listdir("./raw_output")
sorted_files = sorted(input_files, reverse=True)
file_set = "130624-112003"
snapshot_data = f"./raw_output/{file_set}"

viridis = cm.get_cmap("viridis")

with open(os.path.join(snapshot_data, "snapshots.json"), encoding="utf8") as f:
    snapshots = json.load(f)

with open(os.path.join(snapshot_data, "cusp.json"), encoding="utf8") as f:
    cusp = json.load(f)

with open(os.path.join(main_data, "extra_data.json"), encoding="utf8") as f:
    data = json.load(f)

with open(os.path.join(main_data, "crit_data.json"), encoding="utf8") as f:
    crit_data = np.array(json.load(f))

with open("./pdf_data.json", encoding="utf8") as f:
    pdf_data = json.load(f)

eta_dict = {value: np.log10(float(value)) for value in data.keys()}
min_eta = min(eta_dict.values())
eta_range = max(eta_dict.values()) - min_eta

for k, v in eta_dict.items():
    eta_dict[k] = (v - min_eta) / eta_range

etas, fits = compute_critical_approach_fit(data)

figure_1, ax_1 = plt.subplots()
generate_figure_one(ax_1, data, eta_dict)
figure_1.show()
figure_1.savefig("figure_1.svg")

figure_2, ax_2 = plt.subplots()
generate_figure_two(ax_2, data, eta_dict)
figure_2.savefig("figure_2.svg")

figure_3, ax_3 = plt.subplots()
generate_figure_three(ax_3, etas, fits, eta_dict)
figure_3.savefig("figure_3.svg")

figure_4, ax_4 = plt.subplots()
generate_figure_four(ax_4, etas, crit_data)
figure_4.savefig("figure_4.svg")

figure_5, ax_5 = plt.subplots()
generate_figure_five(ax_5, etas, crit_data)
figure_5.savefig("figure_5.svg")

figure_6 = plt.figure(figsize=plt.figaspect(1), dpi=100)
generate_figure_six(figure_6, etas, data)
figure_6.savefig("figure_6.svg")

figure_7 = plt.figure(figsize=plt.figaspect(0.333), dpi=100)
generate_figure_seven(figure_7, snapshots)
figure_7.savefig("figure_7.svg")

figure_8 = plt.figure(figsize=plt.figaspect(0.333), dpi=100)
pdf_snapshot = next(snapshot for snapshot in snapshots if snapshot["eta"] > 0 and snapshot["beta"] == 10)
generate_figure_eight(figure_8, pdf_snapshot, pdf_data)
figure_8.savefig("figure_8.svg")

figure_8a = plt.figure(figsize=plt.figaspect(0.333), dpi=100)
pdf_snapshot = next(snapshot for snapshot in snapshots if snapshot["eta"] == 0 and snapshot["beta"] == 10)
generate_figure_eight_a(figure_8a, pdf_snapshot, pdf_data)
figure_8a.savefig("figure_8a.svg")


plt.show()
print("Done")
