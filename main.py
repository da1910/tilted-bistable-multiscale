import datetime
import json
import os
from pathlib import Path

import scipy.optimize
import numpy as np

from util import *

import sys
auto_directory = os.getenv("AUTO_DIR")
if auto_directory is None:
    home = os.getenv("HOME")
    auto_directory = home+'/auto/07p/'
    sys.path.append(auto_directory+'/python')
else:
    sys.path.append(auto_directory[0] + '/python')

from auto import AUTOCommands as ac
from auto import AUTOclui as acl
from auto import interactiveBindings as ib
from auto import runAUTO as ra


def left_end_function(x: float, a1: float, a2: float) -> float:
    return x ** 3 - a1 * x + a2


beta_initial = 20.0
lambda_initial = -3.0
etas = np.logspace(-3, -1, 21)
lambda_ = lambda_initial
critical_approach_data = np.empty((0, 4), float)

results = {}

dist_tol = 1e-8
beta_tol = 1e-8
timestamp = datetime.datetime.now().strftime("%d%m%y-%H%M%S")

current_dir, _ = os.path.split(os.path.abspath(__file__))
temp_dir = os.path.join(current_dir, "auto_working_dir")
template_dir = os.path.join(current_dir, "auto_templates")

auto_output_dir = os.path.join(current_dir, "raw_output", timestamp)
os.mkdir(auto_output_dir)
processed_output_dir = os.path.join(current_dir, "processed_output", timestamp)
os.mkdir(processed_output_dir)

OUTPUT_FILES = ("d.lmbda", "b.lmbda", "s.lmbda", "d.cusp", "b.cusp", "s.cusp")
runner = ra.runAUTO()

for eta in etas:
    print("Running with eta = {}".format(eta))

    beta = beta_initial
    lend = scipy.optimize.broyden1(lambda x: left_end_function(x, lambda_, eta), [0])[0]

    print(f"Starting at lambda = {lambda_initial}, x = {lend}")
    print("Executing AUTO-07p...")

    model_parameters, solver_parameters = generate_arguments(lambda_, beta, eta, lend)

    os.chdir(Path(__file__).parent / "./auto_working_dir")

    hompdf = ac.load('hompdf', runner=runner)

    # Run and store the result in the Python variable lmbda
    lmbda = ac.run(hompdf, runner=runner, **model_parameters)
    lmbda = ac.relabel(lmbda)

    # Load limit point location and continue in two parameters, saving as cusp
    lp1 = ac.load(lmbda('LP1'), runner=runner, ISW=2, ICP=[1, 2])
    cusp = ac.run(lp1, runner=runner) + ac.run(lp1, runner=runner, DS='-')
    cusp.reverse()
    cusp.data[0].coordarray = np.flip(cusp.data[0].coordarray, axis=1)

    os.chdir("../")

    print("Processing results...")
    param_data = cusp.merge().toArray()
    cusp_data = np.empty((0, 8), float)
    for index, _ in enumerate(param_data):
        current_point = cusp.getIndex(index)
        cusp_data = np.append(cusp_data, [[current_point["BR"], current_point["PT"], current_point["TY number"], current_point["LAB"], *current_point["data"]]], axis=0)


    if -22 in cusp_data[:, 2]:
        row = cusp_data[np.where(cusp_data[:, 2] == -22)]
        if len(row) > 1:
            row_indices = np.where(cusp_data[:, 2] == -22)[0]
            means = np.mean(row, 0)
            cusp_data = np.vstack(
                (
                    cusp_data[range(0, row_indices[0]), :],
                    means,
                    cusp_data[range(row_indices[-1] + 1, cusp_data.shape[0]), :],
                )
            )
            row = np.atleast_2d(row[1])

        critical_approach_data = np.vstack(
            (critical_approach_data, np.hstack((row[0, [4, 6, 7]], eta)))
        )

        row = np.where(cusp_data[:, 2] == -22)[0]

        b1 = cusp_data[range(0, row[0] + 1), :][:, (4, 6, 7)]
        b2 = cusp_data[range(row[0], len(cusp_data)), :][:, (4, 6, 7)]

        b1 = np.column_stack((b1, 1 - np.divide(cusp_data[row, 7], b1[:, 2])))
        b2 = np.column_stack((b2, 1 - np.divide(cusp_data[row, 7], b2[:, 2])))

        approach = np.logspace(-1, -2, 100)
        approach_long = np.logspace(-0.2, -3, 100)

        b1_int = interp1(b1[:, 3], b1[:, [0, 1, 2]], approach)
        b2_int = interp1(b2[:, 3], b2[:, [0, 1, 2]], approach)

        b1_int_long = interp1(b1[:, 3], b1[:, [0, 1]], approach_long)
        b2_int_long = interp1(b2[:, 3], b2[:, [0, 1]], approach_long)

        d_lmbda = abs(b1_int[:, 0] - b2_int[:, 0])
        d_lmbda_long = abs(b1_int_long[:, 0] - b2_int_long[:, 0])

        results[eta] = {}
        results[eta]["raw"] = np.vstack([b1, b2]).tolist()
        results[eta]["approach"] = approach.tolist()
        results[eta]["d_lambda"] = d_lmbda.tolist()
        results[eta]["approach_long"] = approach_long.tolist()
        results[eta]["d_lambda_long"] = d_lmbda_long.tolist()

        valid_rows = ~np.isnan(d_lmbda)

        # Fit following approach from numpy.linalg.lstsq documentation
        y = np.log10(d_lmbda[valid_rows])
        x = np.log10(approach[valid_rows])
        A = np.vstack([x, np.ones(len(x))]).T

        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        results[eta]["fit"] = [m, c]
    else:
        print("No cusp found, skipping...\n")

print("Dumping processed results to '{}'...".format(processed_output_dir))
with open(
    os.path.join(processed_output_dir, "results.json"), "w", encoding="utf8"
) as fp:
    json.dump(results, fp)

with open(
    os.path.join(processed_output_dir, "crit_data.json"), "w", encoding="utf8"
) as fp:
    json.dump(critical_approach_data.tolist(), fp)
