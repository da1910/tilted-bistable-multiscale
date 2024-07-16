import json
import os
import sys
import datetime

from scipy import optimize
from scipy.special import i1, i0

auto_directory = os.getenv("AUTO_DIR")
if auto_directory is None:
    home = os.getenv("HOME")
    auto_directory = home+'/auto/07p/'
    sys.path.append(auto_directory+'/python')
else:
    sys.path.append(auto_directory[0] + '/python')

from auto import runAUTO as ra
from auto import interactiveBindings as ib
from auto import AUTOclui as acl
from auto import AUTOCommands as ac

from util import *


def left_end_function(x: float, a1: float, a2: float, a3: float) -> float:
    return x ** 3 - (a1 + i1((a2 * x ** 2) / 2) / i0((a2 * x ** 2) / 2)) * x + a3


def generate_snapshot(beta: float, eta: float):
    lambda_initial = 3.0
    lambda_ = lambda_initial

    print("Running with eta = {}".format(eta))

    runner = ra.runAUTO()
    lpa = ac.load('hompdf', runner=runner)
    rend_upper = optimize.broyden1(lambda x: left_end_function(x, lambda_, beta, eta), [2.0])[0]
    rend_lower = optimize.broyden1(lambda x: left_end_function(x, lambda_, beta, eta), [-2.0])[0]
    b1 = lpa.run(PAR={1: lambda_, 2: beta, 3: eta}, DS='-', U={1: rend_upper})
    b2 = lpa.run(PAR={1: lambda_, 2: beta, 3: eta}, DS='-', U={1: rend_lower})

    return {
        "eta": eta,
        "beta": beta,
        "b1": b1.toArray(),
        "b2": b2.toArray(),
    }

def generate_cusp_diagram():
    beta_test = 20
    eta_test = 1e-3
    lambda_initial = -3.0

    lend = optimize.broyden1(lambda x: left_end_function(x, lambda_initial, beta_test, eta_test), [0])[0]
    model_parameters, solver_parameters = generate_arguments(lambda_initial, beta_test, eta_test, lend)

    runner = ra.runAUTO()
    hompdf = ac.load('hompdf', runner=runner)
    lmbda = ac.run(hompdf, runner=runner, **model_parameters)
    lmbda = ac.relabel(lmbda)

    # Load limit point location and continue in two parameters, saving as cusp
    lp1 = ac.load(lmbda('LP1'), runner=runner, ISW=2, ICP=[1, 2])
    lp_locus = ac.run(lp1, runner=runner) + ac.run(lp1, runner=runner, DS='-')

    # Continue in two parameters with eta free, run from cusp to one limit first
    cusp1 = ac.load(lp_locus('CP')[0], runner=runner, ISW=2, ICP=[2,3], NMX=6400)
    output1 = ac.run(cusp1, runner=runner, DS='-')

    # Now run backward to sweep the entire space
    cusp2 = ac.load(output1('UZ')[0], runner=runner, ISW=2, ICP=[2,3], NMX=9600)
    output = ac.run(cusp2, runner=runner)

    return {
        "data": output.toArray(),
    }


etas = [0, 0.1]
betas = [40, 10, 7.6, 4.0, 1]
results = []
os.chdir('./auto_working_dir')
for eta in etas:
    for beta in betas:
        results.append(generate_snapshot(beta, eta))

cusp_diagram = generate_cusp_diagram()
os.chdir('../')

current_dir, _ = os.path.split(os.path.abspath(__file__))
timestamp = datetime.datetime.now().strftime("%d%m%y-%H%M%S")
auto_output_dir = os.path.join(current_dir, "raw_output", timestamp)
os.mkdir(auto_output_dir)
output_file = os.path.join(auto_output_dir, "snapshots.json")
with open(output_file, 'w', encoding='utf8') as fp:
    json.dump(results, fp)
cusp_file = os.path.join(auto_output_dir, "cusp.json")
with open(cusp_file, 'w', encoding='utf8') as fp:
    json.dump(cusp_diagram, fp)
