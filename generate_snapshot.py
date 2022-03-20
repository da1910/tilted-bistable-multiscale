import json
import os
import sys
import datetime

sys.path.append('/home/ubuntu/auto/07p/python')
from auto import runAUTO as ra
from auto import interactiveBindings as ib
from auto import AUTOclui as acl
from auto import AUTOCommands as ac

from util import *


def left_end_function(x: float, a1: float, a2: float) -> float:
    return x ** 3 - a1 * x + a2


def generate_snapshot(beta: float, eta: float):
    lambda_initial = 3.0
    lambda_ = lambda_initial

    print("Running with eta = {}".format(eta))

    runner = ra.runAUTO()
    lpa = ac.load('hompdf', runner=runner)
    b1 = lpa.run(PAR={1: lambda_, 2: beta, 3: eta}, DS='-', U={1: 2.0})
    b2 = lpa.run(PAR={1: lambda_, 2: beta, 3: eta}, DS='-', U={1: -2.0})

    return {
        "eta": eta,
        "beta": beta,
        "b1": b1.toArray(),
        "b2": b2.toArray(),
    }


etas = [0.1]
betas = [40, 7.6, 1]
results = []
os.chdir('./auto_working_dir')
for eta in etas:
    for beta in betas:
        results.append(generate_snapshot(beta, eta))
os.chdir('../')

current_dir, _ = os.path.split(os.path.abspath(__file__))
timestamp = datetime.datetime.now().strftime("%d%m%y-%H%M%S")
auto_output_dir = os.path.join(current_dir, "raw_output", timestamp)
os.mkdir(auto_output_dir)
output_file = os.path.join(auto_output_dir, "snapshots.json")
with open(output_file, 'w', encoding='utf8') as fp:
    json.dump(results, fp)
