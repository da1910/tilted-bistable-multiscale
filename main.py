import datetime
import json
import os
import subprocess
import scipy.optimize

from util import *


def left_end_function(x: float, a1: float, a2: float) -> float:
    return x ** 3 - a1 * x + a2


beta_initial = 20.0
lambda_initial = -2.0
etas = np.logspace(-3, -1, 21)
lambda_ = lambda_initial
critical_approach_data = np.empty((0, 4), float)

results = {}

dist_tol = 1e-8
beta_tol = 1e-8
timestamp = datetime.datetime.now().strftime('%d%m%y-%H%M%S')

current_dir, _ = os.path.split(os.path.abspath(__file__))
temp_dir = os.path.join(current_dir, 'auto_working_dir')
template_dir = os.path.join(current_dir, 'auto_templates')

auto_output_dir = os.path.join(current_dir, 'raw_output', timestamp)
os.mkdir(auto_output_dir)
processed_output_dir = os.path.join(current_dir, 'processed_output', timestamp)
os.mkdir(processed_output_dir)

OUTPUT_FILES = ('d.lmbda', 'b.lmbda', 's.lmbda', 'd.cusp', 'b.cusp', 's.cusp')

for eta in etas:
    print('Running with eta = {}'.format(eta))

    beta = beta_initial
    lend = scipy.optimize.broyden1(lambda x: left_end_function(x, lambda_, eta), [0])[0]

    print('Starting at lambda = {}, x = {}'.format(lambda_initial, lend))
    print('Executing AUTO-07p...')
    create_output_file(os.path.join(template_dir, 'raw.f90'),
                       os.path.join(temp_dir, 'hompdf.f90'),
                       {'BETAINPUT': to_fortran_string(beta),
                        'LAMBDAINPUT': to_fortran_string(lambda_initial),
                        'ETAINPUT': to_fortran_string(eta),
                        'X0INPUT': to_fortran_string(lend)})
    os.chdir(temp_dir)
    p = subprocess.Popen(['auto', 'hompdf.auto'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()
    os.chdir('../')

    print('Processing results...')
    cusp_data = process_input(os.path.join(temp_dir, 'b.cusp'))

    if -22 in cusp_data[:, 2]:
        row = cusp_data[np.where(cusp_data[:, 2] == -22)]
        if len(row) == 2:
            row_indices = np.where(cusp_data[:, 2] == -22)[0]
            means = np.mean(row, 0)
            cusp_data = np.vstack((
                cusp_data[range(0, row_indices[0]), :],
                means,
                cusp_data[range(row_indices[1] + 1, cusp_data.shape[0]), :]
            ))
            row = np.atleast_2d(row[1])

        critical_approach_data = np.vstack((critical_approach_data, np.hstack((row[0, [4, 6, 7]], eta))))

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
        results[eta]['approach'] = approach.tolist()
        results[eta]['d_lambda'] = d_lmbda.tolist()
        results[eta]['approach_long'] = approach_long.tolist()
        results[eta]['d_lambda_long'] = d_lmbda_long.tolist()

        valid_rows = ~np.isnan(d_lmbda)

        # Fit following approach from numpy.linalg.lstsq documentation
        y = np.log10(d_lmbda[valid_rows])
        x = np.log10(approach[valid_rows])
        A = np.vstack([x, np.ones(len(x))]).T

        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        results[eta]['fit'] = [m, c]
    else:
        print('No cusp found, skipping...\n')

    print("Dumping raw output from AUTO to '{}'...\n".format(auto_output_dir))
    os.remove(os.path.join(temp_dir, 'hompdf.f90'))
    escaped_folder_name = str(round(eta, 4)).replace('.', '_')
    os.mkdir(os.path.join(auto_output_dir, escaped_folder_name))
    for output_file in OUTPUT_FILES:
        os.rename(os.path.join(temp_dir, output_file), os.path.join(auto_output_dir, escaped_folder_name, output_file))

print("Dumping processed results to '{}'...".format(processed_output_dir))
with open(os.path.join(processed_output_dir, 'results.json'), 'w', encoding='utf8') as fp:
    json.dump(results, fp)

with open(os.path.join(processed_output_dir, 'crit_data.json'), 'w', encoding='utf8') as fp:
    json.dump(critical_approach_data.tolist(), fp)

