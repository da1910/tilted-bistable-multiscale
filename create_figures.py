import numpy as np
from matplotlib import pyplot as plt
import os
import json

input_files = os.listdir('./processed_output')
sorted_files = sorted(input_files, reverse=True)
selected_data = './processed_output/{}'.format(sorted_files[0])

with open(os.path.join(selected_data, 'approach.json'), encoding='utf8') as f:
    approach = json.load(f)

with open(os.path.join(selected_data, 'approach_long.json'), encoding='utf8') as f:
    approach_long = json.load(f)

with open(os.path.join(selected_data, 'd_lambda.json'), encoding='utf8') as f:
    d_lambda = json.load(f)

with open(os.path.join(selected_data, 'd_lambda_long.json'), encoding='utf8') as f:
    d_lambda_long = json.load(f)

with open(os.path.join(selected_data, 'fits.json'), encoding='utf8') as f:
    fits = json.load(f)

with open(os.path.join(selected_data, 'crit_data.json'), encoding='utf8') as f:
    crit_data = json.load(f)


figure_1, ax_1 = plt.subplots()
ax_1.set_xlabel(r"$\log\left(\sigma_{c}-\sigma\right)/\sigma_{c}$")
ax_1.set_ylabel(r"$\log\left|\alpha_{c}\right|/N$")
for eta in d_lambda:
    current_approach = approach[eta]
    current_d_lambda = d_lambda[eta]
    ax_1.scatter(np.log10(current_approach), np.log10(current_d_lambda),
                 90, np.tile(np.log10(float(eta)), (100, 1)), marker='o')
figure_1.savefig('figure_1.svg')

figure_2, ax_2 = plt.subplots()
ax_2.set_xlabel(r'$\left(\sigma_{c}-\sigma\right)/\sigma_{c}$')
ax_2.set_ylabel(r'$\left|\alpha_{c}\right|/N$')
for eta in d_lambda:
    current_approach = approach_long[eta]
    current_d_lambda = d_lambda_long[eta]
    ax_2.scatter(current_approach, current_d_lambda,
                 9, np.tile(np.log10(float(eta)), (100, 1)))
figure_2.savefig('figure_2.svg')

figure_3, ax_3 = plt.subplots()
etas = list(fits.keys())
fitted_polynomials = list(fits.values())
ax_3.scatter(etas, [x[0] for x in fitted_polynomials], 36, np.log10([float(eta) for eta in etas]))
ax_3.set_xlabel(r'\eta')
ax_3.set_ylabel(r'\gamma - Exponent in critical approach')
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

figure_6, ax_6 = plt.subplots()
ax_6.loglog(crit_data[:, 1], crit_data[:, 2])
ax_6.set_xlabel(r'$\lambda_{c}$')
ax_6.set_ylabel(r'$x_{c}$')
figure_6.savefig('figure_6.svg')

plt.show()
print('Done')