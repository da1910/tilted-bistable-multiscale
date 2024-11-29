# Figure Notes

## General Comments
The figures are created using the `create_figures.py` script in the repository root. They rely on data files
that are created by scripts.

Most of the data is generated with the `main.py` script which runs Auto07p and performs numerical
continuation. In order for this to work you must have a working installation of Auto07p with
the python bindings. By default the script will look in `~/auto/07p/bin` for the auto binary,
this can be overridden by setting the `AUTO_DIR` environment variable.

The PDFs are generated using MATLAB. Run the `pdf/create_runs.m` script to run multiple runs with
different parameter values.

## Figure 1 and 2
Figure 1 is the inset from Figure 2.

Each series corresponds to a separate value of the tilt parameter $\eta$, varying from $10^{-4}$ to
$10^{-0.8}$. Larger values of tilt exclude the metastable region entirely.

Each run in this regime is performed by continuing the principal branch in both directions starting from
a value of $\lambda=-3$, $\beta=20$, the initial estimated $x$ value is computed by solving the macroscale equation
$x^3 - \lambda x + \eta$.

## Figure 3
Figure 3 is the same data as from Figure 2, but rather than plotting the distance from the critical point
it fits the distance to an expression of the form $y = x^m$. The value of $m$ is plotted against the tilt $\eta$.

## Figure 4
Figure 4 shows the same data, but presented to show the effect of the tilt on the critical value of $\sigma$.

## Figure 5
The same data set is presented in Figure 5 showing the location of the critical point in $x-\lambda$ space for
different values of $\eta$

## Figure 6
Figure 6 shows all the series from Figures 1 and 2, but plotted in $x-\sigma-\lambda$ space for each value
of $\eta$

## Figure 7
Figure 7 shows the bifurcation diagram for three values noise intensity, here plotted for $\beta=40, \beta=7.6, \beta=1$,
each diagram shows the same value of $\eta=0.1$

## Figure 8a + 8b
Figure 8a and 8b show the bifurcation diagrams and the measured PDFs for example values of $\lambda$, each
is for $\beta=10$, the labels show the values of each slice. Here the experimental scale separation parameter
is $0.1$. For figure 8a the value of the tilt $\eta=0.1$, for 8b it is $\eta=0$.
