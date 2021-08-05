# Tilted Multiscale Bifurcation
##Problem formulation
This repo contains python scripts to interface with AUTO-07p and investigate effect of noise in a multiscale potentials.
The steady-state behaviour of the system

$$dX_t = -\del V(X_t)dt + \sqrt{2\sigma}dB_t$$,

where

$$V(x) = \frac{1}{4}x^{4} + \frac{\alpha}{2}x^{2} - \frac{1}{2}\sin{\frac{2\pi x}{\epsilon}}x^{2} + \eta x$$

where $$\eta$$ is a small parameter destroying the symmetry of the system, and $$\epsilon$$ is the scale separation
parameter.

Taking the limit as $$\epsilon \rightarrow \infty$$ and solving for the extrema gives

$$x^{3} - \frac{I_{1}(\frac{x^2}{2\sigma})}{I_{0}(\frac{x^{2}}{2\sigma})}x + \eta = 0$$.

## Contents of the repo
The repo contains the AUTO-07p source files and templates that are used to solve the equation above by numerical
continuation. 

The file `main.py` is responsible for running the analysis throughout the parameter space of interest, and the 
accompanying `util.py` contains some useful helper functions.

The script `create_figures` can be used to generate the figures that summarise the analysis completed.

## Instructions to run the analysis
The scripts assume that the package AUTO-07p is available on your system and included on the path. The process to 
install this varies by platform, on Ubuntu 20.05 it was necessary to:

```
sudo apt install gfortran libsoqt4-dev xterm libmotif-dev
wget https://github.com/auto-07p/auto-07p/releases/download/v0.9.2/auto-0.9.2.tar.gz
tar -xvf auto-0.9.2.tar.gz
cd auto
mkdir build
cd build
../07p/configure
make
```

Running the `main.py` script should generate the raw and processed output covering the parameter space.
