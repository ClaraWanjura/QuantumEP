# Phase exploration

Here, we show phase exploration in a cluster Ising chain of length 10 described by the Hamiltonian
$` H = g_X \sum_j X_j + g_{ZZ} \sum_j Z_j Z_{j+1} + g_{ZXZ} \sum_j Z_{j-1} X_j Z_{j+1} `$
by optimizing $` \langle X_1 X_4 \rangle `$. 

'findPhases.ipynb' is the Jupyter notebook that implements the optimization with QEP, the
folder results contains the data and the plots for the two runs from different initial points
within the phase diagram as well as the plot of the trajectories in the phase diagram.

Required packages: numpy, scipy (sparse matrices), matplotlib (plotting).
