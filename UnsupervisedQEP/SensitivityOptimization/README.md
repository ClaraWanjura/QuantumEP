# Sensitivity optimization

Here, we show sensitivity optimization in a cluster Ising chain of length 10 described by the Hamiltonian
$` H = g_X \sum_j X_j + g_{ZZ} \sum_j Z_j Z_{j+1} + g_{ZXZ} \sum_j Z_{j-1} X_j Z_{j+1} `$
by optimizing $\partial\langle X_1 X_4 \rangle/ \partial g_X$ with QEP. The necessary gradients are computed
according to the formulas given in the appendix of the preprint https://arxiv.org/abs/2406.06482.

$\partial_{g_X^{(1)}}\mathcal{L}
= \varepsilon \, \frac{\partial_{g_X^{(1)}}\langle \hat X_0 \hat X_4 \rangle\rvert_{g_X^{(1)}}}
{\lvert g_X^{(1)} - g_X^{(2)}\rvert} + \mathrm{sgn} (g_X^{(1)} - g_X^{(2)}) \frac{\mathcal{L}}{\lvert g_X^{(1)} - g_X^{(2)}\rvert}$

$\partial_{g_X^{(2)}}\mathcal{L}
= - \varepsilon \, \frac{\partial_{g_X^{(2)}}\langle \hat X_0 \hat X_4 \rangle\rvert_{g_X^{(2)}}}
{\lvert g_X^{(1)} - g_X^{(2)}\rvert} - \mathrm{sgn} (g_X^{(1)} - g_X^{(2)}) \frac{\mathcal{L}}{\lvert g_X^{(1)} - g_X^{(2)}\rvert}$

$\partial_{g_{ZZ}}\mathcal{L}
= \varepsilon \, \partial_{g_{ZZ}}
\frac{\left(\langle \hat X_0 \hat X_4 \rangle\rvert_{g_X^{(1)}} - \langle \hat X_0 \hat X_4 \rangle\rvert_{g_X^{(2)}}\right)}
{\lvert g_X^{(1)} - g_X^{(2)}\rvert}$

with $\mathcal{L}$ the loss function and the error signal $\varepsilon\equiv \mathrm{sgn}(\langle \hat X_0 \hat X_4\rangle \vert_{g_X^{(1)}} - \langle \hat X_0 \hat X_4\rangle \vert_{g_X^{(2)}})$.

'magneticFiledSensor.ipynb' is the Jupyter notebook for the optimization task and implements QEP; the folders 'run1' and 'run2' contain the results (data files and plots).

Required packages: numpy, scipy (sparse matrices), matplotlib (plotting).
