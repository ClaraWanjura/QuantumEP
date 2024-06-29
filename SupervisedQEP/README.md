# Code for the supervised learning part

Training to recognize quantum phases of a quantum many-body system, 
by coupling a quantum sensor with tuneable parameters and learning those parameters using QEP.

There is the original notebook that contains most of the code to play around with, plus the folder ProductionRuns, which includes the actual runs used for most of the subfigures in the paper (except a few figures that are already inside the notebook).

As you will see, the notebook (or the module qep.py inside the production run folders) contains general functionalities for constructing a spin/qubit Hamiltonian using sparse matrices, for QEP training, and for the specific example of the cluster Ising Hamiltonian and phase recognition.

