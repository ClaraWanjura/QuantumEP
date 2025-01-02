# Cluster Ising model phase recognition via QEP: Revised Manuscript version with soft coupling cutoff

The notebook PhaseSensorFigures.ipynb in this folder produces all the figures regarding cluster Ising model phase recognition
in the revised manuscript, both for the main text and the supplementary material. The only exception is the
analysis of gradient vs shot noise and nudge strength, which is unchanged, and for which the figures
can be found in the upper-level folder (SupervisedQEP) in the QEP_notebook.

The data from a number of training runs under different conditions is stored inside the ProductionRunsNew
folder. Each subfolder there contains several runs (in groups of ten runs), with go.py inside each subfolder
as the python file that is to be run to produce the training data stored inside .npz files. See 
the notebook PhaseSensorFigures.ipynb for how these data are then processed and plotted.
