# Unsupervised QEP

Here, we apply QEP to tasks that are not supervised, such as maximizing certain
expectation values to explore a phase diagram or maximizing the slope of an expectation
value w.r.t. a relevant system parameter, which, for instance, may be used to divise
efficient sensors or find phase boundaries.
Concretely, we discuss the following applications:
- phase exploration in a cluster Ising chain of length 10 by optimizing $\langle X_1 X_4\rangle$
- sensitivity optimization in a cluster Ising chain of length 10 by optimizing $\frac{\partial\langle X_1 X_4\rangle}{\partial g_X}$
with $g_X$ the strength of the magnetic field.
