# QuantumEP

Code for the work presented in "Quantum Equilibrium Propagation for efficient training of quantum systems based on Onsager reciprocity"

Quantum Equilibrium Propagation represents a new method for performing efficient training of quantum systems with many tuneable parameters, e.g. quantum simulators. Onsager reciprocity allows to replace many linear response experiments that would be needed to obtain the training gradients by a single linear response experiment. The classical version of equilibrium propagation already has been the most intensively studied physics-based training method for neuromorphic systems for a while, since its introduction in 2017 (Scellier and Bengio). This is the quantum version.

![image](https://github.com/ClaraWanjura/QuantumEP/assets/66438106/57449129-f557-4188-8035-b2e6a7dd5563)


See: https://arxiv.org/abs/2406.06482

Please cite as: C. C. Wanjura and F. Marquardt, Quantum Equilibrium Propagation for efficient training of quantum systems based on Onsager reciprocity, arXiv:2406.06482 (2024).

```
@article{wanjura2024QEP,
  title={Quantum Equilibrium Propagation for efficient training of quantum systems based on Onsager reciprocity},
  author={Wanjura, Clara C and Marquardt, Florian},
  journal={arXiv preprint arXiv:2406.06482},
  year={2024},
  doi = {arXiv:2406.06482},
  url = {https://arxiv.org/abs/2406.06482}
}
```

Required packages:
Tested with Python 3.11.7;
required packages: numpy (tested with 1.26.2), scipy (tested with 1.11.4), matplotlib (tested with 3.10.0), tqdm (tested with 4.66.4), optax (tested with 0.2.2)

Run instructions:
The Jupyter notebooks can be run using typical code editors such as Visual Studio Code. The python code can be run from the command line. Detailed instructions are included in the README file in the designated subfolder.
