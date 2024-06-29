# Production runs

These subdirectories contain the code for production runs 
that were used in creating the figure on supervised learning 
(phase diagram of the cluster Ising Hamiltonian) in the manuscript.

Inside each subdirectory you find a go.py that imports the qep.py module and
then runs a particular version of training, with a selected set of hyperparameters
and other options. Results of several runs are inside .npz files, in case you do not
want to run the code yourself but still inspect typical results. 

Results were collected from the outputs produced (running the code on a cluster) and inserted into the
notebook provided here for the final plotting. We left the original naming conventions intact, so the filenames correspond to those mentioned inside the notebook that is used for plotting.

The main code base (i.e. the qep.py module) may slightly differ between the runs
in the various subdirectories, when functionalities were added later.
