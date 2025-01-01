#!/bin/bash -l

# Standard output and error:
#SBATCH -o ./tjob.out.%j
#SBATCH -e ./tjob.err.%j
# Initial working directory:
#SBATCH -D ./
#SBATCH -J qep
#SBATCH --partition=standard

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
# memory
#SBATCH --mem=164GB
#
#SBATCH --mail-type=none
#SBATCH --mail-user=florian.marquardt@mpl.mpg.de
#
# Wall clock limit:
#SBATCH --time=168:00:00

#module load anaconda/3/2021.11 gcc cuda/11.6 mkl git intel/21.2.0 impi/2021.2 cudnn
conda activate jax_env

python -u go.py

