#!/bin/bash -l

##############################
#       Job blueprint        #
##############################

# Give your job a name, so you can recognize it in the queue overview
#SBATCH --job-name=tokenize ## CHANGE JOBNAME HERE
#SBATCH --array=0

# Define, how many nodes you need. Here, we ask for 1 node.
#SBATCH -N 1                                        ##nodes
#SBATCH -n 1                                        ##tasks
#SBATCH --cpus-per-task=16

python tokenize_files.py
