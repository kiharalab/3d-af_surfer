#!/usr/bin/env bash

#Usage : run_methods.sh pdb1 pdb2
# A script to run the neural network pipeline to compare two pdb structure 
# The result will be in the output directory
if [ $# -lt 2 ]
  then
    echo "Usage : run_methods.sh pdb1 pdb2"
	exit
fi

date
pdb1dir="$1"
pdb2dir="$2"

echo "now generating the 3dzd for the two structures. output will be in the data/ directory."
python3 generate_3dzd.py ${pdb1dir} data/
python3 generate_3dzd.py ${pdb2dir} data/


# Generating the similarity prediction for both structure

python3 predict_similarity.py --atom_type mainchain
python3 predict_similarity.py --atom_type fullatom

echo 'Done check results in the output directory '
