The code in this repository performs 2 tasks from our paper (doi: https://doi.org/10.1101/2021.10.21.465371)

# TASK 1: Given two PDB files, predict the structural dis-similarity probability.
	This step generate the (dis)-similarity probability for the structure.
	Usage : run_methods.sh pdb1 pdb2
	example:
	```
	./run_methods.sh pdb1 pdb2
	```
	The code takes the 2 pdb files as arguments.
	The generated input feature (3DZD) is stored in the data/ directory
	The predictions is stored in the output/ directory
		
# TASK 2: Given a PDB file, predict the fold classification based on secondary structure.
	This step uses a bagged SVM classifier to predict fold class from secondary structure.
	
	Dependencies:
	 - DSSP ([https://swift.cmbi.umcn.nl/gv/dssp/](https://swift.cmbi.umcn.nl/gv/dssp/))
	 - BioPython ([https://biopython.org/](https://biopython.org/))
	 - scikit-learn ([https://scikit-learn.org](https://scikit-learn.org))
	 
	Usage:
	```
	python3 predict_fold_class_by_model.py pdb1 pdb2
	```

# Questions/Inquires: 
Contact Prof. Daisuke Kihara at [dkihara@purdue.edu](mailto:dkihara@purdue.edu)
