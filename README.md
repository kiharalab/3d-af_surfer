The code in this repository performs 2 task based on our paper (doi: https://doi.org/10.1101/2021.10.21.465371)

# TASK 1: Given two pdb files, predict the structural dis-similarity probability 
	This step generate the (dis)-similarity probability for the structure.
	Usage : run_methods.sh pdb1 pdb2
	example:
	```
	./run_methods.sh pdb1 pdb2
	```
	The code takes the 2 pdb files as arguements.
	The generated input feature(3DZD) is stored in the data/ directory
	The predictions is stored in the output/ directory
		
# TASK 2: Given a pdb file, generate the secondary structure classification.
	This step calculate the secondary structure prediction based on svm....
	```
	Command: python3 ....
	```

# question(s)/Inquires: 
Contact Prof Daisuke Kihara at dkihara@purdue.edu
