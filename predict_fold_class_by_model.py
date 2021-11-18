#!/usr/bin/env python3

# Copyright (C) 2021 Charles Christoffer, Daisuke Kihara, and Purdue University
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

import sys
import os
import csv
from collections import defaultdict

import numpy as np

import sklearn
import sklearn.linear_model
import sklearn.metrics
from sklearn import preprocessing
import sklearn.svm
from sklearn.svm import SVC
from sklearn.ensemble import BaggingClassifier

import joblib

from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP


s1d_ss8_ss3_mapping = {
    "E":"E",
    "B":"E",

    "G":"H",
    "H":"H",
    "I":"H",

    "C":"C",
    "S":"C",
    "T":"C",
	"-":"C",
}

if not sys.argv[1:]:
	print("USAGE:", file=sys.stderr)
	print("    %s [INPUT.pdb]..." % (sys.argv[0]), file=sys.stderr)
	print("EXAMPLE:", file=sys.stderr)
	print("    %s 1cll.pdb" % (sys.argv[0]), file=sys.stderr)
	print("EXAMPLE OUTPUT:" % (), file=sys.stderr)
	print("    structure_file_path,predicted_fold_class", file=sys.stderr)
	print("    1cll.pdb,alpha", file=sys.stderr)
	exit(1)

# load model from files
scaler_path = os.path.join(sys.path[0], "scaler.joblib")
classifier_path = os.path.join(sys.path[0], "baggedrbfsvc.joblib")

scaler = joblib.load(scaler_path)
clf = joblib.load(classifier_path)

parser = PDBParser()

# run dssp for each input file
data = []
for infile in sys.argv[1:]:
	try:
		struct = parser.get_structure(infile, infile)
		model = struct[0]
	except Exception as e:
		print("[error] exception raised while loading input file '%s':" % (infile), e, file=sys.stderr)
		continue

	try:
		dssp = DSSP(model, infile)
		accum = defaultdict(int)
		for k in dssp.keys():
			v = dssp[k]
			accum[s1d_ss8_ss3_mapping[v[2]]] += 1
		
		nres = sum(accum.values())
		
		fracs = {}
		for k, v in accum.items():
			fracs[k] = v/float(nres)
			
		data.append((infile, '?', float(nres), fracs['C'], fracs['E'], fracs['H']))
	except Exception as e:
		print("[error] exception raised while running and collating DSSP for input file '%s':" % (infile), e, file=sys.stderr)
		continue

# scale all dssp features and classify in bulk
features = np.array([tuple(l[2:]) for l in data])
data_scaled = scaler.transform(features)
pred = clf.predict(data_scaled)

classname_mapping = {
	'a': "alpha",
	'b': "beta",
	'x': "alphabeta_other",
	'g': "small_protein",
}

csvwriter = csv.writer(sys.stdout)
csvwriter.writerow(["structure_file_path", "predicted_fold_class"])
csvwriter.writerows((l[0], classname_mapping[p]) for l, p in  zip(data, pred))

