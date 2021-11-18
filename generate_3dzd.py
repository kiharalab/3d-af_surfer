# Copyright (C) 2021 Tunde Aderinwale, Daisuke Kihara, and Purdue University
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
#
# convert pdb file to triangulation data and do 3dzd calculation

import os
import glob
from multiprocessing import Pool
import sys
import Bio.PDB as bpdb
import Bio.PDB
import numpy as np
from io import StringIO
from Bio.PDB import Select

def get_unpacked_list(self):
    """
    Returns all atoms from the residue,
    in case of disordered, keep only first alt loc and remove the alt-loc tag
    """
    atom_list = self.get_list()
    undisordered_atom_list = []
    for atom in atom_list:
        if atom.is_disordered():
            atom.altloc=" "
            undisordered_atom_list.append(atom)
        else:
            undisordered_atom_list.append(atom)
    return undisordered_atom_list
Bio.PDB.Residue.Residue.get_unpacked_list = get_unpacked_list

def extract_and_write_mainchain(filename,output_dir):
	pdb_path = filename
	fileid = filename.split('/')[-1]#.split('.')
	print(fileid)
	
	output_file = output_dir + fileid + '_cacn.pdb'
	print(output_file)
	pdbparser = bpdb.PDBParser(PERMISSIVE=1,QUIET = True)
	pdb_file = pdbparser.get_structure('',pdb_path)[0]

	# structure = bpdb.StructureBuilder.Model(0) # create empty structure
	# atom_list = [atom for atom in pdb_file.get_atoms() if atom.name in ["CA","C","N"]]
	# structure.add(atom_list)
	# print(structure)

	class MainchainSelect(Select):
		def accept_atom(self, atom):
			if atom.get_name() in ["CA","C","N"]:
				return 1
			else:
				return 0


	io = bpdb.PDBIO()
	io.set_structure(pdb_file)
	io.save(output_file,MainchainSelect())
	return output_file


def plytoobj(filename):
	obj_filename = filename[:-4] + '.obj'
	obj_file = open(obj_filename, 'w')

	with open(filename) as ply_file:
		ply_file_content = ply_file.read().split('\n')[:-1]

		for content in ply_file_content:
			content_info = content.split()
			if len(content_info) == 6:
				vertex_info = 'v ' + ' '.join(content_info[0:3])
				obj_file.write(vertex_info + '\n')
			elif len(content_info) == 7:
				vertex1, vertex2, vertex3 = map(int, content_info[1:4])
				vertex1, vertex2, vertex3 = vertex1 + 1, vertex2 + 1, vertex3 + 1
				face_info = 'f ' + str(vertex1) + ' ' + str(vertex2) + ' ' + str(vertex3)
				obj_file.write(face_info + '\n')

		obj_file.close()

def pdbtoinv(filename,output_dir,mainchain):
	ply_dir = output_dir
	fileid = filename.split('/')[-1].split('.')[0]

	if mainchain:
		fileid += '_cacn'

	print(fileid)
	plyfile = ply_dir + fileid + '.ply'
	invfile = ply_dir + fileid + '.inv'

	if not os.path.isfile(invfile):
		surf_command = './bin/EDTSurf -i ' + filename + ' -h 2 -o ' + ply_dir + fileid
		#print(surf_command)
		os.system(surf_command)

		# convert ply file to obj file
		#invfile = ply_dir + fileid + '.inv'
		if os.path.isfile(plyfile):
			plytoobj(plyfile)

			# generate 3dzd
			obj_file = ply_dir + fileid + '.obj'
			cp_command = 'cp ' + obj_file + ' ./' + fileid + '.obj'
			#print(cp_command)
			os.system(cp_command)

			grid_command = './bin/obj2grid -g 64  ./' + fileid + '.obj'
			#print(grid_command)
			os.system(grid_command)

			inv_command = './bin/map2zernike ' + fileid + '.obj.grid -c 0.5 '#--save-moments'
			#print(inv_command)
			os.system(inv_command)

			mv_command = 'mv ' + fileid + '.obj.grid.inv ' + fileid + '.inv'
			#print(mv_command)
			os.system(mv_command)

			mv_command = 'mv ' + fileid + '.* ' + ply_dir
			#print(mv_command)
			os.system(mv_command)

			rm_command = 'rm ' + ply_dir + fileid + '.obj'
			#print(rm_command)
			os.system(rm_command)

			rm_command = 'rm ' + ply_dir + fileid + '.obj.grid'
			#print(rm_command)
			os.system(rm_command)

pdb_file = sys.argv[1]
output_dir = sys.argv[2]

pdbtoinv(pdb_file,output_dir,False)
mainchain_file = extract_and_write_mainchain(pdb_file,output_dir)
pdbtoinv(mainchain_file,output_dir,True)


