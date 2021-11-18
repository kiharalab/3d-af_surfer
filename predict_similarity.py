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

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import random
import argparse
import numpy as np
import pandas as pd
import operator

from os.path import isfile, join
#from models import SimpleEuclideanModel, NeuralNetworkModel
from torch import FloatTensor, LongTensor

# Argument Parsing
parser = argparse.ArgumentParser(description='Generating Predictions')
parser.add_argument('--cuda', type=str, default='true', help='Cuda usage')
parser.add_argument('--device_id', type=int, default=0, help='GPU Device ID number')
parser.add_argument('--model_name', type=str, default='neural_network', help='Name of saved model to use for training')
parser.add_argument('--atom_type',type=str, default='fullatom', help='This is the option for setting either full or mainchain atom')

def pairs_to_features(pairs,alpha_data,scope_data):

    _3DZD_vectors_1, _3DZD_vectors_2 = [], []
    element_vertices_1, element_vertices_2 = [], []
    element_faces_1, element_faces_2 = [], []
    for _pair in pairs:
        id_0, id_1 = str(_pair[0]), str(_pair[1])
        
        _3dzd_1 = list(alpha_data[id_0]['_3dzd'])
        _3dzd_2 = list(scope_data[id_1]['_3dzd'])


        _3DZD_vector_1 = np.asarray(_3dzd_1)
        _3DZD_vector_1 = np.expand_dims(_3DZD_vector_1.squeeze(), axis = 0)
        _3DZD_vector_2 = np.asarray(_3dzd_2)
        _3DZD_vector_2 = np.expand_dims(_3DZD_vector_2.squeeze(), axis = 0)

        element_vertex_1, element_face_1 = tuple([int(x) for x in alpha_data[id_0]['vertex_face']])
        element_vertex_2, element_face_2 = tuple([int(x) for x in scope_data[id_1]['vertex_face']])

        # Update
        _3DZD_vectors_1.append(_3DZD_vector_1)
        _3DZD_vectors_2.append(_3DZD_vector_2)
        element_vertices_1.append(element_vertex_1)
        element_faces_1.append(element_face_1)
        element_vertices_2.append(element_vertex_2)
        element_faces_2.append(element_face_2)

    _3DZD_vectors_1 = FloatTensor(_3DZD_vectors_1).squeeze()
    element_vertices_1 = FloatTensor(element_vertices_1)
    element_faces_1 = FloatTensor(element_faces_1)

    _3DZD_vectors_2 = FloatTensor(_3DZD_vectors_2).squeeze()
    element_vertices_2 = FloatTensor(element_vertices_2)
    element_faces_2 = FloatTensor(element_faces_2)

    vertices_diff = torch.abs(element_vertices_1 - element_vertices_2).unsqueeze(1)
    faces_diff = torch.abs(element_faces_1 - element_faces_2).unsqueeze(1)
    extra_features = torch.cat([vertices_diff, faces_diff], dim  = 1)

    return _3DZD_vectors_1, _3DZD_vectors_2, extra_features

def read_inv(fn):
    vectors = []
    f = open(fn, 'r')
    for line in f:
        vectors.append(float(line.strip()))
    f.close()
    return vectors[1::]
def read_ply(fn):
    element_vertex = None
    element_face = None
    f = open(fn, 'r')
    for line in f:
        line = line.strip()
        if 'element vertex' in line:
            element_vertex = int(line.split()[-1])
        if 'element face' in line:
            element_face = int(line.split()[-1])
        if element_vertex != None and element_face != None:
            return [element_vertex, element_face]
    f.close()
    return [element_vertex, element_face]

def read_dataset(db_structures, atom_type):
    dataset = {}
    for struct in db_structures:
        if atom_type == 'mainchain':
            _3dzd = read_inv('data/' + struct + '_cacn.inv')
        else:
            _3dzd = read_inv('data/' + struct + '.inv')

        _vertex = read_ply('data/' + struct + '.ply')
        data = {}
        data['_3dzd'] = _3dzd
        data['vertex_face'] = _vertex

        dataset[struct] = data
    return dataset

# Main Function
def main():
    # Arguments Parsing
    global args
    args = parser.parse_args()

    cuda = args.cuda
    if cuda == 'true' and torch.cuda.is_available():
        cuda = True
        device_id = args.device_id
    else:
        cuda = False
        device_id = torch.device("cpu")
 
    model_type = 'neural_network'
    atom_type = args.atom_type
    model_name = 'SCOPe_FA_fold' if atom_type == 'fullatom' else 'SCOPe_MC_fold'

    print('model_type = {}'.format(model_type))
    if model_type == 'neural_network':
        print('Neural Network Model')
        if isfile('Best_models/' + model_name):
            if cuda:
                model = torch.load('Best_models/' + model_name)
                model.cuda(device_id)
            else:
                model = torch.load('Best_models/' + model_name,map_location=torch.device('cpu'))
                model.to(device_id)
            print('Best_models/' + model_name + ' Loaded and will be used for evaluation')
        else:
            print(model_name + ' Not found')
            exit()
    elif model_type == 'simple_euclidean_model':
        print('Simple Euclidean Model')
        model = SimpleEuclideanModel()
    model.eval()

    db_structures = [x for x in os.listdir('data/') if '.inv' in x and '_cacn' not in x]
    db_structures = [x.split('.')[0] for x in db_structures]
    print('pdb to compare : ',len(db_structures))
    if len(db_structures) == 0:
        print('There are no structure to compare.')
        exit()
    
    database_dataset = read_dataset(db_structures,atom_type)
    query_pdb = db_structures[0]
    my_pairs = [(query_pdb, j) for j in db_structures]
    inputs_1, inputs_2, extra_features = pairs_to_features(my_pairs,database_dataset,database_dataset)
    if cuda:
        inputs_1 = inputs_1.cuda(device_id)
        inputs_2 = inputs_2.cuda(device_id)
        extra_features = extra_features.cuda(device_id)

    
    outputs = model(inputs_1, inputs_2, extra_features, True)
    outputs = outputs.squeeze().cpu().data.numpy().tolist()    
    with open('output/'+ atom_type + '_prediction.txt','w') as fh:
        fh.write('Query\tTarget\tDis-similarity Probability\n')
        for i in range(0,len(outputs)):
            fh.write(db_structures[0] + '\t' + db_structures[i] + '\t' +str(round(outputs[i],3)) + '\n')


if __name__=="__main__":
    main()
