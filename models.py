import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

#from utils import pearsonr

class SimpleEuclideanModel(nn.Module):
    def __init__(self):
        super(SimpleEuclideanModel, self).__init__()

    def forward(self, inputs_1, inputs_2, extra_features, output_dist = False):
        euclidean_dist = torch.norm(inputs_1 - inputs_2, 2, dim=-1).squeeze()
        if output_dist:
            return euclidean_dist
        return 1.0 / (1.0 + euclidean_dist)

class NeuralNetworkModel(nn.Module):
    def __init__(self, input_dim, hidden_dims, fc_dims, extra_feature_dim):
        super(NeuralNetworkModel, self).__init__()

        self.encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            self.encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            self.encoder_layers.append(nn.BatchNorm1d(hidden_dim))
            self.encoder_layers.append(nn.ReLU(True))
            prev_dim = hidden_dim
        self.encoder = nn.ModuleList(self.encoder_layers)
        self.nb_encoder_layers = len(hidden_dims)

        self.fc_layers = []
        prev_dim = 2 * (input_dim + sum(hidden_dims)) + \
                   2 * (1 + self.nb_encoder_layers) + \
                   extra_feature_dim
        #prev_dim = 4744 # 1444# 4744 #1444
        for fc_dim in fc_dims:
            self.fc_layers.append(nn.Linear(prev_dim, fc_dim))
            self.fc_layers.append(nn.BatchNorm1d(fc_dim))
            self.fc_layers.append(nn.ReLU(True))
            self.fc_layers.append(nn.Dropout(0.2))
            prev_dim = fc_dim
        self.fc_layers_module = nn.ModuleList(self.fc_layers)

        self.last_fc = nn.Linear(prev_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs_1, inputs_2, extra_features, output_dist = False):
        #print(inputs_1.shape)
        outs_1 = []
        out = inputs_1
        outs_1.append(out)
        for layer in self.encoder:
            out = layer(out)
            if 'ReLU' in str(layer):
                outs_1.append(out)

        # Encoding inputs_2
        outs_2 = []
        out = inputs_2
        outs_2.append(out)
        for layer in self.encoder:
            out = layer(out)
            if 'ReLU' in str(layer):
                outs_2.append(out)

        # Creating feature_vectors
        feature_vectors = []
        for i in range(self.nb_encoder_layers + 1):
            euclidean_dist = torch.norm(outs_1[i] - outs_2[i], 2, dim=-1, keepdim=True)
            cosine_similarity = nn.CosineSimilarity(dim=-1, eps=1e-6)(outs_1[i], outs_2[i]).unsqueeze(1)
            subtraction = torch.abs(outs_1[i] - outs_2[i])
            multiplication = outs_1[i] * outs_2[i]
            feature_vectors.append(euclidean_dist)
            feature_vectors.append(cosine_similarity)
            feature_vectors.append(subtraction)
            feature_vectors.append(multiplication)
        feature_vectors = torch.cat(feature_vectors, dim=-1)
        feature_vectors = torch.cat([feature_vectors, extra_features], dim=-1)

        # FC Layers
        out = feature_vectors
        for layer in self.fc_layers_module:
            out = layer(out)
        out = self.last_fc(out)
        if output_dist:
            return 1.0 - self.sigmoid(out)
        return self.sigmoid(out)
