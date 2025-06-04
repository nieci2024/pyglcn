import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import SparseGraphLearn, GraphConvolution

class SGLCN(nn.Module):
    def __init__(self, input_dim, feature_hidden_dim, gcn_hidden_dim, num_classes, 
                 num_nodes, edge_indices_for_sgl, dropout_rate):
        super(SGLCN, self).__init__()
        self.input_dim = input_dim
        self.num_nodes = num_nodes
        self.edge_indices_for_sgl = edge_indices_for_sgl # Edges for the SGL layer to learn from
        
        # SparseGraphLearn layer
        self.sgl_layer = SparseGraphLearn(
            input_dim=input_dim,
            output_dim=feature_hidden_dim, # FLAGS.hidden_gl
            edge_indices=self.edge_indices_for_sgl,
            num_nodes=self.num_nodes,
            dropout_rate=dropout_rate, # Applied to input features within SGL
            act=F.relu,
            bias=False # Original SGL layer doesn't seem to use bias for h=XW but GCN does
        )

        # GraphConvolution layers
        # Layer 1: Takes original features and learned graph S
        self.gcn_layer1 = GraphConvolution(
            input_dim=input_dim, # Original features
            output_dim=gcn_hidden_dim, # FLAGS.hidden_gcn
            dropout_rate=dropout_rate,
            act=F.relu,
            bias=False, # Original GCN layers in SGLCN don't use bias (bias=False default in tf layer)
            sparse_inputs=True # Original features are sparse
        )

        # Layer 2: Takes output of GCN1 and learned graph S
        self.gcn_layer2 = GraphConvolution(
            input_dim=gcn_hidden_dim,
            output_dim=num_classes,
            dropout_rate=dropout_rate,
            act=lambda x: x, # Linear activation for logits
            bias=False,
            sparse_inputs=False # Input from GCN1 is dense
        )

    def forward(self, x_sparse_features):
        # x_sparse_features: initial sparse feature matrix (N x input_dim)
        
        # Graph Learning Layer
        # h_transformed_features are XW from the SGL layer
        # learned_S is the new adjacency matrix (sparse) learned by SGL
        h_transformed_features, learned_S = self.sgl_layer(x_sparse_features)
        
        # GCN Layers
        # GCN layer 1 uses the *original* sparse features and the learned graph S
        x_gcn1 = self.gcn_layer1(x_sparse_features, learned_S)
        
        # GCN layer 2 uses the output of GCN1 and the learned graph S
        logits = self.gcn_layer2(x_gcn1, learned_S)
        
        return logits, h_transformed_features, learned_S

    def get_sgl_parameters(self):
        return self.sgl_layer.parameters()

    def get_gcn_parameters(self):
        return list(self.gcn_layer1.parameters()) + list(self.gcn_layer2.parameters())