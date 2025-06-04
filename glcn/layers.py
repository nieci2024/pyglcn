import torch
import torch.nn as nn
import torch.nn.functional as F
from inits import glorot_init, zeros_init # Assuming inits_pytorch is in the same directory

def sparse_dropout(x, p, training):
    """Dropout for sparse tensors.
    Applies dropout to the non-zero values of a sparse tensor.
    """
    if not training or p == 0.0:
        return x
    if not x.is_sparse: # Should not happen if called correctly
        return F.dropout(x, p=p, training=training)
    
    # MODIFICATION: Ensure tensor is coalesced before accessing values
    x_coalesced = x.coalesce()
    
    values = x_coalesced.values()
    # F.dropout scales by 1/(1-p) during training
    new_values = F.dropout(values, p=p, training=training) 
    
    return torch.sparse_coo_tensor(x_coalesced.indices(), new_values, x_coalesced.size())


class SparseGraphLearn(nn.Module):
    """Sparse Graph learning layer."""
    def __init__(self, input_dim, output_dim, edge_indices, num_nodes, dropout_rate=0.0, act=F.relu, bias=False):
        super(SparseGraphLearn, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.edge_indices = edge_indices 
        self.num_nodes = num_nodes
        self.dropout_rate = dropout_rate
        self.act = act
        self.bias_flag = bias

        self.weights = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        glorot_init(self.weights)

        self.a = nn.Parameter(torch.FloatTensor(output_dim, 1)) # For scoring edge features
        glorot_init(self.a)

        if self.bias_flag:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
            zeros_init(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, x_sparse): 
        # Dropout on input features
        # x_sparse_dropped will be coalesced by sparse_dropout_pytorch if dropout is applied
        x_sparse_dropped = sparse_dropout(x_sparse, self.dropout_rate, self.training)
        
        h = torch.sparse.mm(x_sparse_dropped, self.weights) 
        if self.bias is not None:
            h = h + self.bias # Broadcasting bias might not work directly if h is sparse after mm.
                              # However, h = sparse_mm(sparse, dense) is dense.

        h_i = h[self.edge_indices[0]] 
        h_j = h[self.edge_indices[1]] 
        
        edge_features = torch.abs(h_i - h_j) 
        
        edge_scores = self.act(edge_features @ self.a).squeeze() 
        
        # Potentially add self-loops to edge_indices and edge_scores before creating learned_S
        # if the learned graph should also consider self-connections explicitly from this stage.
        # The original TF SGL layer doesn't seem to add self-loops explicitly at this point,
        # it relies on the input edge_indices.

        # Create sparse graph S. Indices must be coalesced for sparse_softmax.
        # The edge_indices should ideally not have duplicates here for clean softmax,
        # or torch.sparse_coo_tensor will sum them.
        # Let's ensure edge_indices for learned_S are unique or handled correctly by softmax
        learned_S_uncoalesced = torch.sparse_coo_tensor(
            self.edge_indices, 
            edge_scores, 
            torch.Size([self.num_nodes, self.num_nodes])
        )
        learned_S = learned_S_uncoalesced.coalesce() # Ensure S is coalesced before softmax
        
        learned_S_softmax = torch.sparse.softmax(learned_S, dim=1)

        return h, learned_S_softmax

class GraphConvolution(nn.Module):
    """Graph convolution layer (Kipf & Welling)."""
    def __init__(self, input_dim, output_dim, dropout_rate=0.0, act=F.relu, bias=False, sparse_inputs=False):
        super(GraphConvolution, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        self.act = act
        self.bias_flag = bias
        self.sparse_inputs = sparse_inputs 

        self.weights = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        glorot_init(self.weights)

        if self.bias_flag:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
            zeros_init(self.bias)
        else:
            self.register_parameter('bias', None)

    def forward(self, x, adj_sparse): 
        # x_dropped will be coalesced by sparse_dropout_pytorch if sparse_inputs and dropout are active
        if self.sparse_inputs:
            x_dropped = sparse_dropout(x, self.dropout_rate, self.training)
        else: # x is dense
            x_dropped = F.dropout(x, self.dropout_rate, training=self.training)
        
        if self.sparse_inputs:
            support = torch.sparse.mm(x_dropped, self.weights)
        else:
            support = torch.mm(x_dropped, self.weights)
        
        # adj_sparse should be coalesced before mm for performance/correctness
        # Assuming adj_sparse (e.g. learned_S_softmax) is already coalesced
        output = torch.sparse.mm(adj_sparse.coalesce(), support) 
        
        if self.bias is not None:
            output = output + self.bias
            
        return self.act(output)