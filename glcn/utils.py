import numpy as np
import scipy.sparse as sp
import scipy.io as sio
import torch
import torch.nn.functional as F # Added for F.one_hot if needed elsewhere, though not directly in this snippet


def load_data(dataset_str, path="../data/"):
    """
    Loads input data from data directory (adapted for PyTorch).
    :param dataset_str: Dataset name ("cora", "citeseer")
    :param path: Path to data directory
    :return: adj, features, labels, idx_train, idx_val, idx_test
    (all as PyTorch tensors where appropriate, adj as sparse COO)
    """
    data_path = path + dataset_str + "/"
    print(f"Loading {dataset_str} dataset from {data_path}...")

    features_data = sio.loadmat(data_path + "feature.mat")
    features = sp.csr_matrix(features_data['matrix'], dtype=np.float32)

    labels_data = sio.loadmat(data_path + "label.mat")
    labels = torch.LongTensor(np.where(labels_data['matrix'])[1]) # Convert one-hot to class indices

    adj_data = sio.loadmat(data_path + "adj.mat")
    adj = sp.coo_matrix(adj_data['matrix'], dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    if dataset_str == "cora":
        idx_train = torch.LongTensor(range(140))
        idx_val = torch.LongTensor(range(200, 500))
        idx_test = torch.LongTensor(range(500, 1500))
    elif dataset_str == "citeseer":
        try:
            idx_test_data = sio.loadmat(data_path + "test.mat") # Original TF code expects 'test.mat' with 'array'
            # Ensure flattening if it's a column/row vector
            test_indices_flat = idx_test_data['array'].flatten()
            idx_test = torch.LongTensor(test_indices_flat)
        except FileNotFoundError:
            print(f"Warning: {data_path}test.mat not found for Citeseer. Using default split logic.")
            # Fallback logic similar to original TF if test.mat is not present or specific key not found
            num_nodes_cs = adj.shape[0] # Typically 3327 for Citeseer
            idx_train = torch.LongTensor(range(120))
            idx_val = torch.LongTensor(range(120, 620)) # 500 validation nodes
            # The original TF code uses a specific test.mat. If not available, use a placeholder split.
            # Example: use a range of indices after train/val, ensuring it's within bounds.
            # This part highly depends on how Planetoid splits are handled if test.mat is missing.
            # For robustness, ensure test indices don't overlap and are valid.
            # A common Citeseer test set size is 1000.
            idx_test = torch.LongTensor(range(num_nodes_cs - 1000, num_nodes_cs))
            print(f"Using fallback test indices for Citeseer: {num_nodes_cs - 1000} to {num_nodes_cs-1}")


        # Default train/val from original code for citeseer if not overridden by specific file handling
        if 'idx_train' not in locals(): # If not set by specific logic above
            idx_train = torch.LongTensor(range(120))
        if 'idx_val' not in locals():
            idx_val = torch.LongTensor(range(120, 620))
    else:
        raise ValueError(f"Dataset {dataset_str} not supported.")

    features = normalize_features(features) # Returns a sparse tensor

    adj_normalized_for_gcn_baseline, edge_index_for_sgl = preprocess_adj(adj)


    return adj_normalized_for_gcn_baseline, features, labels, idx_train, idx_val, idx_test, edge_index_for_sgl

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix (D^-0.5 * A * D^-0.5)"""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    adj_normalized = adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)
    return sparse_mx_to_torch_sparse_tensor(adj_normalized.tocoo())

def preprocess_adj(adj):
    """
    Preprocessing of adjacency matrix for GCN model and conversion to PyTorch sparse tensor.
    Also extracts edge_index for the SparseGraphLearn layer.
    """
    adj_plus_selfloop = adj + sp.eye(adj.shape[0], dtype=np.float32)
    adj_normalized_scipy = normalize_adj_scipy(adj_plus_selfloop)
    
    edge_index = torch.LongTensor(np.array(adj_normalized_scipy.nonzero()))

    return sparse_mx_to_torch_sparse_tensor(adj_normalized_scipy.tocoo()), edge_index

def normalize_adj_scipy(adj):
    """Symmetrically normalize adjacency matrix (scipy version)."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def normalize_features(mx):
    """Row-normalize sparse matrix and convert to PyTorch sparse tensor"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return sparse_mx_to_torch_sparse_tensor(mx)

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    # MODIFICATION: Coalesce the tensor upon creation
    return torch.sparse_coo_tensor(indices, values, shape).coalesce()

def get_num_classes(dataset_str):
    if dataset_str.lower() == 'cora':
        return 7
    elif dataset_str.lower() == 'citeseer':
        return 6
    raise ValueError(f"Dataset {dataset_str} not recognized for num_classes.")