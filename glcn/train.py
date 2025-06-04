import time
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F

from utils import load_data, get_num_classes
from models import SGLCN
from metrics import masked_softmax_cross_entropy, masked_accuracy

# Mimicking TensorFlow FLAGS
class FLAGS:
    dataset = 'cora'  # 'cora', 'citeseer'
    model = 'sglcn'
    lr1 = 0.005  # Learning rate for Graph Learning Layer
    lr2 = 0.005  # Learning rate for Graph Convolution Layers
    epochs = 1000 # Original TF code: 10000, reduced for faster example runs
    hidden_gcn = 30 # Number of units in GCN hidden layer
    hidden_gl = 70  # Number of units in GraphLearning hidden layer (output_dim of SGL)
    dropout = 0.6   # Dropout rate
    weight_decay = 1e-4 # Weight for L2 loss (used for both SGL and GCN params)
    early_stopping = 100
    losslr1 = 0.01  # Coefficient for the smooth loss term in SGL (trace(X^T(I-S)X))
    losslr2 = 0.0001 # Coefficient for the frobenius norm term in SGL (-trace(S^T S))
    seed = 123
    use_cuda = True # Set to False if no GPU or to run on CPU

def compute_loss1_for_sgl(h_sgl, learned_S_sparse, sgl_params, N, device, flags_obj):
    """ Computes loss1 (Graph Learning specific loss) """
    # L2 Regularization for SGL parameters
    sgl_l2_loss = torch.tensor(0.0, device=device)
    for param in sgl_params:
        sgl_l2_loss += flags_obj.weight_decay * 0.5 * param.pow(2).sum()

    # Graph Learning loss terms
    # Original TF: D = tf.matrix_diag(tf.ones(N))*-1; D = tf.sparse_add(D, self.S)*-1 => D = I - S
    # loss1 += tf.trace(tf.matmul(tf.transpose(self.x), D)) * FLAGS.losslr1
    #         = trace(X^T (I-S) X) * losslr1
    # loss1 -= tf.trace(tf.sparse_tensor_dense_matmul(tf.sparse_transpose(self.S), tf.sparse_tensor_to_dense(self.S))) * FLAGS.losslr2
    #         = -trace(S^T S) * losslr2

    # For PyTorch, ensure S is dense for these trace ops or use sparse equivalents if available
    # learned_S_sparse should already be NxN
    eye_N = torch.eye(N, device=device)
    
    # Efficient way for trace(X^T M X) = sum_i (X_i^T M X_i) = sum_ij (M_ij * (X X^T)_ji)
    # Or (X^T @ M @ X).trace()
    # h_sgl is N x hidden_gl
    # learned_S_sparse is N x N
    
    # Term 1: trace(h_sgl^T (I - S) h_sgl)
    # If learned_S_sparse is very sparse, to_dense() might be an issue for large N.
    # However, N is number of nodes (e.g., ~2700 for Cora), hidden_gl is small (70).
    # SGL paper might have alternatives for large graphs. Here, we replicate TF.
    if learned_S_sparse.is_sparse:
        # This can be memory intensive if N is large
        learned_S_dense = learned_S_sparse.to_dense()
    else:
        learned_S_dense = learned_S_sparse

    I_minus_S = eye_N - learned_S_dense
    smooth_loss_term = torch.trace(h_sgl.T @ I_minus_S @ h_sgl) * flags_obj.losslr1
    
    # Term 2: -trace(S^T S)
    # frobenius_norm_sq_S = torch.trace(learned_S_dense.T @ learned_S_dense)
    # More directly for Frobenius norm squared:
    frobenius_norm_sq_S = (learned_S_dense**2).sum()
    frobenius_loss_term = -frobenius_norm_sq_S * flags_obj.losslr2
    
    total_loss1 = sgl_l2_loss + smooth_loss_term + frobenius_loss_term
    return total_loss1

def compute_loss2_for_gcn(logits, labels_one_hot, labels_mask, gcn_params, device, flags_obj):
    """ Computes loss2 (GCN specific classification loss) """
    # L2 Regularization for GCN parameters
    gcn_l2_loss = torch.tensor(0.0, device=device)
    for param in gcn_params:
        gcn_l2_loss += flags_obj.weight_decay * 0.5 * param.pow(2).sum()
        
    # Masked Cross Entropy classification loss
    classification_loss = masked_softmax_cross_entropy(logits, labels_one_hot, labels_mask)
    
    total_loss2 = gcn_l2_loss + classification_loss
    return total_loss2

def main(flags_obj):
    np.random.seed(flags_obj.seed)
    torch.manual_seed(flags_obj.seed)
    if flags_obj.use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed(flags_obj.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        flags_obj.use_cuda = False
    print(f"Using device: {device}")

    # Load data
    # adj_for_gcn_baseline is not used by SGLCN model internally, as it learns its own S.
    # edge_indices_for_sgl is used by SparseGraphLearn layer.
    _, features_sparse, labels_indices, idx_train, idx_val, idx_test, edge_indices_for_sgl = \
        load_data(flags_obj.dataset)
    
    num_nodes = features_sparse.shape[0]
    input_dim = features_sparse.shape[1]
    num_classes = get_num_classes(flags_obj.dataset)

    # Convert labels to one-hot for metric functions (original format)
    labels_one_hot = F.one_hot(labels_indices, num_classes=num_classes).float().to(device)
    
    # Move data to device
    features_sparse = features_sparse.to(device)
    edge_indices_for_sgl = edge_indices_for_sgl.to(device)
    # idx_train, idx_val, idx_test are already tensors, used for masking
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool, device=device)
    train_mask[idx_train] = True
    val_mask[idx_val] = True
    test_mask[idx_test] = True

    # Create model
    model = SGLCN(
        input_dim=input_dim,
        feature_hidden_dim=flags_obj.hidden_gl,
        gcn_hidden_dim=flags_obj.hidden_gcn,
        num_classes=num_classes,
        num_nodes=num_nodes,
        edge_indices_for_sgl=edge_indices_for_sgl,
        dropout_rate=flags_obj.dropout
    ).to(device)

    sgl_params = list(model.get_sgl_parameters())
    gcn_params = list(model.get_gcn_parameters())

    # Optimizers (mimicking TF's separate optimizers for different parts of the loss)
    optimizer_sgl = optim.Adam(sgl_params, lr=flags_obj.lr1)
    optimizer_gcn = optim.Adam(gcn_params, lr=flags_obj.lr2)
    
    # Schedulers (mimicking TF's exponential_decay)
    # decay_steps=100, decay_rate=0.9, staircase=True
    scheduler_sgl = optim.lr_scheduler.StepLR(optimizer_sgl, step_size=100, gamma=0.9)
    scheduler_gcn = optim.lr_scheduler.StepLR(optimizer_gcn, step_size=100, gamma=0.9)


    best_val_loss = float('inf')
    patience_counter = 0
    best_epoch = 0
    final_test_acc_at_best_val = 0.0

    # Training loop
    for epoch in range(flags_obj.epochs):
        t_epoch_start = time.time()
        model.train()

        # Forward pass
        # logits are final classification scores
        # h_transformed_features are XW from SGL layer (self.x in TF model)
        # learned_S_sparse is the graph learned by SGL layer (self.S in TF model)
        logits, h_transformed_features, learned_S_sparse = model(features_sparse)
        
        # --- Calculate Loss1 (for SGL parameters) ---
        loss1 = compute_loss1_for_sgl(
            h_transformed_features, learned_S_sparse, sgl_params, 
            num_nodes, device, flags_obj
        )
        
        # --- Calculate Loss2 (for GCN parameters) ---
        # Gradients from loss2 should not affect SGL parameters via optimizer_gcn.
        # The TF var_list ensures this. In PyTorch, using torch.autograd.grad provides explicit control.
        # Alternatively, if learned_S_sparse.detach() was used for GCN input, loss2.backward() wouldn't reach SGL params.
        # Here, we use torch.autograd.grad to isolate gradient application.
        loss2 = compute_loss2_for_gcn(
            logits, labels_one_hot, train_mask, gcn_params, device, flags_obj
        )
        
        # --- Backward and Optimize (carefully replicating TF's var_list behavior) ---
        # Optimizer for SGL layer (updates sgl_params based on loss1)
        optimizer_sgl.zero_grad()
        # Grads for SGL params from loss1
        grads_loss1_sgl = torch.autograd.grad(loss1, sgl_params, retain_graph=True, allow_unused=True)
        for param, grad in zip(sgl_params, grads_loss1_sgl):
            if grad is not None:
                param.grad = grad.clone() # Assign new grad
        optimizer_sgl.step()

        # Optimizer for GCN layers (updates gcn_params based on loss2)
        optimizer_gcn.zero_grad()
        # Grads for GCN params from loss2. Grads from loss2 to SGL params are computed but ignored by optimizer_gcn.
        # learned_S_sparse (and thus h_transformed_features if it were part of GCN input) creates the path from loss2 to SGL params.
        # torch.autograd.grad for loss2 on gcn_params will only return these specific gradients.
        grads_loss2_gcn = torch.autograd.grad(loss2, gcn_params, allow_unused=True)
        for param, grad in zip(gcn_params, grads_loss2_gcn):
            if grad is not None:
                param.grad = grad.clone() # Assign new grad
        optimizer_gcn.step()
        
        # Step schedulers
        scheduler_sgl.step()
        scheduler_gcn.step()

        # --- Evaluation ---
        model.eval()
        with torch.no_grad():
            logits_eval, _, _ = model(features_sparse) # Use the current model state for eval
            
            train_loss_eval = masked_softmax_cross_entropy(logits_eval, labels_one_hot, train_mask).item()
            train_acc_eval = masked_accuracy(logits_eval, labels_one_hot, train_mask).item()
            
            val_loss = masked_softmax_cross_entropy(logits_eval, labels_one_hot, val_mask).item()
            val_acc = masked_accuracy(logits_eval, labels_one_hot, val_mask).item()
            
            test_acc = masked_accuracy(logits_eval, labels_one_hot, test_mask).item()

        epoch_duration = time.time() - t_epoch_start
        print(f"Epoch: {epoch+1:04d} "
              f"train_loss={train_loss_eval:.5f} train_acc={train_acc_eval:.5f} | "
              f"val_loss={val_loss:.5f} val_acc={val_acc:.5f} | "
              f"test_acc={test_acc:.5f} | time={epoch_duration:.5f}s")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            final_test_acc_at_best_val = test_acc
            patience_counter = 0
            # Could save model checkpoint here if needed
            # torch.save(model.state_dict(), f"sglcn_{flags_obj.dataset}_best.pth")
        else:
            patience_counter += 1

        if patience_counter >= flags_obj.early_stopping:
            print("Early stopping triggered.")
            break
            
    print("Optimization Finished!")
    print(f"Best epoch: {best_epoch:04d} with validation loss: {best_val_loss:.5f}")
    print(f"Test accuracy at best validation epoch: {final_test_acc_at_best_val:.5f}")
    # The original TF code prints test_acc_list[-101], which corresponds to the test accuracy
    # from 100 epochs before early stopping (the epoch of best validation performance).
    # Our `final_test_acc_at_best_val` variable stores this.

if __name__ == '__main__':
    flags = FLAGS()
    # You can modify flags here, e.g., for different datasets:
    # flags.dataset = 'citeseer'
    # flags.lr1 = 0.001 # Example of changing a hyperparameter
    main(flags)