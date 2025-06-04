import torch
import torch.nn.functional as F

def masked_softmax_cross_entropy(preds, labels_one_hot, mask):
    """
    Softmax cross-entropy loss with masking (preds are logits).
    labels_one_hot: N x C one-hot tensor
    mask: N boolean tensor
    """
    # Convert one-hot labels to class indices for PyTorch's CrossEntropyLoss
    labels_indices = labels_one_hot.max(dim=1)[1] # Get indices of max value (true class)
    
    loss = F.cross_entropy(preds, labels_indices, reduction='none') # N
    mask = mask.float()
    mask = mask / mask.mean() # Normalize mask as in TF code
    loss = loss * mask
    return loss.mean()

def masked_accuracy(preds, labels_one_hot, mask):
    """
    Accuracy with masking (preds are logits).
    labels_one_hot: N x C one-hot tensor
    mask: N boolean tensor
    """
    # Convert one-hot labels to class indices
    labels_indices = labels_one_hot.max(dim=1)[1]
    
    preds_indices = preds.max(dim=1)[1] # Get predicted class indices
    correct_prediction = torch.eq(preds_indices, labels_indices).float() # N, boolean
    
    mask = mask.float()
    mask = mask / mask.mean() # Normalize mask
    correct_prediction = correct_prediction * mask
    return correct_prediction.mean()