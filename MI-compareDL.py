import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, GATConv, TransformerConv
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from sklearn.metrics import roc_auc_score, accuracy_score, average_precision_score, recall_score, confusion_matrix, balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
import os
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.neighbors import kneighbors_graph
import json
import math
from sklearn.metrics.pairwise import cosine_similarity
from torch_geometric.utils import get_laplacian
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import eigsh

# --- Define device ---
device = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
print(f"Using device: {device}")

# --- Focal Loss ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean', pos_weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        # Ensure pos_weight is on the correct device when FocalLoss is initialized
        self.pos_weight = pos_weight.to(device) if pos_weight is not None else None

    def forward(self, inputs, targets):
        # Dynamically move pos_weight if its device doesn't match inputs
        if self.pos_weight is not None and self.pos_weight.device != inputs.device:
             self.pos_weight = self.pos_weight.to(inputs.device)

        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none', pos_weight=self.pos_weight)
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs) # pt should be on inputs.device
        
        # Create alpha_t on the same device as inputs
        alpha_scalar = self.alpha
        one_minus_alpha_scalar = 1 - self.alpha
        alpha_t = torch.where(targets == 1, 
                              torch.full_like(inputs, alpha_scalar, device=inputs.device), 
                              torch.full_like(inputs, one_minus_alpha_scalar, device=inputs.device))
        
        focal_weight = alpha_t * torch.pow(1 - pt, self.gamma)
        loss = focal_weight * BCE_loss
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# --- GNN Model ---
class GCNNet(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, dropout_rate=0.3):
        super(GCNNet, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_channels)
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.fc = nn.Linear(hidden_channels, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class GATNet(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, heads=4, dropout_rate=0.3):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(num_features, hidden_channels, heads=heads, dropout=dropout_rate, concat=True)
        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout_rate, concat=False)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.fc(x)
        return x

class GraphTransformerNet(nn.Module):
    def __init__(self, num_features, hidden_channels_per_head, num_classes, 
                 num_encoder_layers=3, heads_per_layer=4, dropout_rate=0.3):
        super(GraphTransformerNet, self).__init__()
        
        self.num_encoder_layers = num_encoder_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList() # Batchnorm for each conv layer's output
        self.dropouts = nn.ModuleList() # Optional: add dropout after activation

        current_dim = num_features

        for i in range(num_encoder_layers):
            is_last_layer_for_attention_stack = (i == num_encoder_layers - 1)
            
            current_heads = heads_per_layer
            concat_output = True

            if is_last_layer_for_attention_stack:
                self.convs.append(
                    TransformerConv(current_dim, hidden_channels_per_head, heads=current_heads,
                                    dropout=dropout_rate, concat=True)
                )
                output_dim_of_conv = hidden_channels_per_head * current_heads
            else:
                # Intermediate layers
                self.convs.append(
                    TransformerConv(current_dim, hidden_channels_per_head, heads=current_heads,
                                    dropout=dropout_rate, concat=True)
                )
                output_dim_of_conv = hidden_channels_per_head * current_heads

            self.bns.append(nn.BatchNorm1d(output_dim_of_conv))
            self.dropouts.append(nn.Dropout(dropout_rate)) # Dropout after activation
            current_dim = output_dim_of_conv # Input for the next layer

        self.fc = nn.Linear(current_dim, num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        for i in range(self.num_encoder_layers):
            x = self.convs[i](x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = self.dropouts[i](x) # Apply dropout after activation

        x = self.fc(x)
        return x
    
# --- Non-Graph Baselines ---
class LSTMNet(nn.Module):
    def __init__(self, num_features, hidden_channels, num_classes, num_lstm_layers=2, dropout_rate=0.3):
        super(LSTMNet, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_lstm_layers = num_lstm_layers
        
        # LSTM layer
        # batch_first=True means input/output tensors are (batch, seq, feature)
        # For node classification, each node is a "batch" item, and its feature vector is a "sequence" of length 1.
        self.lstm = nn.LSTM(input_size=num_features, 
                            hidden_size=hidden_channels,
                            num_layers=num_lstm_layers, 
                            batch_first=True, 
                            dropout=dropout_rate if num_lstm_layers > 1 else 0) # Dropout only between LSTM layers
        
        self.bn1 = nn.BatchNorm1d(hidden_channels)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_channels, num_classes)

    def forward(self, data):
        x = data.x # Shape: (num_nodes, num_features)
        
        # Reshape x for LSTM: (batch_size, seq_len, input_size)
        # Here, num_nodes is batch_size, seq_len is 1.
        x_lstm_input = x.unsqueeze(1) # Shape: (num_nodes, 1, num_features)
        
        # LSTM forward pass
        # h0 and c0 default to zeros if not provided
        lstm_out, _ = self.lstm(x_lstm_input) # lstm_out shape: (num_nodes, 1, hidden_channels)
        
        # We only need the output of the last time step (which is the only time step here)
        x = lstm_out[:, -1, :] # Shape: (num_nodes, hidden_channels)
        
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc(x) # Shape: (num_nodes, num_classes)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1) # Shape: (max_len, 1, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (seq_len, batch_size, d_model) for Transformer
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class VanillaTransformerNet(nn.Module): # Standard Transformer Encoder (Non-Graph)
    def __init__(self, num_features, hidden_channels, num_classes, nhead=4, num_encoder_layers=3, dim_feedforward=512, dropout_rate=0.3):
        super(VanillaTransformerNet, self).__init__()
        self.model_type = 'Transformer'
        
        # Input embedding layer (linear projection)
        self.input_fc = nn.Linear(num_features, hidden_channels)
        self.pos_encoder = PositionalEncoding(hidden_channels, dropout_rate) # Optional for non-sequential node data
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_channels, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            batch_first=False # Expects (S, N, E) if False, (N, S, E) if True. For nodes, S=num_nodes, N=1 is common
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_encoder_layers)
        
        self.bn1 = nn.BatchNorm1d(hidden_channels) # Apply after Transformer, before final FC
        self.fc = nn.Linear(hidden_channels, num_classes)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, data):
        x = data.x # Shape: (num_nodes, num_features)
        
        # Project input features to hidden_channels (embedding dimension for Transformer)
        x = self.input_fc(x) # Shape: (num_nodes, hidden_channels)
        
        # TransformerEncoder expects (seq_len, batch_size, embed_dim)
        # Here, we treat num_nodes as seq_len, and batch_size as 1.
        x = x.unsqueeze(1) # Shape: (num_nodes, 1, hidden_channels)
        
        # Add positional encoding (optional, can be experimented with)
        # x = self.pos_encoder(x) 
        
        # Pass through Transformer Encoder
        transformer_out = self.transformer_encoder(x) # Shape: (num_nodes, 1, hidden_channels)
        
        # Remove the batch_size=1 dimension
        x = transformer_out.squeeze(1) # Shape: (num_nodes, hidden_channels)

        # Post-transformer processing
        x = self.bn1(x) # Batchnorm on features of each node
        x = F.relu(x)   # Activation
        x = self.dropout(x) # Dropout
        x = self.fc(x) # Final classification layer. Shape: (num_nodes, num_classes)
        return x
    
# --- Data Preparation ---
myocardial_infarction_complications = fetch_ucirepo(id=579)
X_df_orig = myocardial_infarction_complications.data.features.copy()
y_df_orig = myocardial_infarction_complications.data.targets.copy()
target_names = list(y_df_orig.columns)

feature_collection_time = {
    'AGE': 'admission', 'SEX': 'admission', 'INF_ANAM': 'admission', 'STENOK_AN': 'admission',
    'FK_STENOK': 'admission', 'IBS_POST': 'admission', 'IBS_NASL': 'admission', 'GB': 'admission',
    'SIM_GIPERT': 'admission', 'DLIT_AG': 'admission', 'ZSN_A': 'admission', 'nr_11': 'admission',
    'nr_01': 'admission', 'nr_02': 'admission', 'nr_03': 'admission', 'nr_04': 'admission',
    'nr_07': 'admission', 'nr_08': 'admission', 'np_01': 'admission', 'np_04': 'admission',
    'np_05': 'admission', 'np_07': 'admission', 'np_08': 'admission', 'np_09': 'admission',
    'np_10': 'admission', 'endocr_01': 'admission', 'endocr_02': 'admission', 'endocr_03': 'admission',
    'zab_leg_01': 'admission', 'zab_leg_02': 'admission', 'zab_leg_03': 'admission',
    'zab_leg_04': 'admission', 'zab_leg_06': 'admission', 'S_AD_KBRIG': 'admission',
    'D_AD_KBRIG': 'admission', 'S_AD_ORIT': 'admission', 'D_AD_ORIT': 'admission',
    'O_L_POST': 'admission', 'K_SH_POST': 'admission', 'MP_TP_POST': 'admission',
    'SVT_POST': 'admission', 'GT_POST': 'admission', 'FIB_G_POST': 'admission',
    'ant_im': 'admission', 'lat_im': 'admission', 'inf_im': 'admission', 'post_im': 'admission',
    'IM_PG_P': 'admission', 'ritm_ecg_p_01': 'admission', 'ritm_ecg_p_02': 'admission',
    'ritm_ecg_p_04': 'admission', 'ritm_ecg_p_06': 'admission', 'ritm_ecg_p_07': 'admission',
    'ritm_ecg_p_08': 'admission', 'n_r_ecg_p_01': 'admission', 'n_r_ecg_p_02': 'admission',
    'n_r_ecg_p_03': 'admission', 'n_r_ecg_p_04': 'admission', 'n_r_ecg_p_05': 'admission',
    'n_r_ecg_p_06': 'admission', 'n_r_ecg_p_08': 'admission', 'n_r_ecg_p_09': 'admission',
    'n_r_ecg_p_10': 'admission', 'n_p_ecg_p_01': 'admission', 'n_p_ecg_p_03': 'admission',
    'n_p_ecg_p_04': 'admission', 'n_p_ecg_p_05': 'admission', 'n_p_ecg_p_06': 'admission',
    'n_p_ecg_p_07': 'admission', 'n_p_ecg_p_08': 'admission', 'n_p_ecg_p_09': 'admission',
    'n_p_ecg_p_10': 'admission', 'n_p_ecg_p_11': 'admission', 'n_p_ecg_p_12': 'admission',
    'fibr_ter_01': 'admission', 'fibr_ter_02': 'admission', 'fibr_ter_03': 'admission',
    'fibr_ter_05': 'admission', 'fibr_ter_06': 'admission', 'fibr_ter_07': 'admission',
    'fibr_ter_08': 'admission', 'GIPO_K': 'admission', 'K_BLOOD': 'admission',
    'GIPER_NA': 'admission', 'NA_BLOOD': 'admission', 'ALT_BLOOD': 'admission',
    'AST_BLOOD': 'admission', 'KFK_BLOOD': 'admission', 'L_BLOOD': 'admission',
    'ROE': 'admission', 'TIME_B_S': 'admission', 'NA_KB': 'admission', 'NOT_NA_KB': 'admission',
    'LID_KB': 'admission', 'NITR_S': 'admission', 'LID_S_n': 'admission',
    'B_BLOK_S_n': 'admission', 'ANT_CA_S_n': 'admission', 'GEPAR_S_n': 'admission',
    'ASP_S_n': 'admission', 'TIKL_S_n': 'admission', 'TRENT_S_n': 'admission',
}
admission_features = [f for f, t in feature_collection_time.items() if t == 'admission' and f in X_df_orig.columns]
print(f"Using {len(admission_features)} admission features.")

X_processed = X_df_orig[admission_features].copy()
for col in admission_features:
    if X_processed[col].isnull().any():
        X_processed[col].fillna(X_processed[col].median(), inplace=True)

y_processed = y_df_orig.copy()
for col in y_processed.columns:
    if y_processed[col].isnull().any():
        y_processed[col].fillna(y_processed[col].median(), inplace=True)
    y_processed[col] = y_processed[col].astype(int)

label_11_col_name = y_processed.columns[10]
y_processed[label_11_col_name] = (y_processed[label_11_col_name] > 0).astype(int)

num_patients = len(X_processed)
indices = np.arange(num_patients)
stratify_labels_global = (y_processed.sum(axis=1) > 0).astype(int)

train_val_indices, test_indices = train_test_split(
    indices, test_size=0.2, random_state=42, stratify=stratify_labels_global
)
train_indices, val_indices = train_test_split(
    train_val_indices, test_size=0.125, random_state=42,
    stratify=stratify_labels_global[train_val_indices]
)

scaler = StandardScaler()
X_scaled_np = scaler.fit_transform(X_processed)
X_scaled_tensor = torch.tensor(X_scaled_np, dtype=torch.float)
y_tensor = torch.tensor(y_processed.values, dtype=torch.float)

k_neighbors = 10
adj_matrix = kneighbors_graph(X_scaled_np, k_neighbors, mode='connectivity', include_self=False)
edge_index = torch.tensor(np.array(adj_matrix.nonzero()), dtype=torch.long)

train_mask = torch.zeros(num_patients, dtype=torch.bool)
val_mask = torch.zeros(num_patients, dtype=torch.bool)
test_mask = torch.zeros(num_patients, dtype=torch.bool)
train_mask[train_indices] = True
val_mask[val_indices] = True
test_mask[test_indices] = True

graph_data = Data(x=X_scaled_tensor, edge_index=edge_index, y=y_tensor,
                  train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
graph_data = graph_data.to(device)

y_train_on_device = graph_data.y[graph_data.train_mask]
num_positives_train = y_train_on_device.sum(axis=0)
num_train_samples_tensor = graph_data.train_mask.sum()
num_negatives_train = num_train_samples_tensor - num_positives_train
condition = num_positives_train > 0
ratio = torch.ones_like(num_positives_train, dtype=torch.float, device=device)
non_zero_mask = num_positives_train > 0
if non_zero_mask.any():
    ratio[non_zero_mask] = num_negatives_train[non_zero_mask] / num_positives_train[non_zero_mask]
pos_weight_tensor = torch.where(condition, ratio, torch.ones_like(condition, dtype=torch.float, device=device))
print("Calculated pos_weight for loss (based on training set):", pos_weight_tensor)
print(f"pos_weight_tensor device: {pos_weight_tensor.device}")


# --- Training and Evaluation ---
def train_step(model, data, criterion_instance, optimizer_instance):
    model.train()
    optimizer_instance.zero_grad()
    out = model(data)
    loss = criterion_instance(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer_instance.step()
    return loss.item()

def evaluate_model(model, data, criterion_instance, mask_name_str):
    model.eval()
    current_mask = getattr(data, mask_name_str)
    with torch.no_grad():
        out = model(data)
        if hasattr(criterion_instance, 'pos_weight') and criterion_instance.pos_weight is not None:
            if criterion_instance.pos_weight.device != out.device:
                criterion_instance.pos_weight = criterion_instance.pos_weight.to(out.device)
        
        loss = criterion_instance(out[current_mask], data.y[current_mask])
        preds_prob = torch.sigmoid(out[current_mask])
        labels = data.y[current_mask]

    avg_loss = loss.item()
    all_preds_prob_np = preds_prob.cpu().numpy()
    all_labels_np = labels.cpu().numpy()
    threshold = 0.5
    all_preds_binary_np = (all_preds_prob_np > threshold).astype(int)

    num_labels_total = all_labels_np.shape[1]
    metrics = {
        'per_label_acc': [], 'per_label_recall': [], 'per_label_specificity': [],
        'per_label_ap': [], 'per_label_roc_auc': [], 'per_label_balanced_acc': [],
        'confusion_matrices': []
    }

    for i in range(num_labels_total):
        true_1d = all_labels_np[:, i]
        prob_1d = all_preds_prob_np[:, i]
        pred_1d = all_preds_binary_np[:, i]
        unique_true = np.unique(true_1d)

        acc = accuracy_score(true_1d, pred_1d)
        metrics['per_label_acc'].append(acc)

        if len(unique_true) < 2:
            metrics['per_label_recall'].append(np.nan)
            metrics['per_label_specificity'].append(np.nan)
            metrics['per_label_ap'].append(np.nan)
            metrics['per_label_roc_auc'].append(np.nan)
            metrics['per_label_balanced_acc'].append(np.nan if np.isnan(acc) else acc)
            if unique_true[0] == 0: tn, fp, fn, tp = np.sum(pred_1d == 0), np.sum(pred_1d == 1), 0, 0
            else: tn, fp, fn, tp = 0, 0, np.sum(pred_1d == 0), np.sum(pred_1d == 1)
            metrics['confusion_matrices'].append((tn, fp, fn, tp))
            continue

        cm = confusion_matrix(true_1d, pred_1d, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        metrics['confusion_matrices'].append((tn, fp, fn, tp))
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        balanced_acc = (recall + specificity) / 2 if not (np.isnan(recall) or np.isnan(specificity)) else np.nan
        
        metrics['per_label_recall'].append(recall)
        metrics['per_label_specificity'].append(specificity)
        metrics['per_label_balanced_acc'].append(balanced_acc)

        try:
            ap = average_precision_score(true_1d, prob_1d)
            roc_auc = roc_auc_score(true_1d, prob_1d)
        except ValueError:
            ap, roc_auc = np.nan, np.nan
        metrics['per_label_ap'].append(ap)
        metrics['per_label_roc_auc'].append(roc_auc)
        
    return {
        'avg_loss': avg_loss,
        'mean_acc': np.nanmean(metrics['per_label_acc']),
        'per_label_acc': metrics['per_label_acc'],
        'mean_recall': np.nanmean(metrics['per_label_recall']),
        'per_label_recall': metrics['per_label_recall'],
        'mean_specificity': np.nanmean(metrics['per_label_specificity']),
        'per_label_specificity': metrics['per_label_specificity'],
        'mean_ap': np.nanmean(metrics['per_label_ap']),
        'per_label_ap': metrics['per_label_ap'],
        'mean_roc_auc': np.nanmean(metrics['per_label_roc_auc']),
        'per_label_roc_auc': metrics['per_label_roc_auc'],
        'mean_balanced_acc': np.nanmean(metrics['per_label_balanced_acc']),
        'per_label_balanced_acc': metrics['per_label_balanced_acc'],
        'confusion_matrices': metrics['confusion_matrices']
    }

# --- Save Metrics and Plot ---
os.makedirs('results', exist_ok=True)

def save_training_plots(backbone_name, train_metrics_log, val_metrics_log):
    os.makedirs(f'results/{backbone_name}', exist_ok=True)
    
    epochs_list = range(1, len(train_metrics_log) + 1)
    df_data = {
        'epoch': epochs_list,
        'train_loss': [m['avg_loss'] for m in train_metrics_log],
        'train_mean_acc': [m['mean_acc'] for m in train_metrics_log],
        'train_mean_recall': [m['mean_recall'] for m in train_metrics_log],
        'train_mean_roc_auc': [m['mean_roc_auc'] for m in train_metrics_log],
        'val_loss': [m['avg_loss'] for m in val_metrics_log],
        'val_mean_acc': [m['mean_acc'] for m in val_metrics_log],
        'val_mean_recall': [m['mean_recall'] for m in val_metrics_log],
        'val_mean_roc_auc': [m['mean_roc_auc'] for m in val_metrics_log],
    }
    metrics_df = pd.DataFrame(df_data)
    metrics_df.to_csv(f'results/{backbone_name}/training_metrics_detailed.csv', index=False)

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{backbone_name} Training Progress', fontsize=16)

    axs[0, 0].plot(metrics_df['epoch'], metrics_df['train_loss'], label='Train Loss', linestyle='--')
    axs[0, 0].plot(metrics_df['epoch'], metrics_df['val_loss'], label='Val Loss')
    axs[0, 0].set_title('Loss vs. Epoch'); axs[0, 0].set_xlabel('Epoch'); axs[0, 0].set_ylabel('Loss'); axs[0, 0].legend(); axs[0, 0].grid(True)

    axs[0, 1].plot(metrics_df['epoch'], metrics_df['train_mean_acc'], label='Train Mean Acc', linestyle='--')
    axs[0, 1].plot(metrics_df['epoch'], metrics_df['val_mean_acc'], label='Val Mean Acc')
    axs[0, 1].set_title('Mean Accuracy vs. Epoch'); axs[0, 1].set_xlabel('Epoch'); axs[0, 1].set_ylabel('Accuracy'); axs[0, 1].legend(); axs[0, 1].grid(True)

    axs[1, 0].plot(metrics_df['epoch'], metrics_df['train_mean_recall'], label='Train Mean Recall', linestyle='--')
    axs[1, 0].plot(metrics_df['epoch'], metrics_df['val_mean_recall'], label='Val Mean Recall')
    axs[1, 0].set_title('Mean Recall vs. Epoch'); axs[1, 0].set_xlabel('Epoch'); axs[1, 0].set_ylabel('Recall'); axs[1, 0].legend(); axs[1, 0].grid(True)

    axs[1, 1].plot(metrics_df['epoch'], metrics_df['train_mean_roc_auc'], label='Train Mean ROC-AUC', linestyle='--')
    axs[1, 1].plot(metrics_df['epoch'], metrics_df['val_mean_roc_auc'], label='Val Mean ROC-AUC')
    axs[1, 1].set_title('Mean ROC-AUC vs. Epoch'); axs[1, 1].set_xlabel('Epoch'); axs[1, 1].set_ylabel('ROC-AUC'); axs[1, 1].legend(); axs[1, 1].grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f'results/{backbone_name}/train_val_curves_mpl.png', dpi=300)
    plt.close(fig)

    chart_datasets = [
        {"label": "Train Loss", "data": list(metrics_df['train_loss']), "borderColor": "#1f77b4", "fill": False, "yAxisID": "y-loss"},
        {"label": "Val Loss", "data": list(metrics_df['val_loss']), "borderColor": "#aec7e8", "fill": False, "yAxisID": "y-loss"},
        {"label": "Train Mean ROC-AUC", "data": list(metrics_df['train_mean_roc_auc']), "borderColor": "#2ca02c", "fill": False, "yAxisID": "y-metric"},
        {"label": "Val Mean ROC-AUC", "data": list(metrics_df['val_mean_roc_auc']), "borderColor": "#98df8a", "fill": False, "yAxisID": "y-metric"},
    ]
    chart_config = {
        "type": "line", "data": {"labels": list(metrics_df['epoch'].astype(str)), "datasets": chart_datasets},
        "options": {"responsive": True, "maintainAspectRatio": False, "interaction": {"mode": "index", "intersect": False},
                    "title": {"display": True, "text": f"{backbone_name} Training Curves (Loss & AUC)"},
                    "scales": {"x": {"title": {"display": True, "text": "Epoch"}},
                               "y-loss": {"type": "linear", "display": True, "position": "left", "title": {"display": True, "text": "Loss"}},
                               "y-metric": {"type": "linear", "display": True, "position": "right", "title": {"display": True, "text": "ROC-AUC"}, "grid": {"drawOnChartArea": False }}
                    }}}
    with open(f'results/{backbone_name}/train_val_curves_chartjs.json', 'w') as f:
        json.dump(chart_config, f, indent=4)


def plot_combined_confusion_matrices(backbone_name, test_metrics_dict, target_names_list, num_cols=4):
    cm_dir = f'results/{backbone_name}/confusion_matrices_combined'
    os.makedirs(cm_dir, exist_ok=True)
    
    num_labels = len(test_metrics_dict['confusion_matrices'])
    num_rows = (num_labels + num_cols - 1) // num_cols
    
    fig = plt.figure(figsize=(num_cols * 4, num_rows * 3.5))
    gs = gridspec.GridSpec(num_rows, num_cols, figure=fig, hspace=0.4, wspace=0.3)

    for i, (tn, fp, fn, tp) in enumerate(test_metrics_dict['confusion_matrices']):
        ax = fig.add_subplot(gs[i // num_cols, i % num_cols])
        cm_array = np.array([[tn, fp], [fn, tp]])
        sns.heatmap(cm_array, annot=True, fmt='d', cmap='Blues', cbar=False, 
                    xticklabels=['Pred Neg', 'Pred Pos'], 
                    yticklabels=['True Neg', 'True Pos'], ax=ax, annot_kws={"size": 8})
        safe_target_name = target_names_list[i].replace("/", "_").replace(" ", "_")
        ax.set_title(f'{target_names_list[i]}\n({backbone_name})', fontsize=9)
        ax.set_ylabel('True Label', fontsize=8)
        ax.set_xlabel('Predicted Label', fontsize=8)
        ax.tick_params(axis='both', which='major', labelsize=7)
    
    for i in range(num_labels, num_rows * num_cols):
        fig.delaxes(fig.axes[i])
        
    plt.savefig(f'{cm_dir}/all_cms_combined_{backbone_name}.png', dpi=300)
    plt.close(fig)

def plot_performance_bar_chartjs(all_backbone_results):
    backbones = list(all_backbone_results.keys())
    datasets_dict = {
        "Mean ROC-AUC": {"data": [all_backbone_results[b]['mean_roc_auc'] for b in backbones], "backgroundColor": "rgba(255, 99, 132, 0.7)"},
        "Mean Recall": {"data": [all_backbone_results[b]['mean_recall'] for b in backbones], "backgroundColor": "rgba(54, 162, 235, 0.7)"},
        "Mean AP": {"data": [all_backbone_results[b]['mean_ap'] for b in backbones], "backgroundColor": "rgba(255, 206, 86, 0.7)"},
        "Mean Specificity": {"data": [all_backbone_results[b]['mean_specificity'] for b in backbones], "backgroundColor": "rgba(75, 192, 192, 0.7)"},
        "Mean Accuracy": {"data": [all_backbone_results[b]['mean_acc'] for b in backbones], "backgroundColor": "rgba(153, 102, 255, 0.7)"},
        "Mean Balanced Acc": {"data": [all_backbone_results[b]['mean_balanced_acc'] for b in backbones], "backgroundColor": "rgba(255, 159, 64, 0.7)"}
    }
    chart_config = {"type": "bar", "data": {"labels": backbones, "datasets": [{"label": k, **v} for k,v in datasets_dict.items()]},
                    "options": {"responsive": True, "maintainAspectRatio": False, 
                                "plugins": {"title": {"display": True, "text": "Backbone Performance Comparison"},
                                            "legend": {"position": "top"}},
                                "scales": {"y": {"beginAtZero": True, "title": {"display": True, "text": "Score"}}}}}
    with open('results/overall_performance_summary_chartjs.json', 'w') as f:
        json.dump(chart_config, f, indent=4)

# --- Main Training Loop ---
num_node_features = graph_data.num_node_features
num_output_classes = graph_data.y.size(1)

backbone_configs = {
    'GCN': {
        'model': GCNNet,
        'params': {
            'hidden_channels': 128, 
            'dropout_rate': 0.5
        }
    },
    'GAT': {
        'model': GATNet, 
        'params': {
            'hidden_channels': 64, 
            'heads': 4, 
            'dropout_rate': 0.6
        }
    },
    'GraphTransformer': {
        'model': GraphTransformerNet,
        'params': {
            'hidden_channels_per_head': 32,
            'num_encoder_layers': 3,
            'heads_per_layer': 4,
            'dropout_rate': 0.4,
        }
    },
    'LSTM': {
        'model': LSTMNet, 
        'params': {
            'hidden_channels': 128, 
            'num_lstm_layers': 2, 
            'dropout_rate': 0.4
        }
    },
    'VanillaTransformer': {
        'model': VanillaTransformerNet, 
        'params': {
            'hidden_channels': 128, 
            'nhead': 4, 
            'num_encoder_layers': 3, 
            'dim_feedforward': 256, 
            'dropout_rate': 0.4
        }
    }
}
all_backbone_test_results = {}

epochs = 300
early_stopping_patience = 50

for backbone_name, config in backbone_configs.items():
    print(f"\n--- Training {backbone_name} ---")
    model_class = config['model']
    model_params = config['params']
    model = model_class(num_features=num_node_features, num_classes=num_output_classes, **model_params).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    criterion = FocalLoss(alpha=0.25, gamma=2.0, reduction='mean', pos_weight=pos_weight_tensor).to(device)
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=15, verbose=True)

    train_metrics_log = []
    val_metrics_log = []
    best_val_auc_score = -1.0
    current_patience = 0
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        train_loss_value = train_step(model, graph_data, criterion, optimizer)
        
        train_epoch_metrics = evaluate_model(model, graph_data, criterion, 'train_mask')
        train_epoch_metrics['avg_loss'] = train_loss_value
        train_metrics_log.append(train_epoch_metrics)

        val_epoch_metrics = evaluate_model(model, graph_data, criterion, 'val_mask')
        val_metrics_log.append(val_epoch_metrics)

        current_val_auc = val_epoch_metrics["mean_roc_auc"] if not np.isnan(val_epoch_metrics["mean_roc_auc"]) else -1
        scheduler.step(current_val_auc)

        print(f'Epoch: {epoch:03d}/{epochs} | {backbone_name} | Train Loss: {train_loss_value:.4f} | Val Loss: {val_epoch_metrics["avg_loss"]:.4f} | Val AUC: {current_val_auc:.4f} | Val Bal.Acc: {val_epoch_metrics["mean_balanced_acc"]:.4f}')

        if current_val_auc > best_val_auc_score:
            best_val_auc_score = current_val_auc
            best_epoch = epoch
            current_patience = 0
            torch.save(model.state_dict(), f'results/{backbone_name}/best_model_weights.pth')
            print(f'    Best Val AUC: {best_val_auc_score:.4f} at epoch {best_epoch}. Model saved.')
        else:
            current_patience += 1
            if current_patience >= early_stopping_patience:
                print(f'--- Early stopping for {backbone_name} at epoch {epoch} (Patience: {early_stopping_patience})---')
                break
    
    save_training_plots(backbone_name, train_metrics_log, val_metrics_log)

    print(f"\n--- Evaluating {backbone_name} on Test Set (Best model from epoch {best_epoch}) ---")
    best_model_path = f'results/{backbone_name}/best_model_weights.pth'
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
    else:
        print(f"Warning: Best model not found for {backbone_name}. Evaluating with last model state.")

    test_result_dict = evaluate_model(model, graph_data, criterion, 'test_mask')
    all_backbone_test_results[backbone_name] = test_result_dict
    
    plot_combined_confusion_matrices(backbone_name, test_result_dict, target_names)

    print(f'\n--- Test Set Performance for {backbone_name} ---')
    print(f'  Best Validation AUC during training: {best_val_auc_score:.4f} (at epoch {best_epoch})')
    print(f'  Test Loss: {test_result_dict["avg_loss"]:.4f}')
    print(f'  Mean Accuracy: {test_result_dict["mean_acc"]:.4f}')
    print(f'  Mean Balanced Accuracy: {test_result_dict["mean_balanced_acc"]:.4f}')
    print(f'  Mean Recall (Sensitivity): {test_result_dict["mean_recall"]:.4f}')
    print(f'  Mean Specificity: {test_result_dict["mean_specificity"]:.4f}')
    print(f'  Mean ROC-AUC: {test_result_dict["mean_roc_auc"]:.4f}')
    print(f'  Mean AP (Precision-Recall AUC): {test_result_dict["mean_ap"]:.4f}')

    print(f'\n--- Per-Label Test Performance for {backbone_name} ---')
    for i in range(num_output_classes):
        label_name = target_names[i] if i < len(target_names) else f"Label_{i}"
        acc_i = test_result_dict['per_label_acc'][i] if i < len(test_result_dict['per_label_acc']) else np.nan
        bal_acc_i = test_result_dict['per_label_balanced_acc'][i] if i < len(test_result_dict['per_label_balanced_acc']) else np.nan
        rec_i = test_result_dict['per_label_recall'][i] if i < len(test_result_dict['per_label_recall']) else np.nan
        spec_i = test_result_dict['per_label_specificity'][i] if i < len(test_result_dict['per_label_specificity']) else np.nan
        roc_auc_i = test_result_dict['per_label_roc_auc'][i] if i < len(test_result_dict['per_label_roc_auc']) else np.nan
        ap_i = test_result_dict['per_label_ap'][i] if i < len(test_result_dict['per_label_ap']) else np.nan
        print(f'  {label_name + ":":<20} Acc: {acc_i:.4f}, Bal.Acc: {bal_acc_i:.4f} Rec: {rec_i:.4f}, Spec: {spec_i:.4f}, AUC: {roc_auc_i:.4f}, AP: {ap_i:.4f}')


performance_summary_list = []
for bn_name, metrics in all_backbone_test_results.items():
    performance_summary_list.append({
        'Backbone': bn_name, 'Mean Accuracy': metrics['mean_acc'], 'Mean Balanced Acc': metrics['mean_balanced_acc'],
        'Mean Recall': metrics['mean_recall'], 'Mean Specificity': metrics['mean_specificity'],
        'Mean ROC-AUC': metrics['mean_roc_auc'], 'Mean AP': metrics['mean_ap'], 'Test Loss': metrics['avg_loss']
    })
performance_summary_df = pd.DataFrame(performance_summary_list)
performance_summary_df.to_csv('results/overall_performance_summary.csv', index=False)
print("\nOverall performance summary saved to results/overall_performance_summary.csv")
print(performance_summary_df)

if all_backbone_test_results:
    plot_performance_bar_chartjs(all_backbone_test_results)
    print("Overall performance bar chart (Chart.js config) saved to results/overall_performance_summary_chartjs.json")

print("\n--- Script Finished ---")