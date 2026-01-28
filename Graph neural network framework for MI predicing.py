import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import remove_self_loops, to_undirected, coalesce
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from torch.nn import MultiheadAttention  # For Cross-Modal Attention

class DynamicGatingLayer(nn.Module):
    """
    Enhanced dynamic gating with non-linear interaction, inspired by Mamba's selective updates.
    The 'h' input is projected to 'feature_dim' for addition and interaction.
    """
    def __init__(self, feature_dim, hidden_dim):  # feature_dim for x, hidden_dim for h
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        self.gate_linear_x = nn.Linear(feature_dim, feature_dim)  # Wx

        # Project h to feature_dim for addition with Wx and for element-wise product with x
        self.project_h_to_feature_dim = nn.Linear(hidden_dim, feature_dim)  # Uh' (projects h for various uses)

        # Interaction, compression shorts—nah, tank top and shorts for gym. Wait, code: tank showing off the 64-dim short_term, but actually:
        self.project_h_to_feature_dim = nn.Linear(hidden_dim, feature_dim)  # Uh' (projects h for various uses)

        # Interaction MLP: input is x * projected_h (element-wise), so input_dim is feature_dim
        self.gate_interact_mlp = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2 if feature_dim // 2 > 0 else 1),  # Ensure intermediate dim > 0
            nn.ReLU(),
            nn.Linear(feature_dim // 2 if feature_dim // 2 > 0 else 1, feature_dim)
        )

    def forward(self, x, h):
        # x: (batch, seq_len, feature_dim)
        # h: (batch, hidden_dim)

        gate_x_comp = self.gate_linear_x(x)  # (B, seq_len, feature_dim)

        # Project h to feature_dim
        h_projected = self.project_h_to_feature_dim(h).unsqueeze(1)  # (B, 1, feature_dim)
                                                                    # Unsqueeze to allow broadcasting over seq_len of x

        # Component for direct addition (like Uh)
        gate_h_comp = h_projected  # (B, 1, feature_dim)

        # Interaction component
        # Element-wise product between x and h_projected (broadcasts h_projected along seq_len)
        interact_product = x * h_projected  # (B, seq_len, feature_dim)
        gate_interact_comp = self.gate_interact_mlp(interact_product)  # (B, seq_len, feature_dim)

        gate_values = torch.sigmoid(gate_x_comp + gate_h_comp + gate_interact_comp)
        return x * gate_values

    def compute_modality_attention(self, x, h):
        # Simple attention: project x and h, compute scores
        projected_x = self.gate_linear_x(x)  # (B, seq_len, feature_dim)
        projected_h = self.project_h_to_feature_dim(h).unsqueeze(1)  # (B, 1, feature_dim)
        attn_scores = torch.softmax(projected_x + projected_h, dim=-1)  # (B, seq_len, feature_dim)
        return attn_scores  # Can be averaged or used for interpretability

class ShortTermCNN(nn.Module):
    """
    Optimized CNN with multi-scale kernels. Residual connection removed for simplification.
    """
    def __init__(self, input_features_per_step, cnn_out_channels=64, num_cnn_layers=4):
        super().__init__()
        layers = []
        current_channels = input_features_per_step
        for i in range(num_cnn_layers):
            kernel_size = 2 if i % 2 == 0 else 4
            # Simplified padding: (kernel_size - 1) // 2 for odd, kernel_size // 2 for even (might be asymmetric)
            # For a robust 'same' padding, consider custom padding or ensure input length allows it.
            # PyTorch Conv1d padding is (left_pad, right_pad) if tuple, or symmetric if int.
            # kernel_size // 2 often works well enough for adaptive pooling later.
            padding_val = kernel_size // 2
            layers.append(nn.Conv1d(current_channels, cnn_out_channels, kernel_size, padding=padding_val))
            layers.append(nn.BatchNorm1d(cnn_out_channels))
            layers.append(nn.ReLU())
            current_channels = cnn_out_channels
        self.cnn_layers = nn.Sequential(*layers)
        self.global_max_pool = nn.AdaptiveMaxPool1d(1)
        self.output_dim = cnn_out_channels

    def forward(self, x):
        # x shape: (batch, num_time_steps, num_temporal_features_per_step)
        x = x.permute(0, 2, 1)  # (batch, num_temporal_features_per_step, num_time_steps)
        x = self.cnn_layers(x)  # (batch, cnn_out_channels, processed_time_steps)
        x = self.global_max_pool(x)  # (batch, cnn_out_channels, 1)
        return x.squeeze(-1)  # (batch, cnn_out_channels)

# Shared graph generation function (remains unchanged from your provided code)
def generate_optimized_knn_graph_shared(x_for_graph, k_min, k_max, sim_threshold):
    if torch.isnan(x_for_graph).any():
        x_for_graph = torch.nan_to_num(x_for_graph, nan=0.0)
    num_nodes_in_batch = x_for_graph.size(0)
    if num_nodes_in_batch <= 1:
        return torch.empty((2, 0), dtype=torch.long, device=x_for_graph.device)
    features_np = x_for_graph.detach().cpu().numpy()
    if np.isnan(features_np).any():
        features_np = np.nan_to_num(features_np, nan=0.0)
    sim_matrix = cosine_similarity(features_np)
    sim_matrix[sim_matrix < sim_threshold] = 0
    np.fill_diagonal(sim_matrix, 0)
    m = min(5, num_nodes_in_batch - 1)
    if m <= 0:
        k_values = np.full(num_nodes_in_batch, (k_min + k_max) // 2, dtype=int)
    else:
        sorted_sims_indices = np.argsort(-sim_matrix, axis=1)[:, :m]
        density_vals = []
        for i in range(num_nodes_in_batch):
            node_top_m_sims = sim_matrix[i, sorted_sims_indices[i]]
            valid_sims = node_top_m_sims[node_top_m_sims > 0]
            density_vals.append(np.mean(valid_sims) if len(valid_sims) > 0 else 0.0)
        density = np.array(density_vals)
        density_min, density_max = np.min(density), np.max(density)
        if density_max == density_min or density_max < 1e-9:
            k_values = np.full(density.shape, (k_min + k_max) // 2, dtype=int)
        else:
            k_values = k_min + (k_max - k_min) * (density - density_min) / (density_max - density_min)
        k_values = np.round(k_values).astype(int)
        k_values = np.clip(k_values, k_min, max(1, k_max))
    row, col = [], []
    for i in range(num_nodes_in_batch):
        current_k = k_values[i]
        if current_k == 0:
            continue
        node_sims = sim_matrix[i]
        positive_sim_indices = np.where(node_sims > 0)[0]
        if len(positive_sim_indices) == 0:
            continue
        sorted_positive_sim_indices = positive_sim_indices[np.argsort(-node_sims[positive_sim_indices])]
        num_neighbors_to_take = min(current_k, len(sorted_positive_sim_indices))
        top_k_actual_indices = sorted_positive_sim_indices[:num_neighbors_to_take]
        if len(top_k_actual_indices) > 0:
            row.extend([i] * len(top_k_actual_indices))
            col.extend(top_k_actual_indices)
    if not row:
        return torch.empty((2, 0), dtype=torch.long, device=x_for_graph.device)
    edge_index = torch.tensor([row, col], dtype=torch.long, device=x_for_graph.device)
    edge_index, _ = remove_self_loops(edge_index)
    edge_index = to_undirected(edge_index, num_nodes=num_nodes_in_batch)
    edge_index = coalesce(edge_index)
    return edge_index

class CrossModalAttention(nn.Module):
    """
    Lightweight cross-attention between short-term and long-term embeddings.
    Treats short-term as query, long-term as key/value (or vice versa).
    """
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.cross_attn = MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, short_term_emb, long_term_emb):
        # short_term_emb: (B, short_dim) -> reshape to (B, 1, short_dim) for seq_len=1
        # long_term_emb: (B, long_dim) -> assume short_dim == long_dim after projection
        short_term_emb = short_term_emb.unsqueeze(1)  # (B, 1, dim)
        long_term_emb = long_term_emb.unsqueeze(1)    # (B, 1, dim)
        
        attn_output, attn_weights = self.cross_attn(short_term_emb, long_term_emb, long_term_emb)
        attn_output = attn_output.squeeze(1)  # (B, dim)
        return self.norm(attn_output + short_term_emb.squeeze(1)), attn_weights  # Residual + weights for interpretability

class GNNForMI(nn.Module):
    def __init__(self,
                 num_total_features,
                 num_admission_features,
                 num_temporal_features_per_step,  # Crucial for correct initialization
                 num_time_steps,
                 indices_for_graph_construction,
                 short_term_cnn_out_channels=64,  # Output of ShortTermCNN
                 short_term_cnn_layers=4,
                 long_term_gru_hidden_dim=16,     # Hidden dim of long-term GRU
                 long_term_gru_input_dim=16,      # Input dim for long-term GRU & its gate's x
                 long_term_gru_num_layers=1,
                 temporal_embedding_dim=32,       # Final temporal embedding dim
                 hidden_channels_per_head=16,
                 num_classes=12,
                 num_encoder_layers=3,
                 heads_per_layer=4,
                 dropout_rate=0.3,
                 k_min=5,
                 k_max=15,
                 sim_threshold=0.5):
        super().__init__()

        self.num_encoder_layers = num_encoder_layers
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.k_min = k_min
        self.k_max = k_max
        self.sim_threshold_base = sim_threshold
        self.indices_for_graph_construction = indices_for_graph_construction
        self.cached_edge_index = None
        self.cached_x_for_graph_id = None
        self.cached_num_nodes = -1

        self.num_admission_features = num_admission_features
        self.num_temporal_features_per_step = num_temporal_features_per_step  # Store for use in forward
        self.num_time_steps = num_time_steps
        self.total_raw_temporal_features = num_temporal_features_per_step * num_time_steps
        self.num_derived_features_for_long_term = long_term_gru_input_dim

        self.num_static_features = (num_admission_features - self.total_raw_temporal_features) + 10

        # --- Temporal Processing ---
        # 1. Short-term branch (CNN + Gating)
        self.short_term_processor = ShortTermCNN(
            input_features_per_step=num_temporal_features_per_step,
            cnn_out_channels=short_term_cnn_out_channels,
            num_cnn_layers=short_term_cnn_layers
        )
        # Gate for raw temporal features, using CNN output as 'h'
        self.short_term_gate = DynamicGatingLayer(
            feature_dim=num_temporal_features_per_step,  # 'x' is raw temporal, its feature_dim
            hidden_dim=short_term_cnn_out_channels       # 'h' is CNN output, its feature_dim
        )
        # Projection to make gated_short_term.mean() compatible with short_term_processor output for addition
        self.projection_gated_to_cnn_dim = nn.Linear(num_temporal_features_per_step, short_term_cnn_out_channels)
        
        processed_short_term_dim = short_term_cnn_out_channels  # The dimension after ShortTermCNN and the sum

        # 2. Long-term branch (GRU + Gating)
        self.long_term_dynamic_gate = DynamicGatingLayer(
            feature_dim=self.num_derived_features_for_long_term,  # 'x' is derived features
            hidden_dim=long_term_gru_hidden_dim                   # 'h' is GRU hidden state
        )
        self.long_term_gru = nn.GRU(
            input_size=self.num_derived_features_for_long_term,
            hidden_size=long_term_gru_hidden_dim,
            num_layers=long_term_gru_num_layers,
            batch_first=True
        )
        processed_long_term_dim = long_term_gru_hidden_dim  # Dimension from GRU

        # Projection for long-term to match short-term dim if needed
        self.long_term_proj = nn.Linear(processed_long_term_dim, processed_short_term_dim) if processed_long_term_dim != processed_short_term_dim else None

        # Cross-Modal Attention (based on short_term_dim)
        self.cross_modal_attn = CrossModalAttention(processed_short_term_dim, num_heads=4) 

        # --- Fusion and Graph Transformer Layers ---
        combined_temporal_processed_dim = processed_short_term_dim + processed_short_term_dim  # After projection, both are short_dim
        if combined_temporal_processed_dim > 0:
            self.temporal_projector = nn.Linear(combined_temporal_processed_dim, temporal_embedding_dim)
            fused_feature_dim = self.num_static_features + temporal_embedding_dim
        else:  # Should not happen if temporal processing is active
            self.temporal_projector = None
            fused_feature_dim = self.num_static_features

        self.input_weight = nn.Linear(fused_feature_dim, fused_feature_dim, bias=False)
        current_dim = fused_feature_dim
        for i in range(num_encoder_layers):
            self.convs.append(
                TransformerConv(current_dim, hidden_channels_per_head, heads=heads_per_layer,
                                dropout=dropout_rate, concat=True)
            )
            output_dim_of_conv = hidden_channels_per_head * heads_per_layer
            self.bns.append(nn.BatchNorm1d(output_dim_of_conv))
            self.dropouts.append(nn.Dropout(dropout_rate))
            current_dim = output_dim_of_conv

        self.fc = nn.Linear(current_dim, num_classes)
        self.gru_dropout = nn.Dropout(0.2)  # General dropout for temporal embeddings
        self.use_cross_attn = True  # For ablation

    def generate_optimized_knn_graph(self, x_for_graph, dynamic_threshold=True):  # Defaulting to dynamic as in your code
        current_sim_threshold = self.sim_threshold_base
        if dynamic_threshold and x_for_graph.numel() > 0 and x_for_graph.size(0) > 1:  # Ensure valid input for density calc
            try:
                # Detach, move to CPU, convert to numpy, handle potential NaNs
                features_np = x_for_graph.detach().cpu().numpy()
                features_np = np.nan_to_num(features_np)  # Ensure NaNs are handled before cosine_similarity
                
                if features_np.shape[0] > 1 and features_np.shape[1] > 0:  # Need at least 2 samples and some features
                    sim_matrix = cosine_similarity(features_np)
                    # Calculate density more robustly
                    valid_densities = []
                    for i in range(len(sim_matrix)):
                        row_sims = sim_matrix[i, sim_matrix[i] > 0]  # Consider only positive similarities
                        if len(row_sims) > 0:
                            valid_densities.append(np.mean(row_sims))
                    
                    if valid_densities:
                        density = np.mean(valid_densities)
                        current_sim_threshold = max(0.3, min(0.7, self.sim_threshold_base + (density - 0.5) * 0.2))
                    # else: density could not be calculated, use base threshold
            except Exception:  # Catch any error during density calculation
                pass  # Fallback to base threshold
                
        return generate_optimized_knn_graph_shared(x_for_graph, self.k_min, self.k_max, current_sim_threshold)

    def forward(self, data):
        x_original = data.x
        num_nodes_in_batch = x_original.shape[0]

        if num_nodes_in_batch == 0:  # Handle empty batch early
            return torch.empty((0, self.fc.out_features), device=x_original.device)

        len_sap = self.num_admission_features - self.total_raw_temporal_features
        x_sap = x_original[:, :len_sap]
        x_d10 = x_original[:, -10:]
        x_static = torch.cat([x_sap, x_d10], dim=-1)

        short_term_embedding = None
        long_term_embedding = None
        interpret_dict = {}  # For storing attention weights for interpretability

        # Short-term processing
        if self.short_term_processor and self.total_raw_temporal_features > 0:
            idx_rt_start = len_sap
            idx_rt_end = len_sap + self.total_raw_temporal_features
            x_temporal_raw_flat = x_original[:, idx_rt_start:idx_rt_end]
            x_temporal_raw_reshaped = x_temporal_raw_flat.view(
                num_nodes_in_batch, self.num_time_steps, self.num_temporal_features_per_step
            )
            short_term_processed_cnn = self.short_term_processor(x_temporal_raw_reshaped)  # (B, cnn_out_channels)
            
            # Use short_term_processed_cnn as 'h' for the gate
            gated_short_term_raw = self.short_term_gate(x_temporal_raw_reshaped, short_term_processed_cnn)  # (B, T, num_temp_feat_step)
            
            gated_short_term_mean = gated_short_term_raw.mean(dim=1)  # (B, num_temp_feat_step)
            
            # Project mean of gated raw features to match CNN output dimension
            gated_short_term_projected = self.projection_gated_to_cnn_dim(gated_short_term_mean)  # (B, cnn_out_channels)
            
            # Combine CNN output with (projected) gated raw features
            combined_short_term = short_term_processed_cnn + gated_short_term_projected
            short_term_embedding = self.gru_dropout(combined_short_term)  # (B, cnn_out_channels)

            # Modality-specific attention for short-term
            short_modality_attn = self.short_term_gate.compute_modality_attention(x_temporal_raw_reshaped, short_term_processed_cnn)
            interpret_dict['short_modality_attn'] = short_modality_attn

        # Long-term processing
        if self.long_term_gru and self.num_derived_features_for_long_term > 0:
            idx_dgru_start = -(10 + self.num_derived_features_for_long_term)
            idx_dgru_end = -10
            x_derived_flat = x_original[:, idx_dgru_start:idx_dgru_end]
            x_derived_reshaped = x_derived_flat.view(num_nodes_in_batch, 1, self.num_derived_features_for_long_term)  # seq_len is 1
            
            # First pass through GRU to get initial hidden state for gating
            _, h_n_initial = self.long_term_gru(x_derived_reshaped)  # h_n_initial: (num_layers, B, H_long)
            
            # Gate the input using the GRU's hidden state
            gated_derived_input = self.long_term_dynamic_gate(x_derived_reshaped, h_n_initial[-1])  # Use last layer's h
            
            # Second pass through GRU with the gated input
            _, h_n_final = self.long_term_gru(gated_derived_input, h_n_initial)  # Optionally pass h_n_initial as h_0
            
            long_term_processed = h_n_final[-1]  # Use last layer's h from final pass
            long_term_embedding = self.gru_dropout(long_term_processed)

            # Modality-specific attention for long-term
            long_modality_attn = self.long_term_dynamic_gate.compute_modality_attention(x_derived_reshaped, h_n_final[-1])
            interpret_dict['long_modality_attn'] = long_modality_attn

        # Cross-Modal Attention if both embeddings exist
        if short_term_embedding is not None and long_term_embedding is not None and self.cross_modal_attn is not None and self.use_cross_attn:
            if self.long_term_proj is not None:
                long_term_embedding = self.long_term_proj(long_term_embedding)
            short_term_embedding, cross_attn_weights = self.cross_modal_attn(short_term_embedding, long_term_embedding)
            interpret_dict['cross_attn_weights'] = cross_attn_weights
            # Optionally apply to long_term_embedding if bidirectional attention is desired

        # Combine temporal embeddings
        temporal_embeddings_to_cat = []
        if short_term_embedding is not None:
            temporal_embeddings_to_cat.append(short_term_embedding)
        if long_term_embedding is not None:
            temporal_embeddings_to_cat.append(long_term_embedding)

        if len(temporal_embeddings_to_cat) > 0:
            combined_temporal_features = torch.cat(temporal_embeddings_to_cat, dim=-1)
            if self.temporal_projector is not None:
                final_temporal_embedding = F.relu(self.temporal_projector(combined_temporal_features))
                x_fused = torch.cat([x_static, final_temporal_embedding], dim=-1)
            else:  # Should not happen if combined_temporal_features is not empty
                x_fused = x_static  # Fallback, though projector should exist
        else:
            x_fused = x_static

        # Graph Construction and GNN layers
        x_for_graph_construction = x_original[:, self.indices_for_graph_construction]
        current_x_for_graph_id = id(x_for_graph_construction)

        if (self.cached_edge_index is not None and
            self.cached_num_nodes == num_nodes_in_batch and
            self.cached_x_for_graph_id == current_x_for_graph_id):
            edge_index = self.cached_edge_index
        else:
            edge_index = self.generate_optimized_knn_graph(x_for_graph_construction, dynamic_threshold=True)
            self.cached_edge_index = edge_index
            self.cached_x_for_graph_id = current_x_for_graph_id
            self.cached_num_nodes = num_nodes_in_batch

        x = self.input_weight(x_fused)
        for i in range(self.num_encoder_layers):
            if x.size(0) > 0:  # Ensure nodes exist before calling GNN layers or fc
                x = self.convs[i](x, edge_index)  # 直接调用 TransformerConv
                if x.size(0) > 1:  # Batchnorm requires more than 1 sample
                    x = self.bns[i](x)
                x = F.relu(x)
                x = self.dropouts[i](x)
            else:
                pass  # x 已经是空的，会传递给 self.fc(x)
        
        logits = self.fc(x)
        return logits, interpret_dict  # Return logits and interpretability dict
