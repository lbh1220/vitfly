"""
@authors: Modified for AirSim dataset with Graph Attention Network
@organization: GRASP Lab, University of Pennsylvania
@date: ...
@license: ...

@brief: This module contains model architectures for AirSim dataset training with GAT-based traffic processing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import sys, os
from os.path import join as opj
import numpy as np

# Import ViT submodules
sys.path.append(opj(os.path.dirname(os.path.abspath(__file__))))
from ViTsubmodules import *

def refine_inputs(X):
    # if input depth images are not of right shape, resize
    if X[0].shape[-2] != 60 or X[0].shape[-1] != 90:
        X[0] = F.interpolate(X[0], size=(60, 90), mode='bilinear')
    return X

class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer implementation
    """
    def __init__(self, input_dim, output_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        
        assert output_dim % num_heads == 0, "output_dim must be divisible by num_heads"
        
        self.W_q = nn.Linear(input_dim, output_dim, bias=False)
        self.W_k = nn.Linear(input_dim, output_dim, bias=False)
        self.W_v = nn.Linear(input_dim, output_dim, bias=False)
        self.W_o = nn.Linear(output_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(output_dim)
        
    def forward(self, node_features, adjacency_mask=None):
        """
        Args:
            node_features: (batch_size, num_nodes, input_dim)
            adjacency_mask: (batch_size, num_nodes, num_nodes) - optional
        Returns:
            output: (batch_size, num_nodes, output_dim)
        """
        batch_size, num_nodes, _ = node_features.shape
        
        # Compute queries, keys, values
        Q = self.W_q(node_features)  # (B, N, output_dim)
        K = self.W_k(node_features)  # (B, N, output_dim)
        V = self.W_v(node_features)  # (B, N, output_dim)
        
        # Reshape for multi-head attention
        Q = Q.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        K = K.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        V = V.view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)  # (B, H, N, d)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)  # (B, H, N, N)
        
        # Apply adjacency mask if provided
        if adjacency_mask is not None:
            adjacency_mask = adjacency_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            scores = scores.masked_fill(adjacency_mask == 0, -1e9)
        
        # Apply softmax
        attention_weights = F.softmax(scores, dim=-1)  # (B, H, N, N)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        out = torch.matmul(attention_weights, V)  # (B, H, N, d)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.output_dim)  # (B, N, output_dim)
        out = self.W_o(out)
        
        # Residual connection and layer norm
        if self.input_dim == self.output_dim:
            out = self.layer_norm(out + node_features)
        else:
            out = self.layer_norm(out)
        
        return out

class LSTMNetVIT_GAT(nn.Module):
    """
    ViT+GAT+LSTM Network for 2D navigation with traffic awareness
    """
    def __init__(self, 
                 robot_state_dim=4,      # [goal_x, goal_y, yaw, desired_vel]
                 traffic_state_dim=5,    # [rel_x, rel_y, vel_x, vel_y, radius]
                 num_traffic_agents=5,   # Number of traffic agents
                 embed_dim=64,           # Embedding dimension for robot and traffic states
                 gat_hidden_dim=128,     # GAT hidden dimension
                 gat_num_heads=4,        # Number of attention heads
                 gat_num_layers=2):      # Number of GAT layers
        super().__init__()
        
        self.robot_state_dim = robot_state_dim
        self.traffic_state_dim = traffic_state_dim
        self.num_traffic_agents = num_traffic_agents
        self.embed_dim = embed_dim
        
        # ViT encoder blocks for image processing
        self.encoder_blocks = nn.ModuleList([
            MixTransformerEncoderLayer(1, 32, patch_size=7, stride=4, padding=3, n_layers=2, reduction_ratio=8, num_heads=1, expansion_factor=8),
            MixTransformerEncoderLayer(32, 64, patch_size=3, stride=2, padding=1, n_layers=2, reduction_ratio=4, num_heads=2, expansion_factor=8)
        ])
        
        self.decoder = spectral_norm(nn.Linear(4608, 512))
        
        # Robot state encoder
        self.robot_encoder = nn.Sequential(
            spectral_norm(nn.Linear(robot_state_dim, embed_dim)),
            nn.ReLU(),
            nn.Dropout(0.1),
            spectral_norm(nn.Linear(embed_dim, embed_dim))
        )
        
        # Traffic state encoder  
        self.traffic_encoder = nn.Sequential(
            spectral_norm(nn.Linear(traffic_state_dim, embed_dim)),
            nn.ReLU(),
            nn.Dropout(0.1),
            spectral_norm(nn.Linear(embed_dim, embed_dim))
        )
        
        # Graph Attention Network layers
        self.gat_layers = nn.ModuleList([
            GraphAttentionLayer(embed_dim, gat_hidden_dim, num_heads=gat_num_heads) if i == 0 
            else GraphAttentionLayer(gat_hidden_dim, gat_hidden_dim, num_heads=gat_num_heads)
            for i in range(gat_num_layers)
        ])
        
        # Final projection for graph features
        self.graph_projection = spectral_norm(nn.Linear(gat_hidden_dim, embed_dim))
        
        # Feature fusion and normalization
        fusion_input_size = 512 + embed_dim * 2  # image_features + robot_features + graph_features
        self.feature_fusion_norm = nn.LayerNorm(fusion_input_size)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(input_size=fusion_input_size, hidden_size=128, num_layers=3, dropout=0.1)
        
        # Output layer (2D velocity commands, normalized by desired_vel during training)
        self.output_layer = spectral_norm(nn.Linear(128, 2))

        # ViT supporting layers
        self.up_sample = nn.Upsample(size=(16,24), mode='bilinear', align_corners=True)
        self.pxShuffle = nn.PixelShuffle(upscale_factor=2)
        self.down_sample = nn.Conv2d(48,12,3, padding=1)

    def create_adjacency_mask(self, batch_size, num_nodes):
        """
        Create adjacency mask for fully connected graph
        Args:
            batch_size: batch size
            num_nodes: total number of nodes (1 robot + N traffic agents)
        Returns:
            adjacency_mask: (batch_size, num_nodes, num_nodes)
        """
        # For now, create fully connected graph (all nodes connected to all nodes)
        adjacency_mask = torch.ones(batch_size, num_nodes, num_nodes, 
                                   device=next(self.parameters()).device)
        return adjacency_mask

    def forward(self, X):
        """
        Forward pass
        Args:
            X: [images, robot_states, traffic_states, hidden_state(optional)]
        Returns:
            output: (batch_size, 2) - velocity commands
            hidden_state: LSTM hidden state
        """
        X = refine_inputs(X)
        
        # Extract inputs
        images = X[0]                    # (batch_size, 1, H, W)
        robot_states = X[1]              # (batch_size, 4)
        traffic_states = X[2]            # (batch_size, 5, 5)
        hidden_state = X[3] if len(X) > 3 else None
        
        batch_size = images.shape[0]
        
        # Process images with ViT
        embeds = [images]
        for block in self.encoder_blocks:
            embeds.append(block(embeds[-1]))        
        out = embeds[1:]
        out = torch.cat([self.pxShuffle(out[1]), self.up_sample(out[0])], dim=1) 
        out = self.down_sample(out)
        image_features = self.decoder(out.flatten(1))  # (batch_size, 512)
        
        # Encode robot states
        robot_features = self.robot_encoder(robot_states)  # (batch_size, embed_dim)
        
        # Encode traffic states
        traffic_features = self.traffic_encoder(traffic_states.view(-1, self.traffic_state_dim))  # (batch_size * 5, embed_dim)
        traffic_features = traffic_features.view(batch_size, self.num_traffic_agents, self.embed_dim)  # (batch_size, 5, embed_dim)
        
        # Create graph: robot node + traffic nodes
        robot_node = robot_features.unsqueeze(1)  # (batch_size, 1, embed_dim)
        graph_nodes = torch.cat([robot_node, traffic_features], dim=1)  # (batch_size, 6, embed_dim)
        
        # Store original robot features for residual connection
        original_robot_features = robot_features.clone()
        
        # Apply Graph Attention Network
        adjacency_mask = self.create_adjacency_mask(batch_size, self.num_traffic_agents + 1)
        
        graph_output = graph_nodes
        for gat_layer in self.gat_layers:
            graph_output = gat_layer(graph_output, adjacency_mask)
        
        # Extract robot node features after GAT processing
        processed_robot_features = graph_output[:, 0, :]  # (batch_size, gat_hidden_dim)
        processed_robot_features = self.graph_projection(processed_robot_features)  # (batch_size, embed_dim)
        
        # Feature fusion: image + original_robot + processed_robot
        fused_features = torch.cat([
            image_features,           # (batch_size, 512)
            original_robot_features,  # (batch_size, embed_dim)
            processed_robot_features  # (batch_size, embed_dim)
        ], dim=1)  # (batch_size, 512 + embed_dim * 2)
        
        # Apply layer normalization
        normalized_features = self.feature_fusion_norm(fused_features)
        
        # LSTM processing
        lstm_input = normalized_features.unsqueeze(0)  # (1, batch_size, fusion_input_size)
        
        if hidden_state is not None:
            lstm_output, new_hidden_state = self.lstm(lstm_input, hidden_state)
        else:
            lstm_output, new_hidden_state = self.lstm(lstm_input)
        
        lstm_output = lstm_output.squeeze(0)  # (batch_size, 128)
        
        # Generate output (2D velocity commands)
        output = self.output_layer(lstm_output)  # (batch_size, 2)
        
        return output, new_hidden_state

class LSTMNetVIT_NoTraffic_2D(nn.Module):
    """
    ViT+LSTM Network without traffic information for 2D navigation
    For comparison purposes
    """
    def __init__(self, robot_state_dim=4, embed_dim=64):
        super().__init__()
        
        self.robot_state_dim = robot_state_dim
        self.embed_dim = embed_dim
        
        # ViT encoder blocks for image processing
        self.encoder_blocks = nn.ModuleList([
            MixTransformerEncoderLayer(1, 32, patch_size=7, stride=4, padding=3, n_layers=2, reduction_ratio=8, num_heads=1, expansion_factor=8),
            MixTransformerEncoderLayer(32, 64, patch_size=3, stride=2, padding=1, n_layers=2, reduction_ratio=4, num_heads=2, expansion_factor=8)
        ])

        self.decoder = spectral_norm(nn.Linear(4608, 512))
        
        # Robot state encoder
        self.robot_encoder = nn.Sequential(
            spectral_norm(nn.Linear(robot_state_dim, embed_dim)),
            nn.ReLU(),
            nn.Dropout(0.1),
            spectral_norm(nn.Linear(embed_dim, embed_dim))
        )
        
        # Feature fusion and normalization
        fusion_input_size = 512 + embed_dim  # image_features + robot_features
        self.feature_fusion_norm = nn.LayerNorm(fusion_input_size)
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(input_size=fusion_input_size, hidden_size=128, num_layers=3, dropout=0.1)
        
        # Output layer (2D velocity commands, normalized by desired_vel during training)
        self.output_layer = spectral_norm(nn.Linear(128, 2))

        # ViT supporting layers
        self.up_sample = nn.Upsample(size=(16,24), mode='bilinear', align_corners=True)
        self.pxShuffle = nn.PixelShuffle(upscale_factor=2)
        self.down_sample = nn.Conv2d(48,12,3, padding=1)

    def forward(self, X):
        """
        Forward pass
        Args:
            X: [images, robot_states, hidden_state(optional)]
        Returns:
            output: (batch_size, 2) - velocity commands
            hidden_state: LSTM hidden state
        """
        X = refine_inputs(X)
        
        # Extract inputs
        images = X[0]                    # (batch_size, 1, H, W)
        robot_states = X[1]              # (batch_size, 4)
        hidden_state = X[2] if len(X) > 2 else None
        
        # Process images with ViT
        embeds = [images]
        for block in self.encoder_blocks:
            embeds.append(block(embeds[-1]))        
        out = embeds[1:]
        out = torch.cat([self.pxShuffle(out[1]), self.up_sample(out[0])], dim=1) 
        out = self.down_sample(out)
        image_features = self.decoder(out.flatten(1))  # (batch_size, 512)
        
        # Encode robot states
        robot_features = self.robot_encoder(robot_states)  # (batch_size, embed_dim)
        
        # Feature fusion
        fused_features = torch.cat([image_features, robot_features], dim=1)
        
        # Apply layer normalization
        normalized_features = self.feature_fusion_norm(fused_features)
        
        # LSTM processing
        lstm_input = normalized_features.unsqueeze(0)  # (1, batch_size, fusion_input_size)
        
        if hidden_state is not None:
            lstm_output, new_hidden_state = self.lstm(lstm_input, hidden_state)
        else:
            lstm_output, new_hidden_state = self.lstm(lstm_input)
        
        lstm_output = lstm_output.squeeze(0)  # (batch_size, 128)
        
        # Generate output (2D velocity commands)
        output = self.output_layer(lstm_output)  # (batch_size, 2)
        
        return output, new_hidden_state

if __name__ == '__main__':
    print("MODEL NUM PARAMS ARE")
    
    # Test GAT model
    model_with_gat = LSTMNetVIT_GAT().float()
    print("LSTMNetVIT_GAT: {}".format(sum(p.numel() for p in model_with_gat.parameters() if p.requires_grad)))
    
    # Test no-traffic model
    model_without_traffic = LSTMNetVIT_NoTraffic_2D().float()
    print("LSTMNetVIT_NoTraffic_2D: {}".format(sum(p.numel() for p in model_without_traffic.parameters() if p.requires_grad)))
    
    # Test forward pass
    batch_size = 2
    images = torch.randn(batch_size, 1, 60, 90)
    robot_states = torch.randn(batch_size, 4)
    traffic_states = torch.randn(batch_size, 5, 5)
    
    print("\nTesting forward pass...")
    with torch.no_grad():
        # Test GAT model
        output_gat, hidden_gat = model_with_gat([images, robot_states, traffic_states])
        print("GAT model output shape: {}".format(output_gat.shape))
        
        # Test no-traffic model
        output_no_traffic, hidden_no_traffic = model_without_traffic([images, robot_states])
        print("No-traffic model output shape: {}".format(output_no_traffic.shape)) 