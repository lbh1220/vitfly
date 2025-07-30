"""
@authors: Modified for AirSim dataset with traffic information
@organization: GRASP Lab, University of Pennsylvania
@date: ...
@license: ...

@brief: This module contains model architectures for AirSim dataset training with traffic information
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
import sys, os
from os.path import join as opj

# Import ViT submodules
sys.path.append(opj(os.path.dirname(os.path.abspath(__file__))))
from ViTsubmodules import *

def refine_inputs(X):
    # fill quaternion rotation if not given
    # make it [1, 0, 0, 0] repeated with numrows = X[0].shape[0]
    if X[2] is None:
        X[2] = torch.zeros((X[0].shape[0], 4)).float().to(X[0].device)
        X[2][:, 0] = 1

    # if input depth images are not of right shape, resize
    if X[0].shape[-2] != 60 or X[0].shape[-1] != 90:
        X[0] = F.interpolate(X[0], size=(60, 90), mode='bilinear')

    return X

class LSTMNetVIT_Traffic(nn.Module):
    """
    ViT+LSTM Network with Traffic Information
    Modified to include 14-dimensional traffic data input
    """
    def __init__(self, use_traffic=True):
        super().__init__()
        self.use_traffic = use_traffic
        
        self.encoder_blocks = nn.ModuleList([
            MixTransformerEncoderLayer(1, 32, patch_size=7, stride=4, padding=3, n_layers=2, reduction_ratio=8, num_heads=1, expansion_factor=8),
            MixTransformerEncoderLayer(32, 64, patch_size=3, stride=2, padding=1, n_layers=2, reduction_ratio=4, num_heads=2, expansion_factor=8)
        ])

        self.decoder = spectral_norm(nn.Linear(4608, 512))
        
        # Calculate LSTM input size based on whether traffic data is used
        if self.use_traffic:
            lstm_input_size = 512 + 1 + 4 + 14  # vision + desired_vel + quaternion + traffic = 531
        else:
            lstm_input_size = 512 + 1 + 4  # vision + desired_vel + quaternion = 517
            
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=128, num_layers=3, dropout=0.1)
        self.nn_fc2 = spectral_norm(nn.Linear(128, 3))

        self.up_sample = nn.Upsample(size=(16,24), mode='bilinear', align_corners=True)
        self.pxShuffle = nn.PixelShuffle(upscale_factor=2)
        self.down_sample = nn.Conv2d(48,12,3, padding=1)

    def forward(self, X):
        X = refine_inputs(X)

        x = X[0]  # depth images
        embeds = [x]
        for block in self.encoder_blocks:
            embeds.append(block(embeds[-1]))        
        out = embeds[1:]
        out = torch.cat([self.pxShuffle(out[1]), self.up_sample(out[0])], dim=1) 
        out = self.down_sample(out)
        out = self.decoder(out.flatten(1))
        
        # Concatenate features: vision + desired_vel + quaternion
        features = [out, X[1]/10, X[2]]
        
        # Add traffic data if available and enabled
        if self.use_traffic and len(X) > 3:
            # Check if traffic data is provided (X[3] could be traffic or hidden state)
            if len(X) > 4:  # X[3] is traffic, X[4] is hidden state
                features.append(X[3])  # Add traffic data
                out = torch.cat(features, dim=1).float()
                out, h = self.lstm(out, X[4])
            else:  # X[3] could be traffic data or hidden state
                # Assume if X[3] has 14 features it's traffic data, otherwise it's hidden state
                if hasattr(X[3], 'shape') and len(X[3].shape) == 2 and X[3].shape[1] == 14:
                    features.append(X[3])  # Add traffic data
                    out = torch.cat(features, dim=1).float()
                    out, h = self.lstm(out)
                else:  # X[3] is hidden state
                    out = torch.cat(features, dim=1).float()
                    out, h = self.lstm(out, X[3])
        else:
            # No traffic data
            out = torch.cat(features, dim=1).float()
            if len(X) > 3:
                out, h = self.lstm(out, X[3])  # X[3] is hidden state
            else:
                out, h = self.lstm(out)
        
        out = self.nn_fc2(out)
        return out, h

class LSTMNetVIT_NoTraffic(nn.Module):
    """
    Original ViT+LSTM Network without Traffic Information
    For comparison purposes
    """
    def __init__(self):
        super().__init__()
        self.encoder_blocks = nn.ModuleList([
            MixTransformerEncoderLayer(1, 32, patch_size=7, stride=4, padding=3, n_layers=2, reduction_ratio=8, num_heads=1, expansion_factor=8),
            MixTransformerEncoderLayer(32, 64, patch_size=3, stride=2, padding=1, n_layers=2, reduction_ratio=4, num_heads=2, expansion_factor=8)
        ])

        self.decoder = spectral_norm(nn.Linear(4608, 512))
        self.lstm = nn.LSTM(input_size=517, hidden_size=128, num_layers=3, dropout=0.1)
        self.nn_fc2 = spectral_norm(nn.Linear(128, 3))

        self.up_sample = nn.Upsample(size=(16,24), mode='bilinear', align_corners=True)
        self.pxShuffle = nn.PixelShuffle(upscale_factor=2)
        self.down_sample = nn.Conv2d(48,12,3, padding=1)

    def forward(self, X):
        X = refine_inputs(X)

        x = X[0]
        embeds = [x]
        for block in self.encoder_blocks:
            embeds.append(block(embeds[-1]))        
        out = embeds[1:]
        out = torch.cat([self.pxShuffle(out[1]), self.up_sample(out[0])], dim=1) 
        out = self.down_sample(out)
        out = self.decoder(out.flatten(1))
        out = torch.cat([out, X[1]/10, X[2]], dim=1).float()
        
        if len(X) > 3:
            out, h = self.lstm(out, X[3])
        else:
            out, h = self.lstm(out)
        out = self.nn_fc2(out)
        return out, h

if __name__ == '__main__':
    print("MODEL NUM PARAMS ARE")
    
    model_with_traffic = LSTMNetVIT_Traffic(use_traffic=True).float()
    print("LSTMNetVIT with Traffic: ")
    print(sum(p.numel() for p in model_with_traffic.parameters() if p.requires_grad))
    
    model_without_traffic = LSTMNetVIT_NoTraffic().float()
    print("LSTMNetVIT without Traffic: ")
    print(sum(p.numel() for p in model_without_traffic.parameters() if p.requires_grad)) 