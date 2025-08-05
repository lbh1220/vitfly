#!/usr/bin/env python3
"""
Script to analyze and compare GAT experiment results
"""
import os
import glob
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def analyze_experiments():
    log_dir = 'training/logs_airsim/gat'
    experiments = []
    
    # Find all experiment directories
    for exp_dir in glob.glob(os.path.join(log_dir, '*with_traffic*gat*')):
        if os.path.isdir(exp_dir):
            exp_name = os.path.basename(exp_dir)
            
            # Try to extract final validation loss
            try:
                ea = EventAccumulator(exp_dir)
                ea.Reload()
                
                if 'val/loss' in ea.Tags()['scalars']:
                    val_losses = ea.Scalars('val/loss')
                    if val_losses:
                        final_val_loss = val_losses[-1].value
                        experiments.append({
                            'experiment': exp_name,
                            'final_val_loss': final_val_loss,
                            'directory': exp_dir
                        })
            except Exception as e:
                print(f"Error processing {exp_name}: {e}")
    
    # Sort by validation loss
    experiments.sort(key=lambda x: x['final_val_loss'])
    
    print("=== GAT Experiment Results (sorted by validation loss) ===")
    print(f"{'Rank':<4} {'Experiment':<60} {'Final Val Loss':<15}")
    print("-" * 80)
    
    for i, exp in enumerate(experiments, 1):
        print(f"{i:<4} {exp['experiment']:<60} {exp['final_val_loss']:<15.6f}")
    
    # Generate detailed analysis
    print("\n=== Detailed Analysis ===")
    analyze_parameter_impact(experiments)
    
    return experiments

def analyze_parameter_impact(experiments):
    """Analyze the impact of different parameters on performance"""
    
    # Group by embed_dim
    embed_groups = {}
    hidden_groups = {}
    heads_groups = {}
    layers_groups = {}
    
    for exp in experiments:
        exp_name = exp['experiment']
        
        # Extract parameters from experiment name
        if 'embed32' in exp_name:
            embed_dim = 32
        elif 'embed64' in exp_name:
            embed_dim = 64
        elif 'embed96' in exp_name:
            embed_dim = 96
        elif 'embed128' in exp_name:
            embed_dim = 128
        else:
            embed_dim = 'unknown'
        
        if 'gat64' in exp_name:
            hidden_dim = 64
        elif 'gat128' in exp_name:
            hidden_dim = 128
        else:
            hidden_dim = 'unknown'
        
        if '2heads' in exp_name:
            num_heads = 2
        elif '4heads' in exp_name:
            num_heads = 4
        elif '8heads' in exp_name:
            num_heads = 8
        else:
            num_heads = 'unknown'
        
        if '2layers' in exp_name:
            num_layers = 2
        else:
            num_layers = 1
        
        # Group experiments
        if embed_dim not in embed_groups:
            embed_groups[embed_dim] = []
        embed_groups[embed_dim].append(exp)
        
        if hidden_dim not in hidden_groups:
            hidden_groups[hidden_dim] = []
        hidden_groups[hidden_dim].append(exp)
        
        if num_heads not in heads_groups:
            heads_groups[num_heads] = []
        heads_groups[num_heads].append(exp)
        
        if num_layers not in layers_groups:
            layers_groups[num_layers] = []
        layers_groups[num_layers].append(exp)
    
    # Print analysis
    print("\n--- Embedding Dimension Impact ---")
    for embed_dim, exps in embed_groups.items():
        avg_loss = sum(exp['final_val_loss'] for exp in exps) / len(exps)
        best_loss = min(exp['final_val_loss'] for exp in exps)
        print(f"Embed {embed_dim}: Avg={avg_loss:.6f}, Best={best_loss:.6f} ({len(exps)} exps)")
    
    print("\n--- Hidden Dimension Impact ---")
    for hidden_dim, exps in hidden_groups.items():
        avg_loss = sum(exp['final_val_loss'] for exp in exps) / len(exps)
        best_loss = min(exp['final_val_loss'] for exp in exps)
        print(f"Hidden {hidden_dim}: Avg={avg_loss:.6f}, Best={best_loss:.6f} ({len(exps)} exps)")
    
    print("\n--- Number of Heads Impact ---")
    for num_heads, exps in heads_groups.items():
        avg_loss = sum(exp['final_val_loss'] for exp in exps) / len(exps)
        best_loss = min(exp['final_val_loss'] for exp in exps)
        print(f"Heads {num_heads}: Avg={avg_loss:.6f}, Best={best_loss:.6f} ({len(exps)} exps)")
    
    print("\n--- Number of Layers Impact ---")
    for num_layers, exps in layers_groups.items():
        avg_loss = sum(exp['final_val_loss'] for exp in exps) / len(exps)
        best_loss = min(exp['final_val_loss'] for exp in exps)
        print(f"Layers {num_layers}: Avg={avg_loss:.6f}, Best={best_loss:.6f} ({len(exps)} exps)")

if __name__ == '__main__':
    analyze_experiments()