#!/usr/bin/env python3
"""
Script to analyze and compare MLP experiment results
"""
import os
import glob
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def analyze_experiments():
    log_dir = 'training/logs_airsim'
    experiments = []
    
    # Find all experiment directories
    for exp_dir in glob.glob(os.path.join(log_dir, '*with_traffic*mlp*')):
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
    
    print("=== MLP Experiment Results (sorted by validation loss) ===")
    print(f"{'Rank':<4} {'Experiment':<50} {'Final Val Loss':<15}")
    print("-" * 70)
    
    for i, exp in enumerate(experiments, 1):
        print(f"{i:<4} {exp['experiment']:<50} {exp['final_val_loss']:<15.6f}")
    
    return experiments

if __name__ == '__main__':
    analyze_experiments()
