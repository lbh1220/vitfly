#!/bin/bash

# Batch training script for testing different traffic MLP configurations
# This script will generate config files with different MLP parameters and run training for each

echo "Starting MLP parameter experiments for traffic-aware models..."

# Base configuration template
base_config="training/config/train_airsim_with_traffic.txt"
config_dir="training/config"
temp_config_dir="training/config/mlp_experiments"

# Create temporary config directory
mkdir -p "$temp_config_dir"

# Define MLP parameter combinations to test
# Format: "hidden_size,output_size,description"
mlp_configs=(
    "64,32,baseline"
    "64,64,h64_o64" 
    "128,32,h128_o32"
    "128,64,h128_o64"
    "128,128,h128_o128"
    "96,48,h96_o48"
)

# Different learning rates to test
learning_rates=(
    "5e-5"
    "3e-5" 
    "1e-4"
)

echo "Will test ${#mlp_configs[@]} MLP configurations with ${#learning_rates[@]} learning rates each"
echo "Total experiments: $((${#mlp_configs[@]} * ${#learning_rates[@]}))"

experiment_count=0

# Loop through MLP configurations
for mlp_config in "${mlp_configs[@]}"
do
    IFS=',' read -r hidden output description <<< "$mlp_config"
    
    # Loop through learning rates  
    for lr in "${learning_rates[@]}"
    do
        experiment_count=$((experiment_count + 1))
        
        # Generate experiment name
        exp_name="mlp_${description}_lr${lr}"
        config_file="$temp_config_dir/train_${exp_name}.txt"
        
        echo ""
        echo "=== Experiment $experiment_count: $exp_name ==="
        echo "MLP Hidden: $hidden, Output: $output, LR: $lr"
        
        # Generate config file by modifying the base config
        cp "$base_config" "$config_file"
        
        # Update MLP parameters and learning rate in the config file
        sed -i "s/--traffic_mlp_hidden = [0-9]*/--traffic_mlp_hidden = $hidden/" "$config_file"
        sed -i "s/--traffic_mlp_output = [0-9]*/--traffic_mlp_output = $output/" "$config_file"
        sed -i "s/--lr=[0-9e\-]*/--lr=$lr/" "$config_file"
        
        # Add experiment suffix to workspace name
        echo "--ws_suffix=_${exp_name}" >> "$config_file"
        
        echo "Generated config: $config_file"
        echo "Starting training..."
        
        # Run training
        python training/train_airsim.py --config "$config_file"
        
        if [ $? -eq 0 ]; then
            echo "✓ Experiment $exp_name completed successfully"
        else
            echo "✗ Experiment $exp_name failed"
        fi
        
        echo "----------------------------------------"
    done
done

echo ""
echo "All experiments completed!"
echo "Temporary config files are stored in: $temp_config_dir"
echo "You can delete them after reviewing results: rm -rf $temp_config_dir"

# Generate summary script
summary_script="training/analyze_mlp_results.py"
cat > "$summary_script" << EOF
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
EOF

chmod +x "$summary_script"
echo "Created analysis script: $summary_script"
echo "Run it after experiments complete: python $summary_script" 