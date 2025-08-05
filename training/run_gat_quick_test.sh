#!/bin/bash

# Quick test script for GAT configurations
# This script tests a subset of key GAT parameters for quick validation

echo "Starting quick GAT parameter test..."

# Base configuration template
base_config="training/config/train_airsim_with_traffic.txt"
temp_config_dir="training/config/gat_quick_test"

# Create temporary config directory
mkdir -p "$temp_config_dir"

# Define a subset of key GAT configurations for quick testing
# Format: "embed_dim,gat_hidden_dim,gat_num_heads,gat_num_layers,description"
quick_gat_configs=(
    "32,64,2,1,small_2heads"
    "64,64,4,1,medium_4heads"
    "64,128,4,1,medium_128_4heads"
    "64,128,8,1,medium_128_8heads"
    "64,128,4,2,medium_128_4heads_2layers"
)

# Single learning rate for quick test
learning_rate="1e-4"

echo "Will test ${#quick_gat_configs[@]} GAT configurations with learning rate: $learning_rate"
echo "Total experiments: ${#quick_gat_configs[@]}"

experiment_count=0

# Loop through GAT configurations
for gat_config in "${quick_gat_configs[@]}"
do
    IFS=',' read -r embed_dim gat_hidden gat_heads gat_layers description <<< "$gat_config"
    
    experiment_count=$((experiment_count + 1))
    
    # Generate experiment name
    exp_name="gat_quick_${description}"
    config_file="$temp_config_dir/train_${exp_name}.txt"
    
    echo ""
    echo "=== Quick Test $experiment_count: $exp_name ==="
    echo "GAT Embed: $embed_dim, Hidden: $gat_hidden, Heads: $gat_heads, Layers: $gat_layers"
    
    # Generate config file by modifying the base config
    cp "$base_config" "$config_file"
    
    # Update GAT parameters in the config file
    sed -i "s/--embed_dim = [0-9]*/--embed_dim = $embed_dim/" "$config_file"
    sed -i "s/--gat_hidden_dim = [0-9]*/--gat_hidden_dim = $gat_hidden/" "$config_file"
    sed -i "s/--gat_num_heads = [0-9]*/--gat_num_heads = $gat_heads/" "$config_file"
    sed -i "s/--gat_num_layers = [0-9]*/--gat_num_layers = $gat_layers/" "$config_file"
    sed -i "s/--lr=[0-9e\-]*/--lr=$learning_rate/" "$config_file"
    
    # Add experiment suffix to workspace name
    echo "--ws_suffix=_${exp_name}" >> "$config_file"
    
    # Reduce epochs for quick test
    sed -i "s/--N_eps=[0-9]*/--N_eps=50/" "$config_file"
    
    echo "Generated config: $config_file"
    echo "Starting training..."
    
    # Run training
    python training/train_airsim.py --config "$config_file"
    
    if [ $? -eq 0 ]; then
        echo "✓ Quick test $exp_name completed successfully"
    else
        echo "✗ Quick test $exp_name failed"
    fi
    
    echo "----------------------------------------"
done

echo ""
echo "Quick GAT tests completed!"
echo "Temporary config files are stored in: $temp_config_dir"
echo "You can delete them after reviewing results: rm -rf $temp_config_dir"

# Create a quick analysis script
quick_analysis_script="training/analyze_quick_gat_results.py"
cat > "$quick_analysis_script" << EOF
#!/usr/bin/env python3
"""
Quick analysis script for GAT experiment results
"""
import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def analyze_quick_experiments():
    log_dir = 'training/logs_airsim'
    experiments = []
    
    # Find all quick test experiment directories
    for exp_dir in glob.glob(os.path.join(log_dir, '*with_traffic*gat_quick*')):
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
    
    print("=== Quick GAT Test Results (sorted by validation loss) ===")
    print(f"{'Rank':<4} {'Experiment':<50} {'Final Val Loss':<15}")
    print("-" * 70)
    
    for i, exp in enumerate(experiments, 1):
        print(f"{i:<4} {exp['experiment']:<50} {exp['final_val_loss']:<15.6f}")
    
    if experiments:
        print(f"\nBest configuration: {experiments[0]['experiment']}")
        print(f"Best validation loss: {experiments[0]['final_val_loss']:.6f}")
    
    return experiments

if __name__ == '__main__':
    analyze_quick_experiments()
EOF

chmod +x "$quick_analysis_script"
echo "Created quick analysis script: $quick_analysis_script"
echo "Run it after quick tests complete: python $quick_analysis_script" 