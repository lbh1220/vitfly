#!/bin/bash

# Batch training script for testing different GAT configurations
# This script will generate config files with different GAT parameters and run training for each

echo "Starting GAT parameter experiments for traffic-aware models..."

# Base configuration template
base_config="training/config/train_airsim_with_traffic.txt"
config_dir="training/config"
temp_config_dir="training/config/gat_experiments"

# Create temporary config directory
mkdir -p "$temp_config_dir"

# Define GAT parameter combinations to test
# Format: "embed_dim,gat_hidden_dim,gat_num_heads,gat_num_layers,description"
gat_configs=(
    "32,64,2,1,baseline_small"
    "32,64,4,1,small_4heads"
    "32,64,8,1,small_8heads"
    "64,64,2,1,medium_2heads"
    "64,64,4,1,medium_4heads"
    "64,64,8,1,medium_8heads"
    "64,128,2,1,medium_128_2heads"
    "64,128,4,1,medium_128_4heads"
    "64,128,8,1,medium_128_8heads"
    "64,128,4,2,medium_128_4heads_2layers"
    "64,128,8,2,medium_128_8heads_2layers"
    "96,128,4,1,large_128_4heads"
    "96,128,8,1,large_128_8heads"
    "96,128,4,2,large_128_4heads_2layers"
    "96,128,8,2,large_128_8heads_2layers"
    "128,128,4,1,xlarge_128_4heads"
    "128,128,8,1,xlarge_128_8heads"
    "128,128,4,2,xlarge_128_4heads_2layers"
    "128,128,8,2,xlarge_128_8heads_2layers"
)

# Different learning rates to test
learning_rates=(
    "5e-5"
    "1e-4"
    "2e-4"
)

echo "Will test ${#gat_configs[@]} GAT configurations with ${#learning_rates[@]} learning rates each"
echo "Total experiments: $((${#gat_configs[@]} * ${#learning_rates[@]}))"

experiment_count=0

# Loop through GAT configurations
for gat_config in "${gat_configs[@]}"
do
    IFS=',' read -r embed_dim gat_hidden gat_heads gat_layers description <<< "$gat_config"
    
    # Loop through learning rates  
    for lr in "${learning_rates[@]}"
    do
        experiment_count=$((experiment_count + 1))
        
        # Generate experiment name
        exp_name="gat_${description}_lr${lr}"
        config_file="$temp_config_dir/train_${exp_name}.txt"
        
        echo ""
        echo "=== Experiment $experiment_count: $exp_name ==="
        echo "GAT Embed: $embed_dim, Hidden: $gat_hidden, Heads: $gat_heads, Layers: $gat_layers, LR: $lr"
        
        # Generate config file by modifying the base config
        cp "$base_config" "$config_file"
        
        # Update GAT parameters and learning rate in the config file
        sed -i "s/--embed_dim = [0-9]*/--embed_dim = $embed_dim/" "$config_file"
        sed -i "s/--gat_hidden_dim = [0-9]*/--gat_hidden_dim = $gat_hidden/" "$config_file"
        sed -i "s/--gat_num_heads = [0-9]*/--gat_num_heads = $gat_heads/" "$config_file"
        sed -i "s/--gat_num_layers = [0-9]*/--gat_num_layers = $gat_layers/" "$config_file"
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
echo "All GAT experiments completed!"
echo "Temporary config files are stored in: $temp_config_dir"
echo "You can delete them after reviewing results: rm -rf $temp_config_dir"

# Generate summary script
summary_script="training/analyze_gat_results.py"
cat > "$summary_script" << EOF
#!/usr/bin/env python3
"""
Script to analyze and compare GAT experiment results
"""
import os
import glob
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def analyze_experiments():
    log_dir = 'training/logs_airsim'
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
EOF

chmod +x "$summary_script"
echo "Created analysis script: $summary_script"
echo "Run it after experiments complete: python $summary_script"

# Create a quick comparison script
comparison_script="training/compare_gat_vs_baseline.py"
cat > "$comparison_script" << EOF
#!/usr/bin/env python3
"""
Script to compare GAT model results with baseline (no-traffic) results
"""
import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def compare_models():
    log_dir = 'training/logs_airsim'
    
    # Find best GAT experiment
    gat_experiments = []
    baseline_experiments = []
    
    for exp_dir in glob.glob(os.path.join(log_dir, '*')):
        if os.path.isdir(exp_dir):
            exp_name = os.path.basename(exp_dir)
            
            try:
                ea = EventAccumulator(exp_dir)
                ea.Reload()
                
                if 'val/loss' in ea.Tags()['scalars']:
                    val_losses = ea.Scalars('val/loss')
                    if val_losses:
                        final_val_loss = val_losses[-1].value
                        
                        if 'gat' in exp_name.lower():
                            gat_experiments.append((exp_name, final_val_loss))
                        elif 'no_traffic' in exp_name.lower():
                            baseline_experiments.append((exp_name, final_val_loss))
            except Exception as e:
                print(f"Error processing {exp_name}: {e}")
    
    # Sort by validation loss
    gat_experiments.sort(key=lambda x: x[1])
    baseline_experiments.sort(key=lambda x: x[1])
    
    print("=== Model Comparison ===")
    print("\n--- Best GAT Models ---")
    for i, (name, loss) in enumerate(gat_experiments[:5], 1):
        print(f"{i}. {name}: {loss:.6f}")
    
    print("\n--- Best Baseline Models ---")
    for i, (name, loss) in enumerate(baseline_experiments[:5], 1):
        print(f"{i}. {name}: {loss:.6f}")
    
    if gat_experiments and baseline_experiments:
        best_gat = gat_experiments[0]
        best_baseline = baseline_experiments[0]
        
        improvement = (best_baseline[1] - best_gat[1]) / best_baseline[1] * 100
        
        print(f"\n--- Summary ---")
        print(f"Best GAT: {best_gat[0]} ({best_gat[1]:.6f})")
        print(f"Best Baseline: {best_baseline[0]} ({best_baseline[1]:.6f})")
        print(f"Improvement: {improvement:.2f}%")

if __name__ == '__main__':
    compare_models()
EOF

chmod +x "$comparison_script"
echo "Created comparison script: $comparison_script"
echo "Run it to compare GAT vs baseline: python $comparison_script" 