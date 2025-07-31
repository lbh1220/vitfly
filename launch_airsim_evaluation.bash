#!/bin/bash

# AirSim Vision-based Evaluation Launch Script
# Usage: ./launch_airsim_evaluation.bash [model_type] [model_path] [des_vel]

# Default parameters
MODEL_TYPE="${1:-ViTLSTM}"
MODEL_PATH="${2:-../../training/logs/d07_10_t12_10/model_000099.pth}"
DES_VEL="${3:-5.0}"

echo
echo "[AIRSIM LAUNCH] Starting AirSim vision-based evaluation..."
echo "[AIRSIM LAUNCH] Model Type: $MODEL_TYPE"
echo "[AIRSIM LAUNCH] Model Path: $MODEL_PATH" 
echo "[AIRSIM LAUNCH] Desired Velocity: $DES_VEL"
echo

# Change to the script directory
cd ./envtest/ros/

# Launch the airsim competition node with vision-based mode
echo "[AIRSIM LAUNCH] Starting agile pilot airsim node..."
python3 run_competition_airsim.py --vision_based --model_type "$MODEL_TYPE" --model_path "$MODEL_PATH" --des_vel "$DES_VEL"

echo "[AIRSIM LAUNCH] AirSim evaluation completed." 