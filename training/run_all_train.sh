#!/bin/bash

# 配置文件夹前缀
config_dir="training/config"

# 只写配置文件名
configs=(
    "train_lrdecay.txt"
    "train_200epoch.txt"
)

# 依次训练
for config in "${configs[@]}"
do
    config_path="$config_dir/$config"
    echo "Start training with $config_path ..."
    python training/train.py --config "$config_path"
    echo "Finished training with $config_path"
    echo "----------------------------------------"
done

echo "All training finished!"