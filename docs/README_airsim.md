# AirSim Dataset Training Scripts

这些脚本专门为AirSim数据集格式设计，支持带有和不带有交通信息的训练。

## 文件说明

### 核心文件
- `dataloading_airsim.py`: AirSim数据集的数据加载器
- `train_airsim.py`: AirSim数据集的训练脚本
- `model_airsim.py`: 修改后的模型架构，支持交通信息输入

### 配置文件
- `config/train_airsim_with_traffic.txt`: 带交通信息的训练配置
- `config/train_airsim_no_traffic.txt`: 不带交通信息的训练配置

## 数据集要求

数据集应按照以下结构组织：
```
/dataset_root/
├── Drone1/
│   ├── 1753789555/                    # 轨迹文件夹
│   │   ├── CAM_FRONT_DEPTH/          # 深度图像目录
│   │   │   └── timestamp.png         # 深度图像文件
│   │   ├── data_vitfly.csv           # ego飞机状态数据
│   │   └── traffic_vitfly.csv        # 交通数据（可选）
│   └── ...
```

### 必需的数据文件

#### data_vitfly.csv
包含以下列：
- `timestamp`: 时间戳
- `desired_vel`: 期望速度
- `quat_1`, `quat_2`, `quat_3`, `quat_4`: 四元数姿态
- `pos_x`, `pos_y`, `pos_z`: 位置
- `vel_x`, `vel_y`, `vel_z`: 当前速度
- `velcmd_x`, `velcmd_y`, `velcmd_z`: 速度命令
- `ct_cmd`: 推力命令
- `br_cmd_x`, `br_cmd_y`, `br_cmd_z`: 制动命令
- `is_collide`: 碰撞标志

#### traffic_vitfly.csv（可选）
包含14列交通数据：
- `nearest_uav_rel_x`, `nearest_uav_rel_y`, `nearest_uav_rel_z`: 最近UAV相对位置
- `nearest_uav_vel_x`, `nearest_uav_vel_y`, `nearest_uav_vel_z`: 最近UAV速度
- `nearest_uav_radius`: 最近UAV半径
- `nearest_evtol_rel_x`, `nearest_evtol_rel_y`, `nearest_evtol_rel_z`: 最近eVTOL相对位置
- `nearest_evtol_vel_x`, `nearest_evtol_vel_y`, `nearest_evtol_vel_z`: 最近eVTOL速度
- `nearest_evtol_radius`: 最近eVTOL半径

## 使用方法

### 1. 带交通信息的训练

```bash
cd src/vitfly/training
python train_airsim.py --config config/train_airsim_with_traffic.txt
```

或者直接使用命令行参数：
```bash
python train_airsim.py \
  --basedir /path/to/vitfly \
  --datadir /path/to/dataset \
  --dataset Drone1 \
  --use_traffic \
  --model_type LSTMNetVIT_Traffic \
  --N_eps 100 \
  --device cuda
```

### 2. 不带交通信息的训练

```bash
cd src/vitfly/training
python train_airsim.py --config config/train_airsim_no_traffic.txt
```

或者：
```bash
python train_airsim.py \
  --basedir /path/to/vitfly \
  --datadir /path/to/dataset \
  --dataset Drone1 \
  --model_type LSTMNetVIT_NoTraffic \
  --N_eps 100 \
  --device cuda
```

### 3. 从检查点继续训练

```bash
python train_airsim.py \
  --config config/train_airsim_with_traffic.txt \
  --load_checkpoint \
  --checkpoint_path /path/to/model_000050.pth
```

## 模型架构

### LSTMNetVIT_Traffic
- 支持14维交通信息输入
- LSTM输入维度：531 (512视觉特征 + 1期望速度 + 4四元数 + 14交通信息)
- 参数数量：约3.6M（增加了交通信息处理）

### LSTMNetVIT_NoTraffic
- 原始ViTLSTM架构，不使用交通信息
- LSTM输入维度：517 (512视觉特征 + 1期望速度 + 4四元数)
- 参数数量：约3.5M

## 主要参数说明

- `--use_traffic`: 是否使用交通信息（自动设置模型类型）
- `--dataset`: 数据集文件夹名称（如Drone1）
- `--short`: 限制加载的轨迹数量（用于快速测试）
- `--val_split`: 验证集比例（默认0.2）
- `--seed`: 随机种子（用于可重复实验）

## 输出文件

训练过程中会在以下目录生成文件：
```
logs_airsim/
├── d01_15_t14_30_with_traffic/     # 带交通信息的实验
│   ├── args.txt                    # 训练参数
│   ├── log.txt                     # 训练日志
│   ├── model_000025.pth           # 模型检查点
│   ├── train_val_dirs.npy         # 训练验证集分割
│   └── events.out.tfevents.*      # TensorBoard日志
└── d01_15_t14_45_no_traffic/      # 不带交通信息的实验
    └── ...
```

## 性能对比

训练两个模型后，可以通过以下方式对比性能：

1. **TensorBoard可视化**：
```bash
tensorboard --logdir logs_airsim
```

2. **验证损失对比**：查看各自的`log.txt`文件中的验证损失

3. **模型推理对比**：在实际环境中测试两个模型的性能差异

## 故障排除

### 常见问题

1. **数据加载错误**：
   - 检查数据集路径是否正确
   - 确认`data_vitfly.csv`和深度图像是否存在
   - 验证CSV文件格式是否正确

2. **内存不足**：
   - 减少`--short`参数值，限制加载的轨迹数量
   - 使用更小的批次大小（需要修改代码）

3. **CUDA错误**：
   - 检查GPU可用性：`nvidia-smi`
   - 尝试使用CPU：`--device cpu`

4. **交通数据维度错误**：
   - 确认`traffic_vitfly.csv`包含正确的14列
   - 检查数据是否与ego数据时间戳对齐

## 注意事项

1. **路径配置**：修改配置文件中的路径以匹配你的环境
2. **计算资源**：带交通信息的模型需要稍多的计算资源
3. **数据质量**：确保深度图像和状态数据的时间戳正确对齐
4. **实验记录**：建议使用不同的`--ws_suffix`来区分不同的实验 