# ViTFly Dataset Format

本数据集是AirSim多传感器仿真数据的ViTFly格式版本，经过坐标系转换和时间戳对齐处理，适用于ViTFly项目的训练和测试。

## 数据集结构

```
/dataset_root/
├── Drone1/
│   ├── 1753789555/                    # 轨迹文件夹（以任务起始时间戳命名）
│   │   ├── CAM_FRONT/                 # RGB相机数据
│   │   │   └── timestamp.jpg          # RGB图像文件
│   │   ├── CAM_FRONT_DEPTH/          # 深度相机数据
│   │   │   └── timestamp.png          # 深度图像文件（归一化到0-255）
│   │   ├── LIDAR_TOP/                # LiDAR数据
│   │   │   └── timestamp.npy          # 点云数据（Nx4 float32数组）
│   │   ├── data.csv                   # 原始ego无人机状态数据
│   │   ├── data_vitfly.csv            # 处理后的ego无人机状态数据（ViTFly格式）
│   │   ├── traffic_vitfly.csv         # 交通状态数据（ViTFly格式）
│   │   ├── mission_info.json          # 任务信息
│   │   └── traffic/                   # 原始交通数据
│   │       ├── traffic_summary.json   # 交通概览
│   │       ├── UAV_001.csv            # UAV轨迹数据
│   │       ├── eVTOL_002.csv          # eVTOL轨迹数据
│   │       └── collision_events.json  # 碰撞事件
│   └── ...                            # 其他轨迹
└── ...                                # 其他无人机
```

## 处理后的数据文件说明

### 1. data_vitfly.csv

经过处理的ego无人机状态数据，包含以下字段：

| 字段名 | 类型 | 单位 | 说明 |
|--------|------|------|------|
| timestamp | float | 秒 | 时间戳（与深度图像对齐） |
| desired_vel | float | m/s | 期望速度（任务参数） |
| quat_1~quat_4 | float | - | 四元数姿态（NED坐标系，x,y,z,w） |
| pos_x~pos_z | float | 米 | 位置（NED坐标系） |
| vel_x~vel_z | float | m/s | **速度（机体坐标系）** |
| velcmd_x~velcmd_z | float | m/s | **期望速度（机体坐标系）** |
| ct_cmd | float | - | 推力指令（ViTFly格式，当前为0） |
| br_cmd_x~br_cmd_z | float | - | 制动指令（ViTFly格式，当前为0） |
| is_collide | bool | - | 是否发生碰撞 |

**重要说明**：
- 速度数据已从世界坐标系转换为机体坐标系（仅2D旋转，基于yaw角）
- 时间戳已与深度图像对齐，确保数据一致性
- 坐标系：NED（North-East-Down）

### 2. traffic_vitfly.csv

交通状态数据，包含最近UAV和eVTOL的状态信息：

| 字段名 | 类型 | 单位 | 说明 |
|--------|------|------|------|
| nearest_uav_rel_x~z | float | 米 | 最近UAV的相对位置（ego机体坐标系） |
| nearest_uav_vel_x~z | float | m/s | 最近UAV的速度（ego机体坐标系） |
| nearest_uav_radius | float | 米 | 最近UAV的半径 |
| nearest_evtol_rel_x~z | float | 米 | 最近eVTOL的相对位置（ego机体坐标系） |
| nearest_evtol_vel_x~z | float | m/s | 最近eVTOL的速度（ego机体坐标系） |
| nearest_evtol_radius | float | 米 | 最近eVTOL的半径 |

**重要说明**：
- 相对位置：其他飞机相对于ego飞机的位置（ego机体坐标系）
- 速度：其他飞机的绝对速度转换到ego机体坐标系
- 如果某类型飞机不存在，对应字段为0
- 时间戳与data_vitfly.csv完全对应

## 坐标系转换说明

### 2D旋转转换（仅yaw角）

```python
# 世界坐标系到机体坐标系的2D转换
yaw = euler_from_quaternion(quaternion)[2]  # 提取yaw角

# 位置转换
body_x = world_x * cos(yaw) + world_y * sin(yaw)
body_y = -world_x * sin(yaw) + world_y * cos(yaw)
body_z = world_z  # z轴不变

# 速度转换
body_vx = world_vx * cos(yaw) + world_vy * sin(yaw)
body_vy = -world_vx * sin(yaw) + world_vy * cos(yaw)
body_vz = world_vz  # z轴速度不变
```

## 数据对齐说明

1. **时间戳对齐**：
   - data_vitfly.csv的时间戳与CAM_FRONT_DEPTH目录下的深度图像时间戳对齐
   - 容差：默认0.1秒，可通过--tolerance参数调整
   - 删除不匹配的数据点

2. **交通数据对齐**：
   - traffic_vitfly.csv与data_vitfly.csv的时间戳完全对应
   - 为每一帧找到最近的UAV和eVTOL（基于2D距离）
   - 时间容差：0.5秒内找到最近的飞机状态

## 传感器数据格式

### 深度图像
- 格式：PNG（推荐）或JPG
- 数据范围：0-255（归一化，假设最大深度100m）
- 文件名：时间戳.png/jpg

### LiDAR点云
- 格式：NPY（推荐）或BIN
- 数据结构：Nx4 float32数组
- 列0-2：x, y, z坐标（米）
- 列3：语义分割标签（float32格式）

### RGB图像
- 格式：JPG（推荐）或PNG
- 质量：95%（JPG格式）
- 兼容NuScenes数据集格式

## 使用建议

1. **数据读取**：
   ```python
   import pandas as pd
   
   # 读取ego状态数据
   ego_data = pd.read_csv('data_vitfly.csv')
   
   # 读取交通数据
   traffic_data = pd.read_csv('traffic_vitfly.csv')
   
   # 确保时间戳对应
   assert len(ego_data) == len(traffic_data)
   ```

2. **图像数据读取**：
   ```python
   import cv2
   import numpy as np
   
   # 读取深度图像
   depth_img = cv2.imread('CAM_FRONT_DEPTH/timestamp.png', cv2.IMREAD_GRAYSCALE)
   
   # 读取LiDAR数据
   lidar_points = np.load('LIDAR_TOP/timestamp.npy')
   ```

3. **坐标系注意**：
   - 所有位置和速度数据都在NED坐标系下
   - 机体坐标系基于ego飞机的yaw角进行2D旋转
   - 相对位置是相对于ego飞机的机体坐标系

## 数据统计

- **ego状态数据**：21列（时间戳、期望速度、姿态、位置、速度、期望速度、推力指令、制动指令、碰撞状态）
- **交通数据**：14列（最近UAV 7列 + 最近eVTOL 7列）
- **总数据维度**：35列（ego 21列 + traffic 14列）

## 处理工具

使用`convert_vitfly.py`脚本进行数据处理：

```bash
# 处理单个轨迹
python3 convert_vitfly.py /path/to/dataset/Drone1 --single-trajectory 1753789555

# 处理整个数据集
python3 convert_vitfly.py /path/to/dataset/Drone1

# 调整时间戳对齐容差
python3 convert_vitfly.py /path/to/dataset/Drone1 --tolerance 0.2
```

脚本会自动：
1. 删除已存在的输出文件
2. 进行坐标系转换
3. 时间戳对齐
4. 生成新的ViTFly格式数据文件 