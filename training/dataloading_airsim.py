"""
@authors: Modified for AirSim dataset
@organization: GRASP Lab, University of Pennsylvania
@date: ...
@license: ...

@brief: This module contains the dataloading routine for AirSim dataset in ViTFly format
"""

import cv2
import glob, os, time
from os.path import join as opj
import numpy as np
import torch
import random
import pandas as pd
import getpass
uname = getpass.getuser()

def dataloader_airsim(data_dir, val_split=0., short=0, seed=None, train_val_dirs=None, use_traffic=True):
    """
    AirSim dataset dataloader
    
    Args:
        data_dir: Path to dataset root (e.g., /dataset_root/Drone1)
        val_split: Fraction of data to use for validation
        short: If nonzero, limit number of trajectory folders to load
        seed: Random seed for reproducibility
        train_val_dirs: Pre-defined train/val split directories
        use_traffic: Whether to include traffic data (14-dim traffic features)
    """
    cropHeight = 60
    cropWidth = 90

    if train_val_dirs is not None:
        traj_folders = train_val_dirs[0] + train_val_dirs[1]
        val_split = len(train_val_dirs[1]) / len(traj_folders)
    else:
        traj_folders = sorted(glob.glob(opj(data_dir, '*')))
        random.seed(seed)
        random.shuffle(traj_folders)

    if short > 0:
        assert short <= len(traj_folders), f"short={short} is greater than the number of folders={len(traj_folders)}"
        traj_folders = traj_folders[:short]
    
    desired_vels = []
    traj_ims_full = []
    traj_meta_full = []
    curr_quats = []
    traffic_data_full = [] if use_traffic else None

    start_dataloading = time.time()

    skippedImages = 0
    skippedFolders = 0
    collisionImages = 0
    collisionFolders = 0

    for i, traj_folder in enumerate(traj_folders):
        if len(traj_folders)//10 > 0 and i % (len(traj_folders)//10) == 0:
            print(f'[DATALOADER] Loading folder {os.path.basename(traj_folder)}, folder # {i+1}/{len(traj_folders)}, time elapsed {time.time()-start_dataloading:.2f}s')
        
        # 1. 查找深度图像目录
        depth_dir = opj(traj_folder, 'CAM_FRONT_DEPTH')
        if not os.path.isdir(depth_dir):
            print(f'[DATALOADER] No CAM_FRONT_DEPTH directory in {os.path.basename(traj_folder)}, skipping')
            skippedFolders += 1
            continue
            
        depth_im_files = sorted(glob.glob(opj(depth_dir, '*.png')))
        
        if len(depth_im_files) < 10:
            print(f'[DATALOADER] Trajectory too short, skipping {os.path.basename(traj_folder)}')
            skippedFolders += 1
            continue

        if len(depth_im_files) == 0:
            print(f'[DATALOADER] No depth images found in {os.path.basename(traj_folder)}, skipping')
            skippedFolders += 1
            continue

        # 2. 读取ViTFly格式的状态数据
        ego_csv_file = opj(traj_folder, 'data_vitfly.csv')
        if not os.path.exists(ego_csv_file):
            print(f'[DATALOADER] No data_vitfly.csv found in {os.path.basename(traj_folder)}, skipping')
            skippedFolders += 1
            continue
        
        try:
            ego_data = pd.read_csv(ego_csv_file)
        except Exception as e:
            print(f'[DATALOADER] Error reading {ego_csv_file}: {e}, skipping')
            skippedFolders += 1
            continue

        # 3. 读取交通数据（如果需要）
        traffic_data = None
        if use_traffic:
            traffic_csv_file = opj(traj_folder, 'traffic_vitfly.csv')
            if os.path.exists(traffic_csv_file):
                try:
                    traffic_data = pd.read_csv(traffic_csv_file)
                    
                    # 检查数据维度是否为14维
                    if traffic_data.shape[1] != 14:
                        # print(f'[DATALOADER] Traffic data has {traffic_data.shape[1]} columns, expected 14 in {os.path.basename(traj_folder)}')
                        if traffic_data.shape[1] == 15:
                            # print(f'[DATALOADER] Removing first column as it appears to be an index')
                            traffic_data = traffic_data.iloc[:, 1:]
                        else:
                            print(f'[DATALOADER] Traffic data dimension mismatch, skipping')
                            skippedFolders += 1
                            continue
                    
                    # 检查数据长度是否一致
                    if len(traffic_data) != len(ego_data):
                        print(f'[DATALOADER] Traffic data length mismatch in {os.path.basename(traj_folder)}, skipping')
                        skippedFolders += 1
                        continue
                except Exception as e:
                    print(f'[DATALOADER] Error reading {traffic_csv_file}: {e}, skipping')
                    skippedFolders += 1
                    continue
            else:
                print(f'[DATALOADER] No traffic_vitfly.csv found in {os.path.basename(traj_folder)}, using zeros')
                # 如果没有交通数据，创建全零的14维数据
                traffic_data = pd.DataFrame(np.zeros((len(ego_data), 14)))

        # 4. 转换为numpy数组格式（兼容原始ViTFly格式）
        # ego_data列：timestamp, desired_vel, quat_1~4, pos_x~z, vel_x~z, velcmd_x~z, ct_cmd, br_cmd_x~z, is_collide
        try:
            # 构建与原始data.csv兼容的数组格式
            traj_meta = np.column_stack([
                np.arange(len(ego_data)),  # 索引列
                ego_data['timestamp'].values,  # 时间戳
                ego_data['desired_vel'].values,  # 期望速度
                ego_data[['quat_1', 'quat_2', 'quat_3', 'quat_4']].values,  # 四元数
                ego_data[['pos_x', 'pos_y', 'pos_z']].values,  # 位置
                ego_data[['vel_x', 'vel_y', 'vel_z']].values,  # 当前速度
                ego_data[['velcmd_x', 'velcmd_y', 'velcmd_z']].values,  # 速度命令
                ego_data['ct_cmd'].values.reshape(-1, 1),  # 推力命令
                ego_data[['br_cmd_x', 'br_cmd_y', 'br_cmd_z']].values,  # 制动命令
                ego_data['is_collide'].values.reshape(-1, 1).astype(bool)  # 碰撞标志
            ])
        except KeyError as e:
            print(f'[DATALOADER] Missing column in {ego_csv_file}: {e}, skipping')
            skippedFolders += 1
            continue

        # 检查是否有NaN值
        if np.isnan(traj_meta).any():
            print(f'[DATALOADER] NaN in {os.path.basename(traj_folder)}, skipping')
            skippedFolders += 1
            continue

        # 5. 时间戳对齐（图像文件名即为时间戳）
        img_timestamps = [float(os.path.basename(f)[:-4]) for f in depth_im_files]
        state_timestamps = traj_meta[:, 1]  # 第2列是时间戳

        # 找到重叠区间
        start_time = max(img_timestamps[0], state_timestamps[0])
        end_time = min(img_timestamps[-1], state_timestamps[-1])

        # 剪裁到重叠区间
        img_indices = [i for i, t in enumerate(img_timestamps) if start_time <= t <= end_time]
        state_indices = [i for i, t in enumerate(state_timestamps) if start_time <= t <= end_time]

        # 只保留重叠区间的图像和状态
        img_timestamps = [img_timestamps[i] for i in img_indices]
        depth_im_files = [depth_im_files[i] for i in img_indices]
        traj_meta = traj_meta[state_indices, :]

        # 对齐交通数据
        if use_traffic and traffic_data is not None:
            traffic_array = traffic_data.iloc[state_indices].values

        # 再次检查数量是否一致
        min_len = min(len(depth_im_files), traj_meta.shape[0])
        depth_im_files = depth_im_files[:min_len]
        traj_meta = traj_meta[:min_len, :]
        if use_traffic and traffic_data is not None:
            traffic_array = traffic_array[:min_len, :]

        if min_len == 0:
            print(f'[DATALOADER] No matched images and states in {os.path.basename(traj_folder)}, skipping')
            skippedFolders += 1
            continue

        # 6. 读取并处理图像
        try:
            traj_ims = np.asarray([cv2.imread(im_file, cv2.IMREAD_GRAYSCALE) for im_file in depth_im_files], dtype=np.float32) / 255.0
            temp = [cv2.resize(img, (cropWidth, cropHeight)) for img in traj_ims]
            traj_ims = np.array(temp)
        except Exception as e:
            print(f'[DATALOADER] Error processing images in {os.path.basename(traj_folder)}: {e}, skipping')
            skippedFolders += 1
            continue

        # 7. 提取数据
        for ii in range(traj_meta.shape[0]):
            desired_vels.append(traj_meta[ii, 2])  # 期望速度
            q = traj_meta[ii, 3:7]  # 四元数
            curr_quats.append(q)

        try:
            traj_ims_full.append(traj_ims)
            traj_meta_full.append(traj_meta)
            if use_traffic and traffic_data is not None:
                traffic_data_full.append(traffic_array)
        except Exception as e:
            print(f'[DATALOADER] {traj_ims.shape}')
            print(f"[DATALOADER] Error appending data for folder {os.path.basename(traj_folder)}: {e}")
            skippedFolders += 1
            continue

    print(f'[DATALOADER] Skipped {skippedFolders} folders')
    print(f'[DATALOADER] Collision folders: {collisionFolders}, collision images: {collisionImages}')

    if len(traj_ims_full) == 0:
        raise ValueError("No valid trajectories found in dataset")

    print("[ANALYZER] Analyzing the data....")
    traj_lengths = np.array([traj_ims.shape[0] for traj_ims in traj_ims_full])
    traj_ims_full = np.concatenate(traj_ims_full).reshape(-1, cropHeight, cropWidth)
    traj_meta_full = np.concatenate(traj_meta_full).reshape(-1, traj_meta_full[0].shape[-1])
    desired_vels = np.array(desired_vels)
    curr_quats = np.array(curr_quats)

    # 处理交通数据
    if use_traffic and traffic_data_full:
        traffic_data_full = np.concatenate(traffic_data_full).reshape(-1, 14)
    else:
        # 如果不使用交通数据，创建全零数组
        traffic_data_full = np.zeros((len(desired_vels), 14))

    # 标准化处理（针对br_cmd数据，第17-20列）
    if traj_meta_full.shape[1] > 20:
        for i in range(4):
            col_idx = 16 + i
            if col_idx < traj_meta_full.shape[1]:
                mean = np.mean(traj_meta_full[:, col_idx])
                std = np.std(traj_meta_full[:, col_idx])
                if std == 0:
                    traj_meta_full[:, col_idx] = 0
                else:
                    traj_meta_full[:, col_idx] = (traj_meta_full[:, col_idx] - mean) / (2 * std)

    curr_ctbr = traj_meta_full[:, 16:20] if traj_meta_full.shape[1] > 20 else np.zeros((len(desired_vels), 4))

    # 对traffic data做手动归一化
    bound_per_row = [50, 50, 5, 1, 1, 1, 1,
                     50, 50, 5, 1, 1, 1, 10]# 分别是x, y, z, vx, vy, vz, radius
    
    for i in range(14):
        traffic_data_full[:, i] = traffic_data_full[:, i] / bound_per_row[i]

    # 训练验证集分割
    num_val_trajs = int(val_split * len(traj_lengths))
    val_idx = np.sum(traj_lengths[:num_val_trajs], dtype=np.int32)
    
    traj_meta_val = traj_meta_full[:val_idx]
    traj_meta_train = traj_meta_full[val_idx:]
    traj_ims_val = traj_ims_full[:val_idx]
    traj_ims_train = traj_ims_full[val_idx:]
    traj_lengths_val = traj_lengths[:num_val_trajs]
    traj_lengths_train = traj_lengths[num_val_trajs:]

    desired_vels_val = desired_vels[:val_idx]
    desired_vels_train = desired_vels[val_idx:]
    curr_quats_val = curr_quats[:val_idx]
    curr_quats_train = curr_quats[val_idx:]
    curr_ctbr_val = curr_ctbr[:val_idx]
    curr_ctbr_train = curr_ctbr[val_idx:]
    
    traffic_data_val = traffic_data_full[:val_idx]
    traffic_data_train = traffic_data_full[val_idx:]

    # 返回数据，包含交通数据
    train_data = (traj_meta_train, traj_ims_train, traj_lengths_train, desired_vels_train, curr_quats_train, curr_ctbr_train, traffic_data_train)
    val_data = (traj_meta_val, traj_ims_val, traj_lengths_val, desired_vels_val, curr_quats_val, curr_ctbr_val, traffic_data_val)
    
    return train_data, val_data, 1, (traj_folders[num_val_trajs:], traj_folders[:num_val_trajs])

def preload(items, device='cpu'):
    """Convert numpy arrays to torch tensors and move to device"""
    return [torch.from_numpy(item).to(device).float() for item in items] 