"""
@authors: A Bhattacharya
@organization: GRASP Lab, University of Pennsylvania
@date: ...
@license: ...

@brief: This module contains the dataloading routine that was used in the paper "Utilizing vision transformer models for end-to-end vision-based
quadrotor obstacle avoidance" by Bhattacharya, et. al
"""

import cv2
import glob, os, time
from os.path import join as opj
import numpy as np
import torch
import random
import getpass
uname = getpass.getuser()

def dataloader(data_dir, val_split=0., short=0, seed=None, train_val_dirs=None):
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

    start_dataloading = time.time()

    skippedImages = 0
    skippedFolders = 0
    collisionImages = 0
    collisionFolders = 0

    for i, traj_folder in enumerate(traj_folders):
        if len(traj_folders)//10 > 0 and i % (len(traj_folders)//10) == 0:
            print(f'[DATALOADER] Loading folder {os.path.basename(traj_folder)}, folder # {i+1}/{len(traj_folders)}, time elapsed {time.time()-start_dataloading:.2f}s')
        # 1. 先查找轨迹文件夹下的png文件
        im_files = sorted(glob.glob(opj(traj_folder, '*.png')))
        if len(im_files) > 0:
            # 兼容原vitfly方案，只读取不带_rgb的png
            depth_im_files = [f for f in im_files if '_rgb' not in f]
        else:
            # 2. 如果没有png文件，查找depth子文件夹
            depth_dir = opj(traj_folder, 'depth')
            if os.path.isdir(depth_dir):
                depth_im_files = sorted(glob.glob(opj(depth_dir, '*.png')))
            else:
                print(f'[DATALOADER] No images in {os.path.basename(traj_folder)}, skipping')
                continue

        if len(depth_im_files) == 0:
            print(f'[DATALOADER] No depth images found in {os.path.basename(traj_folder)}, skipping')
            continue

        csv_file = 'data.csv'
        # float64 is required to read ros timestamps without rounding
        # NOTE not sure if float64 will break training (torch dtypes)
        traj_meta = np.genfromtxt(opj(traj_folder, csv_file), delimiter=',', dtype=np.float64)[1:]
        traj_meta[:,-1] = np.int32(np.genfromtxt(opj(traj_folder, csv_file), delimiter=',', dtype="bool")[1:,-1])

        # check for collisions in trajectory
        # if traj_meta[:,-1].sum() > 0:
        #     print(f'[DATALOADER] Collision in {os.path.basename(traj_folder)}, skipping')
        #     collisionFolders += 1
        #     collisionImages += int(len(traj_meta[:,0]))
        #     continue

        # check for nan in metadata
        if np.isnan(traj_meta).any():
            print(f'[DATALOADER] NaN in {os.path.basename(traj_folder)}, skipping')
            traj_meta = traj_meta[:,:-1]
            

        # 读取所有图像的时间戳
        img_timestamps = [float(os.path.basename(f)[:-4]) for f in depth_im_files]
        state_timestamps = traj_meta[:, 1]  # 假设第2列是时间戳

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

        # 再次检查数量是否一致
        min_len = min(len(depth_im_files), traj_meta.shape[0])
        depth_im_files = depth_im_files[:min_len]
        traj_meta = traj_meta[:min_len, :]

        if min_len == 0:
            print(f'[DATALOADER] No matched images and states in {os.path.basename(traj_folder)}, skipping')
            continue

        # 读取图像
        traj_ims = np.asarray([cv2.imread(im_file, cv2.IMREAD_GRAYSCALE) for im_file in depth_im_files], dtype=np.float32) / 255.0

        temp = [cv2.resize(img, (cropWidth, cropHeight)) for img in traj_ims]
        traj_ims = np.array(temp)
        for ii in range(traj_meta.shape[0]):
            desired_vels.append(traj_meta[ii, 2])
            q = traj_meta[ii, 3:7]
            rmat = q 
            curr_quats.append(rmat)
        try:
            traj_ims_full.append(traj_ims)
            traj_meta_full.append(traj_meta)
        except:
            print(f'[DATALOADER] {traj_ims.shape}')
            print(f"[DATALOADER] Suspected empty image, folder {os.path.basename(traj_folder)}")

    print(skippedFolders, skippedImages)
    print(collisionFolders, collisionImages)

    print("[ANALYZER] Analyzing the data....")
    traj_lengths = np.array([traj_ims.shape[0] for traj_ims in traj_ims_full])
    traj_ims_full = np.concatenate(traj_ims_full).reshape(-1, cropHeight, cropWidth)
    traj_meta_full = np.concatenate(traj_meta_full).reshape(-1, traj_meta.shape[-1])
    desired_vels = np.array(desired_vels)
    curr_quats = np.array(curr_quats)


    #Col: mean, std
    #row: ct. brx/y/z
    for i in range(4):
        mean = np.mean(traj_meta_full[:, 16 + i])
        std = np.std(traj_meta_full[:, 16 + i])
        if std == 0:
            traj_meta_full[:, 16 + i] = 0
        else:
            traj_meta_full[:, 16 + i] = (traj_meta_full[:, 16 + i] - mean) / (2 * std)

    curr_ctbr = traj_meta_full[:, 16:20]

    #Col: mean, std
    #row: ct. brx/y/z
    for i in range(4):
        mean = np.mean(traj_meta[:, 16 + i])
        std = np.std(traj_meta[:, 16 + i])
        if std == 0:
            traj_meta[:, 16 + i] = 0
        else:
            traj_meta[:, 16 + i] = (traj_meta[:, 16 + i] - mean) / (2 * std)

    # make train-val split (relies on earlier shuffle of traj_folders to randomize selection)
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
    #curr_vels_val = curr_vels[:val_idx]
    #curr_vels_train = curr_vels[val_idx:]
    curr_quats_val = curr_quats[:val_idx]
    curr_quats_train = curr_quats[val_idx:]
    curr_ctbr_val = curr_ctbr[:val_idx]
    curr_ctbr_train = curr_ctbr[val_idx:]

    # Note, we return the is_png=1 flag since it indicates old vs new datasets, which indicates how to parse the metadata
    # We also return the traj_folder names for train and val sets, so that they can be saved and later used to specifically generate evaluate plots on each set
    return (traj_meta_train, traj_ims_train, traj_lengths_train, desired_vels_train, curr_quats_train, curr_ctbr_train), (traj_meta_val, traj_ims_val, traj_lengths_val, desired_vels_val, curr_quats_val, curr_ctbr_val), 1, (traj_folders[num_val_trajs:], traj_folders[:num_val_trajs])

def parse_meta_str(meta_str):

    meta = torch.zeros_like(meta_str)

    meta_str

    return meta


def preload(items, device='cpu'):

    return [torch.from_numpy(item).to(device).float() for item in items]