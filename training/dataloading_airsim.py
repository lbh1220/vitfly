"""
@authors: Modified for AirSim dataset with PKL format
@organization: GRASP Lab, University of Pennsylvania
@date: ...
@license: ...

@brief: This module contains the dataloading routine for AirSim dataset in PKL format
"""

import cv2
import glob, os, time
from os.path import join as opj
import numpy as np
import torch
import random
import pickle
import getpass
uname = getpass.getuser()

def dataloader_airsim(data_dir, val_split=0., short=0, seed=None, train_val_dirs=None, use_traffic=True):
    """
    AirSim dataset dataloader for PKL format
    
    Args:
        data_dir: Path to dataset root (e.g., /dataset_root/Drone1)
        val_split: Fraction of data to use for validation
        short: If nonzero, limit number of trajectory folders to load
        seed: Random seed for reproducibility
        train_val_dirs: Pre-defined train/val split directories
        use_traffic: Whether to include traffic data (always True for new format)
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
        assert short <= len(traj_folders), "short={} is greater than the number of folders={}".format(short, len(traj_folders))
        traj_folders = traj_folders[:short]
    
    # Data storage for new format
    all_robot_states = []
    all_traffic_states = []
    all_images = []
    all_actions = []
    traj_lengths = []
    
    start_dataloading = time.time()

    skippedFolders = 0

    for i, traj_folder in enumerate(traj_folders):
        if len(traj_folders)//10 > 0 and i % (len(traj_folders)//10) == 0:
            print('[DATALOADER] Loading folder {}, folder # {}/{}, time elapsed {:.2f}s'.format(
                os.path.basename(traj_folder), i+1, len(traj_folders), time.time()-start_dataloading))
        
        # 1. Load trajectory PKL file
        pkl_file = opj(traj_folder, 'vitfly.pkl')
        if not os.path.exists(pkl_file):
            print('[DATALOADER] No vitfly.pkl found in {}, skipping'.format(os.path.basename(traj_folder)))
            skippedFolders += 1
            continue
        
        try:
            with open(pkl_file, 'rb') as f:
                trajectory_data = pickle.load(f)
        except Exception as e:
            print('[DATALOADER] Error loading {}: {}, skipping'.format(pkl_file, e))
            skippedFolders += 1
            continue
        
        observations = trajectory_data.get('observations', [])
        if len(observations) < 10:
            print('[DATALOADER] Trajectory too short, skipping {}'.format(os.path.basename(traj_folder)))
            skippedFolders += 1
            continue
        
        # Sort observations by timestamp to ensure temporal order
        observations.sort(key=lambda x: x['timestamp'])
        
        # Debug: Print first few timestamps to verify ordering
        # if len(observations) > 0:
        #     print('[DATALOADER] First 5 timestamps after sorting: {}'.format([obs['timestamp'] for obs in observations[:5]]))
        
        # 2. Check if corresponding images exist
        depth_dir = opj(traj_folder, 'CAM_FRONT_DEPTH')
        if not os.path.isdir(depth_dir):
            print('[DATALOADER] No CAM_FRONT_DEPTH directory in {}, skipping'.format(os.path.basename(traj_folder)))
            skippedFolders += 1
            continue
        
        # 3. Load images and extract data
        traj_robot_states = []
        traj_traffic_states = []
        traj_images = []
        traj_actions = []
        
        for obs in observations:
            # Load corresponding depth image
            image_filename = obs.get('image_filename', '{:.3f}.png'.format(obs['timestamp']))
            image_path = opj(depth_dir, image_filename)
            
            if not os.path.exists(image_path):
                continue
            
            try:
                # Load and process image
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                
                # Normalize and resize
                img = np.clip(img.astype(np.float32) / 255.0, 0, 1)
                img = cv2.resize(img, (cropWidth, cropHeight))
                
                # Extract observation data
                robot_states = obs['robot_states']        # (4,) array
                traffic_states = obs['traffic_states']    # (5, 5) array
                action = obs['action']                    # (2,) array
                
                traj_robot_states.append(robot_states)
                traj_traffic_states.append(traffic_states)
                traj_images.append(img)
                traj_actions.append(action)
                
            except Exception as e:
                print('[DATALOADER] Error processing observation in {}: {}'.format(os.path.basename(traj_folder), e))
                continue
        
        if len(traj_images) == 0:
            print('[DATALOADER] No valid images found in {}, skipping'.format(os.path.basename(traj_folder)))
            skippedFolders += 1
            continue
        
        # Convert to numpy arrays
        traj_robot_states = np.array(traj_robot_states)      # (T, 4)
        traj_traffic_states = np.array(traj_traffic_states)  # (T, 5, 5)
        traj_images = np.array(traj_images)                  # (T, H, W)
        traj_actions = np.array(traj_actions)                # (T, 2)
        
        # Append to global lists
        all_robot_states.append(traj_robot_states)
        all_traffic_states.append(traj_traffic_states)
        all_images.append(traj_images)
        all_actions.append(traj_actions)
        traj_lengths.append(len(traj_images))

    print('[DATALOADER] Skipped {} folders'.format(skippedFolders))

    if len(all_images) == 0:
        raise ValueError("No valid trajectories found in dataset")

    print("[ANALYZER] Analyzing the data....")
    traj_lengths = np.array(traj_lengths)
    
    # Concatenate all data
    all_robot_states = np.concatenate(all_robot_states, axis=0)    # (N, 4)
    all_traffic_states = np.concatenate(all_traffic_states, axis=0) # (N, 5, 5)
    all_images = np.concatenate(all_images, axis=0)                # (N, H, W)
    all_actions = np.concatenate(all_actions, axis=0)              # (N, 2)
    
    print("[DATALOADER] Total data points: {}".format(len(all_images)))
    print("[DATALOADER] Robot states shape: {}".format(all_robot_states.shape))
    print("[DATALOADER] Traffic states shape: {}".format(all_traffic_states.shape))
    print("[DATALOADER] Images shape: {}".format(all_images.shape))
    print("[DATALOADER] Actions shape: {}".format(all_actions.shape))

    # Save original desired_vel for action normalization before normalizing robot_states
    original_desired_vel = all_robot_states[:, -1].copy()  # (N,)
    
    # Data normalization for robot states
    # Robot states: [goal_x, goal_y, yaw, desired_vel]
    robot_norm_params = {
        'mean': np.mean(all_robot_states, axis=0),
        'std': np.std(all_robot_states, axis=0)
    }
    
    # Avoid division by zero
    robot_norm_params['std'] = np.where(robot_norm_params['std'] == 0, 1.0, robot_norm_params['std'])
    
    # Normalize robot states
    all_robot_states = (all_robot_states - robot_norm_params['mean']) / robot_norm_params['std']
    
    # Normalize actions by original desired_vel
    # Reshape desired_vel to match action dimensions for broadcasting
    desired_vel_reshaped = original_desired_vel.reshape(-1, 1)  # (N, 1)
    all_actions = all_actions / (desired_vel_reshaped + 1e-8)  # Avoid division by zero
    
    print("[DATALOADER] Action normalization by desired_vel:")
    print("[DATALOADER]   Original desired_vel range: [{:.3f}, {:.3f}]".format(
        original_desired_vel.min(), original_desired_vel.max()))
    print("[DATALOADER]   Normalized actions range: [{:.3f}, {:.3f}]".format(
        all_actions.min(), all_actions.max()))
    
    # Data normalization for traffic states
    # Traffic states: (N, 5, 5) where 5 = [rel_x, rel_y, vel_x, vel_y, radius]
    traffic_norm_params = {
        'mean': np.mean(all_traffic_states, axis=(0, 1)),  # Mean over (N, 5) -> (5,)
        'std': np.std(all_traffic_states, axis=(0, 1))     # Std over (N, 5) -> (5,)
    }
    
    # Avoid division by zero
    traffic_norm_params['std'] = np.where(traffic_norm_params['std'] == 0, 1.0, traffic_norm_params['std'])
    
    # Normalize traffic states
    all_traffic_states = (all_traffic_states - traffic_norm_params['mean']) / traffic_norm_params['std']
    
    print("[DATALOADER] Robot normalization - mean: {}, std: {}".format(robot_norm_params['mean'], robot_norm_params['std']))
    print("[DATALOADER] Traffic normalization - mean: {}, std: {}".format(traffic_norm_params['mean'], traffic_norm_params['std']))

    # Train/validation split
    num_val_trajs = int(val_split * len(traj_lengths))
    val_idx = np.sum(traj_lengths[:num_val_trajs], dtype=np.int32)
    
    # Split data
    train_robot_states = all_robot_states[val_idx:]
    train_traffic_states = all_traffic_states[val_idx:]
    train_images = all_images[val_idx:]
    train_actions = all_actions[val_idx:]
    train_traj_lengths = traj_lengths[num_val_trajs:]
    
    val_robot_states = all_robot_states[:val_idx]
    val_traffic_states = all_traffic_states[:val_idx]
    val_images = all_images[:val_idx]
    val_actions = all_actions[:val_idx]
    val_traj_lengths = traj_lengths[:num_val_trajs]

    # Combine normalization parameters
    norm_params = {
        'robot': robot_norm_params,
        'traffic': traffic_norm_params,
        'method': 'zscore'
    }
    
    # Return data in format compatible with training loop
    train_data = (train_robot_states, train_traffic_states, train_images, train_actions, train_traj_lengths)
    val_data = (val_robot_states, val_traffic_states, val_images, val_actions, val_traj_lengths)
    
    return train_data, val_data, 1, (traj_folders[num_val_trajs:], traj_folders[:num_val_trajs]), norm_params

def preload(items, device='cpu'):
    """Convert numpy arrays to torch tensors and move to device"""
    return [torch.from_numpy(item).to(device).float() for item in items] 