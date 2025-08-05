#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert AirSim dataset to ViTFly format with PKL output
Processes data.csv files to convert world frame velocities to body frame
and align timestamps with depth camera data
Also processes traffic data to find nearest aircraft and create graph-ready data
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse
import glob
import json
import pickle
from typing import List, Tuple, Optional, Dict, Any
import tf.transformations


def world_to_body_velocity_2d(world_vel: np.ndarray, yaw: float) -> np.ndarray:
    """
    Convert world frame velocity to body frame velocity (2D)
    world_vel: [vx, vy] in world frame
    yaw: yaw angle in radians
    Returns: [vx, vy] in body frame
    """
    # Apply 2D rotation (only yaw)
    body_vel = np.array([
        world_vel[0] * np.cos(yaw) + world_vel[1] * np.sin(yaw),
        -world_vel[0] * np.sin(yaw) + world_vel[1] * np.cos(yaw)
    ])
    
    return body_vel


def world_to_body_position_2d(world_pos: np.ndarray, ego_pos: np.ndarray, ego_yaw: float) -> np.ndarray:
    """
    Convert world frame position to body frame position (2D)
    world_pos: [x, y] in world frame
    ego_pos: [x, y] ego vehicle position in world frame
    ego_yaw: ego vehicle yaw angle in radians
    Returns: [x, y] relative position in ego body frame
    """
    # Calculate relative position
    rel_pos = world_pos - ego_pos
    
    # Apply 2D rotation (only yaw)
    body_pos = np.array([
        rel_pos[0] * np.cos(ego_yaw) + rel_pos[1] * np.sin(ego_yaw),
        -rel_pos[0] * np.sin(ego_yaw) + rel_pos[1] * np.cos(ego_yaw)
    ])
    
    return body_pos


def get_depth_timestamps(depth_dir: str) -> List[float]:
    """
    Extract timestamps from depth camera files
    depth_dir: path to CAM_FRONT_DEPTH directory
    Returns: list of timestamps as floats
    """
    if not os.path.exists(depth_dir):
        print(f"Warning: Depth directory {depth_dir} does not exist")
        return []
    
    # Get all image files (jpg or png)
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob.glob(os.path.join(depth_dir, ext)))
    
    # Extract timestamps from filenames
    timestamps = []
    for file_path in image_files:
        filename = os.path.basename(file_path)
        # Remove extension
        timestamp_str = os.path.splitext(filename)[0]
        try:
            timestamp = float(timestamp_str)
            timestamps.append(timestamp)
        except ValueError:
            print(f"Warning: Could not parse timestamp from filename: {filename}")
    
    return sorted(timestamps)


def load_traffic_data(traffic_dir: str) -> Tuple[Dict[str, pd.DataFrame], Dict[str, Dict]]:
    """
    Load traffic data from traffic directory
    traffic_dir: path to traffic directory
    Returns: (aircraft_dataframes, aircraft_static_info)
    """
    aircraft_dataframes = {}
    aircraft_static_info = {}
    
    # Load traffic summary
    summary_path = os.path.join(traffic_dir, "traffic_summary.json")
    if not os.path.exists(summary_path):
        print(f"Warning: traffic_summary.json not found in {traffic_dir}")
        return aircraft_dataframes, aircraft_static_info
    
    try:
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        aircraft_registry = summary.get('aircraft_registry', {})
        
        # Load each aircraft's CSV file
        for aircraft_name, aircraft_info in aircraft_registry.items():
            csv_path = os.path.join(traffic_dir, aircraft_info['data_file'])
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    aircraft_dataframes[aircraft_name] = df
                    aircraft_static_info[aircraft_name] = aircraft_info
                    print(f"Loaded traffic data for {aircraft_name}: {len(df)} data points")
                except Exception as e:
                    print(f"Error loading {aircraft_name} data: {e}")
            else:
                print(f"Warning: CSV file not found for {aircraft_name}: {csv_path}")
    
    except Exception as e:
        print(f"Error loading traffic summary: {e}")
    
    return aircraft_dataframes, aircraft_static_info


def find_nearest_n_aircraft(ego_timestamp: float, ego_pos: np.ndarray, ego_yaw: float,
                           aircraft_dataframes: Dict[str, pd.DataFrame],
                           aircraft_static_info: Dict[str, Dict],
                           n: int = 5) -> np.ndarray:
    """
    Find the nearest N aircraft at given timestamp
    ego_timestamp: timestamp of ego vehicle
    ego_pos: ego vehicle position [x, y]
    ego_yaw: ego vehicle yaw angle in radians
    aircraft_dataframes: dictionary of aircraft dataframes
    aircraft_static_info: dictionary of aircraft static information
    n: number of nearest aircraft to find
    Returns: (N, 5) array with [rel_x, rel_y, vel_x, vel_y, radius] for each aircraft
    """
    aircraft_list = []
    
    for aircraft_name, df in aircraft_dataframes.items():
        # Find closest timestamp
        df_timestamps = df['timestamp'].values
        timestamp_diff = np.abs(df_timestamps - ego_timestamp)
        min_diff_idx = np.argmin(timestamp_diff)
        
        # Check if timestamp is close enough (within 0.5 seconds)
        if timestamp_diff[min_diff_idx] > 0.5:
            continue
        
        # Get aircraft data at this timestamp
        row = df.iloc[min_diff_idx]
        aircraft_pos = np.array([row['pos_x'], row['pos_y']])
        aircraft_vel = np.array([row['vel_x'], row['vel_y']])
        
        # Calculate 2D distance
        distance_2d = np.linalg.norm(aircraft_pos - ego_pos)
        
        # Convert to body frame
        rel_pos = world_to_body_position_2d(aircraft_pos, ego_pos, ego_yaw)
        rel_vel = world_to_body_velocity_2d(aircraft_vel, ego_yaw)
        
        # Get radius from static info
        radius = aircraft_static_info.get(aircraft_name, {}).get('radius', 1.0)
        
        aircraft_data = {
            'distance': distance_2d,
            'rel_pos': rel_pos,
            'rel_vel': rel_vel,
            'radius': radius
        }
        aircraft_list.append(aircraft_data)
    
    # Sort by distance and take nearest N
    aircraft_list.sort(key=lambda x: x['distance'])
    nearest_aircraft = aircraft_list[:n]
    
    # Create output array (N, 5) with [rel_x, rel_y, vel_x, vel_y, radius]
    traffic_states = np.zeros((n, 5))
    for i, aircraft in enumerate(nearest_aircraft):
        traffic_states[i, 0] = aircraft['rel_pos'][0]  # rel_x
        traffic_states[i, 1] = aircraft['rel_pos'][1]  # rel_y
        traffic_states[i, 2] = aircraft['rel_vel'][0]  # vel_x
        traffic_states[i, 3] = aircraft['rel_vel'][1]  # vel_y
        traffic_states[i, 4] = aircraft['radius']      # radius
    
    return traffic_states


def align_timestamps(data_timestamps: List[float], depth_timestamps: List[float], 
                    tolerance: float = 0.1) -> Tuple[List[int], List[int]]:
    """
    Align data timestamps with depth timestamps
    data_timestamps: timestamps from data.csv
    depth_timestamps: timestamps from depth camera files
    tolerance: time tolerance for matching (seconds)
    Returns: (data_indices, depth_indices) of matched timestamps
    """
    data_indices = []
    depth_indices = []
    
    for i, data_ts in enumerate(data_timestamps):
        for j, depth_ts in enumerate(depth_timestamps):
            if abs(data_ts - depth_ts) <= tolerance:
                data_indices.append(i)
                depth_indices.append(j)
                break
    
    return data_indices, depth_indices


def process_trajectory_data(data_csv_path: str, depth_dir: str, traffic_dir: str,
                          output_pkl_path: str, tolerance: float = 0.1) -> bool:
    """
    Process a single trajectory data file and save as PKL
    data_csv_path: path to data.csv file
    depth_dir: path to CAM_FRONT_DEPTH directory
    traffic_dir: path to traffic directory
    output_pkl_path: path to output vitfly.pkl file
    tolerance: time tolerance for timestamp alignment
    Returns: True if successful, False otherwise
    """
    try:
        # Step 1: Read original data.csv
        print(f"Reading data from: {data_csv_path}")
        df = pd.read_csv(data_csv_path)
        
        if df.empty:
            print(f"Warning: Empty data file: {data_csv_path}")
            return False
        
        # Step 2: Get depth timestamps for alignment
        depth_timestamps = get_depth_timestamps(depth_dir)
        if not depth_timestamps:
            print(f"Warning: No depth timestamps found in: {depth_dir}")
            return False
        
        print(f"Found {len(df)} data points and {len(depth_timestamps)} depth images")
        
        # Step 3: Load traffic data
        aircraft_dataframes, aircraft_static_info = load_traffic_data(traffic_dir)
        print(f"Loaded traffic data for {len(aircraft_dataframes)} aircraft")
        
        # Step 4: Align timestamps
        data_timestamps = df['timestamp'].tolist()
        data_indices, depth_indices = align_timestamps(data_timestamps, depth_timestamps, tolerance)
        
        if not data_indices:
            print(f"Warning: No matching timestamps found between data and depth images")
            return False
        
        print(f"Aligned {len(data_indices)} timestamps")
        
        # Step 5: Process each aligned data point
        observations = []
        
        print("Converting to new observation format...")
        
        # Create list of (data_idx, timestamp) pairs for sorting
        timestamp_data_pairs = []
        for data_idx in data_indices:
            timestamp = data_timestamps[data_idx]
            timestamp_data_pairs.append((data_idx, timestamp))
        
        # Sort by timestamp to ensure temporal order
        timestamp_data_pairs.sort(key=lambda x: x[1])
        
        # Debug: Print first few timestamps to verify ordering
        if len(timestamp_data_pairs) > 0:
            print(f"First 5 timestamps after sorting: {[p[1] for p in timestamp_data_pairs[:5]]}")
        
        for data_idx, timestamp in timestamp_data_pairs:
            # Get original row data
            original_row = df.iloc[data_idx]
            
            # Extract ego vehicle data
            ego_quat = np.array([original_row['quat_1'], original_row['quat_2'], 
                                original_row['quat_3'], original_row['quat_4']])
            ego_pos = np.array([original_row['pos_x'], original_row['pos_y']])  # 2D position
            ego_timestamp = original_row['timestamp']
            
            # Get yaw angle from quaternion
            _, _, ego_yaw = tf.transformations.euler_from_quaternion([
                ego_quat[0], ego_quat[1], ego_quat[2], ego_quat[3]  # [x, y, z, w]
            ])
            
            # Convert velocities from world frame to body frame (2D)
            world_vel = np.array([original_row['vel_x'], original_row['vel_y']])
            body_vel = world_to_body_velocity_2d(world_vel, ego_yaw)
            
            world_velcmd = np.array([original_row['velcmd_x'], original_row['velcmd_y']])
            body_velcmd = world_to_body_velocity_2d(world_velcmd, ego_yaw)
            
            # Build robot_states: [goal_x, goal_y, yaw, desired_vel]
            # For now, assume goal is in direction of desired velocity
            # # TODO: use local goal from planner
            # goal_direction = body_velcmd / (np.linalg.norm(body_velcmd) + 1e-8)
            # goal_distance = 10.0  # Fixed goal distance
            # goal_x = goal_direction[0] * goal_distance
            # goal_y = goal_direction[1] * goal_distance
            local_goal = np.array([original_row['local_goal_x'], original_row['local_goal_y']])
            local_goal_body = world_to_body_position_2d(local_goal, ego_pos, ego_yaw)
            goal_x = local_goal_body[0]
            goal_y = local_goal_body[1]
            desired_vel = original_row['desired_vel']
            
            robot_states = np.array([goal_x, goal_y, ego_yaw, desired_vel])
            
            # Get traffic_states: (5, 5) array
            traffic_states = find_nearest_n_aircraft(
                ego_timestamp, ego_pos, ego_yaw, 
                aircraft_dataframes, aircraft_static_info, n=5
            )
            
            # Create observation dictionary
            observation = {
                'timestamp': ego_timestamp,
                'robot_states': robot_states,      # (4,) array
                'traffic_states': traffic_states,  # (5, 5) array
                'action': body_velcmd,             # (2,) array - target velocity
                'image_filename': f"{ego_timestamp:.3f}.png"  # corresponding depth image
            }
            
            observations.append(observation)
        
        # Step 6: Save as PKL file
        trajectory_data = {
            'observations': observations,
            'metadata': {
                'num_timesteps': len(observations),
                'robot_state_dim': 4,  # [goal_x, goal_y, yaw, desired_vel]
                'traffic_state_shape': (5, 5),  # (N, 5) where 5 = [rel_x, rel_y, vel_x, vel_y, radius]
                'action_dim': 2,  # [vel_x, vel_y] in body frame
                'coordinate_frame': 'body',
                'data_type': '2d'
            }
        }
        
        with open(output_pkl_path, 'wb') as f:
            pickle.dump(trajectory_data, f)
        
        print(f"Saved processed data to: {output_pkl_path}")
        print(f"Final observations: {len(observations)}")
        
        return True
        
    except Exception as e:
        print(f"Error processing trajectory: {e}")
        return False


def process_dataset(dataset_root: str, tolerance: float = 0.1):
    """
    Process all trajectories in the dataset
    dataset_root: root directory of the dataset (e.g., /path/to/dataset/occ_openair/Drone1)
    tolerance: time tolerance for timestamp alignment
    """
    if not os.path.exists(dataset_root):
        print(f"Error: Dataset root directory does not exist: {dataset_root}")
        return
    
    # Find all trajectory directories
    trajectory_dirs = []
    for item in os.listdir(dataset_root):
        item_path = os.path.join(dataset_root, item)
        if os.path.isdir(item_path):
            data_csv_path = os.path.join(item_path, "data.csv")
            depth_dir = os.path.join(item_path, "CAM_FRONT_DEPTH")
            traffic_dir = os.path.join(item_path, "traffic")
            
            if os.path.exists(data_csv_path) and os.path.exists(depth_dir):
                trajectory_dirs.append(item)
    
    print(f"Found {len(trajectory_dirs)} trajectories to process")
    
    success_count = 0
    for trajectory_dir in trajectory_dirs:
        print(f"\nProcessing trajectory: {trajectory_dir}")
        
        data_csv_path = os.path.join(dataset_root, trajectory_dir, "data.csv")
        depth_dir = os.path.join(dataset_root, trajectory_dir, "CAM_FRONT_DEPTH")
        traffic_dir = os.path.join(dataset_root, trajectory_dir, "traffic")
        output_pkl_path = os.path.join(dataset_root, trajectory_dir, "vitfly.pkl")
        
        # Remove existing output file if it exists
        if os.path.exists(output_pkl_path):
            os.remove(output_pkl_path)
            print(f"Removed existing {output_pkl_path}")
        
        if process_trajectory_data(data_csv_path, depth_dir, traffic_dir, output_pkl_path, tolerance):
            success_count += 1
        else:
            print(f"Failed to process trajectory: {trajectory_dir}")
    
    print(f"\nProcessing complete: {success_count}/{len(trajectory_dirs)} trajectories processed successfully")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Convert AirSim dataset to ViTFly PKL format")
    parser.add_argument("--dataset_root", help="Root directory of the dataset (e.g., /path/to/dataset/occ_openair/Drone1)")
    parser.add_argument("--tolerance", type=float, default=0.1, 
                       help="Time tolerance for timestamp alignment in seconds (default: 0.1)")
    parser.add_argument("--single-trajectory", help="Process only a single trajectory directory")
    
    args = parser.parse_args()
    
    if args.single_trajectory:
        # Process single trajectory
        trajectory_path = os.path.join(args.dataset_root, args.single_trajectory)
        if not os.path.exists(trajectory_path):
            print(f"Error: Trajectory directory does not exist: {trajectory_path}")
            return
        
        data_csv_path = os.path.join(trajectory_path, "data.csv")
        depth_dir = os.path.join(trajectory_path, "CAM_FRONT_DEPTH")
        traffic_dir = os.path.join(trajectory_path, "traffic")
        output_pkl_path = os.path.join(trajectory_path, "vitfly.pkl")
        
        if not os.path.exists(data_csv_path):
            print(f"Error: data.csv not found in: {trajectory_path}")
            return
        
        if not os.path.exists(depth_dir):
            print(f"Error: CAM_FRONT_DEPTH directory not found in: {trajectory_path}")
            return
        
        print(f"Processing single trajectory: {args.single_trajectory}")
        process_trajectory_data(data_csv_path, depth_dir, traffic_dir, output_pkl_path, args.tolerance)
        
    else:
        # Process entire dataset
        process_dataset(args.dataset_root, args.tolerance)


if __name__ == "__main__":
    main()
