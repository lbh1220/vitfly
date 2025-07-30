#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert AirSim dataset to ViTFly format
Processes data.csv files to convert world frame velocities to body frame
and align timestamps with depth camera data
Also processes traffic data to find nearest UAV and eVTOL
"""

import os
import sys
import numpy as np
import pandas as pd
import argparse
import glob
import json
from typing import List, Tuple, Optional, Dict, Any
import tf.transformations


def world_to_body_velocity_2d(world_vel: np.ndarray, quat: np.ndarray) -> np.ndarray:
    """
    Convert world frame velocity to body frame velocity (2D)
    world_vel: [vx, vy, vz] in world frame
    quat: [x, y, z, w] quaternion representing body orientation
    Returns: [vx, vy, vz] in body frame (2D rotation only)
    """
    # Get yaw angle from quaternion
    roll, pitch, yaw = tf.transformations.euler_from_quaternion([quat[0], quat[1], quat[2], quat[3]])
    
    # Apply 2D rotation (only yaw)
    body_vel = np.array([
        world_vel[0] * np.cos(yaw) + world_vel[1] * np.sin(yaw),
        -world_vel[0] * np.sin(yaw) + world_vel[1] * np.cos(yaw),
        world_vel[2]
    ])
    
    return body_vel


def world_to_body_position_2d(world_pos: np.ndarray, ego_pos: np.ndarray, ego_quat: np.ndarray) -> np.ndarray:
    """
    Convert world frame position to body frame position (2D)
    world_pos: [x, y, z] in world frame
    ego_pos: [x, y, z] ego vehicle position in world frame
    ego_quat: [x, y, z, w] ego vehicle quaternion
    Returns: [x, y, z] relative position in ego body frame (2D rotation only)
    """
    # Get yaw angle from quaternion
    roll, pitch, yaw = tf.transformations.euler_from_quaternion([ego_quat[0], ego_quat[1], ego_quat[2], ego_quat[3]])
    
    # Calculate relative position
    rel_pos = world_pos - ego_pos
    
    # Apply 2D rotation (only yaw)
    body_pos = np.array([
        rel_pos[0] * np.cos(yaw) + rel_pos[1] * np.sin(yaw),
        -rel_pos[0] * np.sin(yaw) + rel_pos[1] * np.cos(yaw),
        rel_pos[2]
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


def find_nearest_aircraft(ego_timestamp: float, ego_pos: np.ndarray, 
                         aircraft_dataframes: Dict[str, pd.DataFrame],
                         aircraft_static_info: Dict[str, Dict],
                         aircraft_type: str) -> Optional[Dict[str, Any]]:
    """
    Find the nearest aircraft of specified type at given timestamp
    ego_timestamp: timestamp of ego vehicle
    ego_pos: ego vehicle position [x, y, z]
    aircraft_dataframes: dictionary of aircraft dataframes
    aircraft_static_info: dictionary of aircraft static information
    aircraft_type: 'uav' or 'evtol'
    Returns: nearest aircraft info or None
    """
    nearest_aircraft = None
    min_distance = float('inf')
    
    for aircraft_name, df in aircraft_dataframes.items():
        # Check if this aircraft is of the specified type
        if aircraft_name not in aircraft_static_info:
            continue
        
        static_info = aircraft_static_info[aircraft_name]
        if static_info.get('aircraft_type') != aircraft_type:
            continue
        
        # Find closest timestamp
        df_timestamps = df['timestamp'].values
        timestamp_diff = np.abs(df_timestamps - ego_timestamp)
        min_diff_idx = np.argmin(timestamp_diff)
        
        # Check if timestamp is close enough (within 0.5 seconds)
        if timestamp_diff[min_diff_idx] > 0.5:
            continue
        
        # Get aircraft position at this timestamp
        row = df.iloc[min_diff_idx]
        aircraft_pos = np.array([row['pos_x'], row['pos_y'], row['pos_z']])
        
        # Calculate 2D distance (ignore z)
        distance_2d = np.sqrt((aircraft_pos[0] - ego_pos[0])**2 + (aircraft_pos[1] - ego_pos[1])**2)
        
        if distance_2d < min_distance:
            min_distance = distance_2d
            nearest_aircraft = {
                'name': aircraft_name,
                'position': aircraft_pos,
                'velocity': np.array([row['vel_x'], row['vel_y'], row['vel_z']]),
                'radius': static_info.get('radius', 1.0),
                'timestamp': row['timestamp'],
                'distance': distance_2d
            }
    
    return nearest_aircraft


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
                          output_csv_path: str, traffic_output_path: str,
                          tolerance: float = 0.1) -> bool:
    """
    Process a single trajectory data file
    data_csv_path: path to data.csv file
    depth_dir: path to CAM_FRONT_DEPTH directory
    traffic_dir: path to traffic directory
    output_csv_path: path to output data_vitfly.csv file
    traffic_output_path: path to output traffic_vitfly.csv file
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
        
        # Step 5: Define ViTFly column order
        vitfly_columns = ['timestamp', 'desired_vel', 
                          'quat_1', 'quat_2', 'quat_3', 'quat_4', 
                          'pos_x', 'pos_y', 'pos_z', 
                          'vel_x', 'vel_y', 'vel_z', 
                          'velcmd_x', 'velcmd_y', 'velcmd_z', 
                          'ct_cmd', 'br_cmd_x', 'br_cmd_y', 'br_cmd_z', 'is_collide']
        
        # Add any missing columns from original data that are not in vitfly_columns
        # original_columns = df.columns.tolist()
        # for col in original_columns:
        #     if col not in vitfly_columns:
        #         vitfly_columns.append(col)
        
        # Step 6: Initialize output data structures
        vitfly_data = []
        traffic_data = []
        
        # Traffic columns
        traffic_columns = [
            'nearest_uav_rel_x', 'nearest_uav_rel_y', 'nearest_uav_rel_z',
            'nearest_uav_vel_x', 'nearest_uav_vel_y', 'nearest_uav_vel_z',
            'nearest_uav_radius',
            'nearest_evtol_rel_x', 'nearest_evtol_rel_y', 'nearest_evtol_rel_z',
            'nearest_evtol_vel_x', 'nearest_evtol_vel_y', 'nearest_evtol_vel_z',
            'nearest_evtol_radius'
        ]
        
        # Step 7: Process each aligned data point
        print("Converting velocities and processing traffic data...")
        
        for data_idx in data_indices:
            # Get original row data
            original_row = df.iloc[data_idx]
            
            # Extract ego vehicle data
            ego_quat = np.array([original_row['quat_1'], original_row['quat_2'], 
                                original_row['quat_3'], original_row['quat_4']])
            ego_pos = np.array([original_row['pos_x'], original_row['pos_y'], original_row['pos_z']])
            ego_timestamp = original_row['timestamp']
            
            # Convert velocities from world frame to body frame
            world_vel = np.array([original_row['vel_x'], original_row['vel_y'], original_row['vel_z']])
            body_vel = world_to_body_velocity_2d(world_vel, ego_quat)
            
            world_velcmd = np.array([original_row['velcmd_x'], original_row['velcmd_y'], original_row['velcmd_z']])
            body_velcmd = world_to_body_velocity_2d(world_velcmd, ego_quat)
            
            # Build ViTFly row with proper column order
            vitfly_row = {}
            
            # Add columns in the specified order
            for col in vitfly_columns:
                if col == 'vel_x':
                    vitfly_row[col] = body_vel[0]
                elif col == 'vel_y':
                    vitfly_row[col] = body_vel[1]
                elif col == 'vel_z':
                    vitfly_row[col] = body_vel[2]
                elif col == 'velcmd_x':
                    vitfly_row[col] = body_velcmd[0]
                elif col == 'velcmd_y':
                    vitfly_row[col] = body_velcmd[1]
                elif col == 'velcmd_z':
                    vitfly_row[col] = body_velcmd[2]
                elif col in ['ct_cmd', 'br_cmd_x', 'br_cmd_y', 'br_cmd_z']:
                    vitfly_row[col] = 0.0  # ViTFly specific columns
                elif col in original_row:
                    vitfly_row[col] = original_row[col]
                else:
                    vitfly_row[col] = 0.0  # Default value for missing columns
            
            vitfly_data.append(vitfly_row)
            
            # Process traffic data
            traffic_row = {col: 0.0 for col in traffic_columns}
            
            # Find nearest UAV
            nearest_uav = find_nearest_aircraft(ego_timestamp, ego_pos, aircraft_dataframes, aircraft_static_info, 'uav')
            if nearest_uav:
                uav_rel_pos = world_to_body_position_2d(nearest_uav['position'], ego_pos, ego_quat)
                uav_body_vel = world_to_body_velocity_2d(nearest_uav['velocity'], ego_quat)
                
                traffic_row['nearest_uav_rel_x'] = uav_rel_pos[0]
                traffic_row['nearest_uav_rel_y'] = uav_rel_pos[1]
                traffic_row['nearest_uav_rel_z'] = uav_rel_pos[2]
                traffic_row['nearest_uav_vel_x'] = uav_body_vel[0]
                traffic_row['nearest_uav_vel_y'] = uav_body_vel[1]
                traffic_row['nearest_uav_vel_z'] = uav_body_vel[2]
                traffic_row['nearest_uav_radius'] = nearest_uav['radius']
            
            # Find nearest eVTOL
            nearest_evtol = find_nearest_aircraft(ego_timestamp, ego_pos, aircraft_dataframes, aircraft_static_info, 'evtol')
            if nearest_evtol:
                evtol_rel_pos = world_to_body_position_2d(nearest_evtol['position'], ego_pos, ego_quat)
                evtol_body_vel = world_to_body_velocity_2d(nearest_evtol['velocity'], ego_quat)
                
                traffic_row['nearest_evtol_rel_x'] = evtol_rel_pos[0]
                traffic_row['nearest_evtol_rel_y'] = evtol_rel_pos[1]
                traffic_row['nearest_evtol_rel_z'] = evtol_rel_pos[2]
                traffic_row['nearest_evtol_vel_x'] = evtol_body_vel[0]
                traffic_row['nearest_evtol_vel_y'] = evtol_body_vel[1]
                traffic_row['nearest_evtol_vel_z'] = evtol_body_vel[2]
                traffic_row['nearest_evtol_radius'] = nearest_evtol['radius']
            
            traffic_data.append(traffic_row)
        
        # Step 8: Create final DataFrames and save
        vitfly_df = pd.DataFrame(vitfly_data)
        traffic_df = pd.DataFrame(traffic_data)
        
        # Save processed data
        vitfly_df.to_csv(output_csv_path, index=True)
        print(f"Saved processed data to: {output_csv_path}")
        print(f"Final data points: {len(vitfly_df)}")
        
        # Save traffic data
        traffic_df.to_csv(traffic_output_path, index=True)
        print(f"Saved traffic data to: {traffic_output_path}")
        
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
        output_csv_path = os.path.join(dataset_root, trajectory_dir, "data_vitfly.csv")
        traffic_output_path = os.path.join(dataset_root, trajectory_dir, "traffic_vitfly.csv")
        
        # Remove existing output files if they exist
        if os.path.exists(output_csv_path):
            os.remove(output_csv_path)
            print(f"Removed existing {output_csv_path}")
        if os.path.exists(traffic_output_path):
            os.remove(traffic_output_path)
            print(f"Removed existing {traffic_output_path}")
        
        if process_trajectory_data(data_csv_path, depth_dir, traffic_dir, output_csv_path, traffic_output_path, tolerance):
            success_count += 1
        else:
            print(f"Failed to process trajectory: {trajectory_dir}")
    
    print(f"\nProcessing complete: {success_count}/{len(trajectory_dirs)} trajectories processed successfully")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Convert AirSim dataset to ViTFly format")
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
        output_csv_path = os.path.join(trajectory_path, "data_vitfly.csv")
        traffic_output_path = os.path.join(trajectory_path, "traffic_vitfly.csv")
        
        if not os.path.exists(data_csv_path):
            print(f"Error: data.csv not found in: {trajectory_path}")
            return
        
        if not os.path.exists(depth_dir):
            print(f"Error: CAM_FRONT_DEPTH directory not found in: {trajectory_path}")
            return
        
        print(f"Processing single trajectory: {args.single_trajectory}")
        process_trajectory_data(data_csv_path, depth_dir, traffic_dir, output_csv_path, traffic_output_path, args.tolerance)
        
    else:
        # Process entire dataset
        process_dataset(args.dataset_root, args.tolerance)


if __name__ == "__main__":
    main()
