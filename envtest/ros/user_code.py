#!/usr/bin/python3

from utils import AgileCommandMode, AgileCommand
from scipy.spatial.transform import Rotation
import cv2
import numpy as np
import torch
from torchvision.transforms import ToTensor

import glob, os, sys, time
from os.path import join as opj

sys.path.append(opj(os.path.dirname(os.path.abspath(__file__)), '../../models'))
from model import *

def load_traffic_norm_params(model_dir):
    """
    Load normalization parameters from model directory
    
    Args:
        model_dir: Directory containing the trained model and norm params
    
    Returns:
        dict: Normalization parameters or None if not found
    """
    import pickle
    norm_params_path = opj(model_dir, 'norm_params.pkl')  # Updated path for new format
    
    if os.path.exists(norm_params_path):
        try:
            with open(norm_params_path, 'rb') as f:
                norm_params = pickle.load(f)
            print("[NORM_PARAMS] Loaded normalization parameters from {}".format(norm_params_path))
            print("[NORM_PARAMS] Method: {}, Keys: {}".format(
                norm_params.get('method', 'unknown'), 
                list(norm_params.keys())
            ))
            return norm_params
        except Exception as e:
            print("[NORM_PARAMS] Error loading normalization parameters: {}".format(e))
            return None
    else:
        # Try old format for backward compatibility
        old_norm_params_path = opj(model_dir, 'traffic_norm_params.pkl')
        if os.path.exists(old_norm_params_path):
            try:
                with open(old_norm_params_path, 'rb') as f:
                    old_norm_params = pickle.load(f)
                print("[NORM_PARAMS] Loaded old format normalization parameters from {}".format(old_norm_params_path))
                return old_norm_params
            except Exception as e:
                print("[NORM_PARAMS] Error loading old format parameters: {}".format(e))
                return None
        else:
            print("[NORM_PARAMS] No normalization parameters found at {} or {}".format(norm_params_path, old_norm_params_path))
            return None

def apply_traffic_normalization(traffic_data, norm_params):
    """
    Apply normalization to traffic data using saved parameters
    
    Args:
        traffic_data: Raw traffic data (shape depends on model type)
        norm_params: Normalization parameters dict
    
    Returns:
        Normalized traffic data
    """
    if norm_params is None:
        print("[NORM_PARAMS] Using fallback manual normalization")
        # Fallback to manual normalization for old format
        if len(traffic_data.shape) == 1 and len(traffic_data) == 14:
            # Old 14-dim format
            bound_per_row = [50, 50, 5, 5, 5, 1, 5, 50, 50, 5, 5, 5, 1, 10]
            normalized_data = traffic_data.copy()
            for i in range(min(14, len(normalized_data))):
                normalized_data[i] = normalized_data[i] / bound_per_row[i]
            return normalized_data
        else:
            # New format - simple scaling
            return traffic_data / 50.0
    
    if norm_params.get('method') == 'zscore':
        # Check if we have traffic-specific normalization
        if 'traffic' in norm_params:
            traffic_norm = norm_params['traffic']
            mean = traffic_norm['mean']
            std = traffic_norm['std']
            normalized_data = (traffic_data - mean) / std
            return normalized_data
        else:
            # Old format - direct mean/std
            mean = norm_params['mean']
            std = norm_params['std']
            normalized_data = (traffic_data - mean) / std
            return normalized_data
    else:
        print("[NORM_PARAMS] Unknown normalization method: {}".format(norm_params.get('method')))
        return traffic_data

# 3D line determined by two points (x1, y1, z1) and (x2, y2, z2)
# sphere determined by a center point (x3, y3, z3) and radius r
# quantity b^2 - 4ac < 0 then there is no intersection, where:
# b = 2*( (x2-x1)*(x1-x3) + (y2-y1)*(y1-y3) + (z2-z1)*(z1-z3) )
# a = (x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2
# c = x3^2 + y3^2 + z3^2 + x1^2 + y1^2 + z1^2 - 2*(x3*x1 + y3*y1 + z3*z1) - r^2
# line is a 2-tuple of 3-tuples, obstacle is a 2-tuple of the center 3-tuple and the radius float
def check_collision(line, obstacle):
    (x1, y1, z1), (x2, y2, z2) = line
    (x3, y3, z3), r = obstacle
    b = 2 * ((x2 - x1) * (x1 - x3) + (y2 - y1) * (y1 - y3) + (z2 - z1) * (z1 - z3))
    a = (x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2
    c = (
        x3**2
        + y3**2
        + z3**2
        + x1**2
        + y1**2
        + z1**2
        - 2 * (x3 * x1 + y3 * y1 + z3 * z1)
        - r**2
    )
    return b**2 - 4 * a * c >= 0


def compute_command_vision_based(state, orig_img, prev_img, desiredVel, trained_model, hidden_state):
    # print("Computing command vision-based!")

    """
    # Example of SRT command
    command_mode = 0
    command = AgileCommand(command_mode)
    command.t = state.t
    command.rotor_thrusts = [1.0, 1.0, 1.0, 1.0]

    # Example of CTBR command
    command_mode = 1
    command = AgileCommand(command_mode)
    command.t = state.t
    command.collective_thrust = 15.0
    command.bodyrates = [0.0, 0.0, 0.0]
    """

    # Example of LINVEL command (velocity is expressed in world frame)
    command_mode = 2
    command = AgileCommand(command_mode)
    command.t = state.t
    # command.velocity = [1.0, 0.0, 0.0]
    command.yawrate = 0.0
    command.mode = 2
    
    # Get model device
    model_device = next(trained_model.parameters()).device
    
    ###############
    ## Load data ##
    ###############

    q = np.array([state.att[0], state.att[1], state.att[2], state.att[3]])
    
    h, w = (60, 90)
    img = cv2.resize(orig_img, (w, h))
    img2 = orig_img.copy() # used for generating debugimg
    img = ToTensor()(np.array(img)).to(model_device)

    if 'LSTMNet' in trained_model.__class__.__name__:
        if trained_model.__class__.__name__ == 'LSTMNet':
            trained_model.lstm.num_layers = 2
            trained_model.lstm.hidden_size = 395
        elif trained_model.__class__.__name__ == 'LSTMNetVIT':
            trained_model.lstm.num_layers = 3
            trained_model.lstm.hidden_size = 128
        elif trained_model.__class__.__name__ == 'UNetConvLSTMNet':
            trained_model.lstm.num_layers = 2
            trained_model.lstm.hidden_size = 200
        else:
            raise Exception ("Incorrect Model specified!!")
        if state.pos[0] < 0.5 or hidden_state is None: 
            hidden_state = (torch.zeros(trained_model.lstm.num_layers, trained_model.lstm.hidden_size).float().to(model_device), torch.zeros(trained_model.lstm.num_layers, trained_model.lstm.hidden_size).float().to(model_device))
        with torch.no_grad():
            x, hidden_state = trained_model([img.view(1, 1, h, w), torch.tensor(desiredVel).view(1, 1).float().to(model_device), torch.tensor(q).view(1,-1).float().to(model_device) ,hidden_state])

    else:

        with torch.no_grad():
            x, hidden_state = trained_model([img.view(1, 1, h, w), torch.tensor(desiredVel).view(1, 1).float().to(model_device), torch.tensor(q).view(1,-1).float().to(model_device)])


    x = x.cpu().squeeze().detach().numpy()
    x[0] = np.clip(x[0], -1, 1)
    x = x/np.linalg.norm(x)
    command.velocity = x*desiredVel

    # manual speedup
    min_xvel_cmd = 1.0
    hardcoded_ctl_threshold = 2.0
    if state.pos[0] < hardcoded_ctl_threshold:
        command.velocity[0] = max(min_xvel_cmd, (state.pos[0]/hardcoded_ctl_threshold)*desiredVel)
    

    # creating debug images,
    # debugimg1 of the stabilized, cropped image with a velocity vector, and 
    # debugimg2 of the original image with the four points used for stabilization

    h, w = img2.shape
    arrow_start = (int(w/2), int(h/2))    
    arrow_end = (int(w/2-command.velocity[1]*(w/3)), int(h/2-command.velocity[2]*(h/3)))
    debugimg1 = cv2.arrowedLine( img2, arrow_start, arrow_end, (0, 0, 255), 10, )

    debugimg2 = orig_img.copy()

    return command, (debugimg1, debugimg2), hidden_state


def compute_command_vision_based_with_traffic(state, orig_img, prev_img, desiredVel, trained_model, hidden_state, traffic_data=None, traffic_norm_params=None):
    """
    Enhanced version of compute_command_vision_based that supports traffic information
    for LSTMNetVIT_Traffic, LSTMNetVIT_NoTraffic, LSTMNetVIT_GAT, and LSTMNetVIT_NoTraffic_2D models
    
    Args:
        state: Current quadrotor state
        orig_img: Original depth image
        prev_img: Previous depth image (unused but kept for compatibility)
        desiredVel: Desired velocity magnitude
        trained_model: Neural network model (LSTMNetVIT_Traffic, LSTMNetVIT_NoTraffic, LSTMNetVIT_GAT, or LSTMNetVIT_NoTraffic_2D)
        hidden_state: LSTM hidden state
        traffic_data: Traffic information array (shape depends on model type)
        traffic_norm_params: Normalization parameters for traffic data (optional)
    """
    
    # Example of LINVEL command (velocity is expressed in world frame)
    command_mode = 2
    command = AgileCommand(command_mode)
    command.t = state.t
    command.yawrate = 0.0
    command.mode = 2
    
    # Get model device
    model_device = next(trained_model.parameters()).device
    
    ###############
    ## Load data ##
    ###############

    q = np.array([state.att[0], state.att[1], state.att[2], state.att[3]])
    
    h, w = (60, 90)
    img = cv2.resize(orig_img, (w, h))
    img2 = orig_img.copy() # used for generating debugimg
    img = ToTensor()(np.array(img)).to(model_device)

    # Handle traffic data based on model type
    model_class_name = trained_model.__class__.__name__
    
    if model_class_name in ['LSTMNetVIT_GAT']:
        # For GAT model, traffic_data should be (5, 5) array
        if traffic_data is not None and traffic_data.shape == (5, 5):
            # Apply normalization using saved parameters
            if traffic_norm_params is not None and 'traffic' in traffic_norm_params:
                traffic_norm = traffic_norm_params['traffic']
                mean = traffic_norm['mean']
                std = traffic_norm['std']
                normalized_traffic = (traffic_data - mean) / std
            else:
                # Fallback normalization
                normalized_traffic = traffic_data / 50.0  # Simple scaling
            traffic_tensor = torch.tensor(normalized_traffic).view(1, 5, 5).float().to(model_device)
        else:
            # Create zero traffic data if not provided
            traffic_tensor = torch.zeros(1, 5, 5).float().to(model_device)
            
        # For GAT model, we need robot_states as well
        # Extract robot_states from state (this should be provided in the new format)
        # For now, we'll construct a basic robot_states from available information
        robot_states = np.array([0.0, 0.0, 0.0, desiredVel])  # [goal_x, goal_y, yaw, desired_vel]
        robot_tensor = torch.tensor(robot_states).view(1, 4).float().to(model_device)
        
    elif model_class_name in ['LSTMNetVIT_Traffic', 'LSTMNetVIT_NoTraffic']:
        # For old traffic models, traffic_data should be 14-dimensional
        if traffic_data is not None:
            # Apply normalization using saved parameters
            normalized_traffic = apply_traffic_normalization(traffic_data, traffic_norm_params)
            traffic_tensor = torch.tensor(normalized_traffic).view(1, -1).float().to(model_device)
        else:
            # Create zero traffic data if not provided
            traffic_tensor = torch.zeros(1, 14).float().to(model_device)
    else:
        # For models without traffic
        traffic_tensor = None

    # Initialize hidden state if needed
    if 'LSTMNet' in trained_model.__class__.__name__:
        if trained_model.__class__.__name__ in ['LSTMNetVIT_Traffic', 'LSTMNetVIT_NoTraffic', 'LSTMNetVIT_GAT', 'LSTMNetVIT_NoTraffic_2D']:
            trained_model.lstm.num_layers = 3
            trained_model.lstm.hidden_size = 128
        elif trained_model.__class__.__name__ == 'LSTMNet':
            trained_model.lstm.num_layers = 2
            trained_model.lstm.hidden_size = 395
        elif trained_model.__class__.__name__ == 'LSTMNetVIT':
            trained_model.lstm.num_layers = 3
            trained_model.lstm.hidden_size = 128
        elif trained_model.__class__.__name__ == 'UNetConvLSTMNet':
            trained_model.lstm.num_layers = 2
            trained_model.lstm.hidden_size = 200
        else:
            raise Exception ("Incorrect Model specified: {}".format(trained_model.__class__.__name__))
            
        if state.pos[0] < 0.5 or hidden_state is None: 
            hidden_state = (
                torch.zeros(trained_model.lstm.num_layers, trained_model.lstm.hidden_size).float().to(model_device), 
                torch.zeros(trained_model.lstm.num_layers, trained_model.lstm.hidden_size).float().to(model_device)
            )
    
    # Prepare model inputs based on model type
    if model_class_name == 'LSTMNetVIT_GAT':
        # GAT model expects: [images, robot_states, traffic_states, hidden_state]
        model_inputs = [
            img.view(1, 1, h, w),
            robot_tensor,
            traffic_tensor,
            hidden_state
        ]
    elif model_class_name == 'LSTMNetVIT_NoTraffic_2D':
        # NoTraffic_2D model expects: [images, robot_states, hidden_state]
        robot_states = np.array([0.0, 0.0, 0.0, desiredVel])  # [goal_x, goal_y, yaw, desired_vel]
        robot_tensor = torch.tensor(robot_states).view(1, 4).float().to(model_device)
        model_inputs = [
            img.view(1, 1, h, w),
            robot_tensor,
            hidden_state
        ]
    else:
        # Old models: [images, desiredVel, quaternion, traffic_data, hidden_state]
        model_inputs = [
            img.view(1, 1, h, w), 
            torch.tensor(desiredVel).view(1, 1).float().to(model_device), 
            torch.tensor(q).view(1, -1).float().to(model_device)
        ]
        
        # Add traffic data for traffic-enabled models
        if hasattr(trained_model, 'use_traffic') and trained_model.use_traffic:
            model_inputs.append(traffic_tensor)
        
        # Add hidden state if LSTM model
        if hidden_state is not None:
            model_inputs.append(hidden_state)
    
    # Forward pass
    with torch.no_grad():
        x, hidden_state = trained_model(model_inputs)

    # Process output based on model type
    x = x.cpu().squeeze().detach().numpy()
    
    if model_class_name in ['LSTMNetVIT_GAT', 'LSTMNetVIT_NoTraffic_2D']:
        # These models output 2D velocity commands (normalized by desired_vel during training)
        # We need to denormalize by multiplying by desired_vel
        x = x * desiredVel
        # Ensure minimum forward velocity
        x[0] = max(1.0, x[0])
        command.velocity = [x[0], x[1], 0.0]  # 2D velocity in body frame
    else:
        # Old models: 3D velocity with normalization
        x[0] = np.clip(x[0], -1, 1)
        x = x/np.linalg.norm(x)
        command.velocity = x*desiredVel

    # manual speedup
    min_xvel_cmd = 1.0
    hardcoded_ctl_threshold = 2.0
    if state.pos[0] < hardcoded_ctl_threshold:
        command.velocity[0] = max(min_xvel_cmd, (state.pos[0]/hardcoded_ctl_threshold)*desiredVel)
    
    # creating debug images,
    # debugimg1 of the stabilized, cropped image with a velocity vector, and 
    # debugimg2 of the original image with the four points used for stabilization

    h, w = img2.shape
    arrow_start = (int(w/2), int(h/2))    
    arrow_end = (int(w/2-command.velocity[1]*(w/3)), int(h/2-command.velocity[2]*(h/3)))
    debugimg1 = cv2.arrowedLine( img2, arrow_start, arrow_end, (0, 0, 255), 10, )

    debugimg2 = orig_img.copy()

    return command, (debugimg1, debugimg2), hidden_state

# helper function for vectorized expert policy (method_id = 1)
def find_closest_zero_index(arr):
    center = np.array(arr.shape) // 2  # find the center point of the array
    dist_to_center = np.abs(np.indices(arr.shape) - center.reshape(-1, 1, 1)).sum(0)  # calculate distance to center for each element
    zero_indices = np.argwhere(arr == 0)  # find indices of all zero elements
    if len(zero_indices) == 0:
        return None  # if no zero elements, return None
    dist_to_zeros = dist_to_center[tuple(zero_indices.T)]  # get distances to center for zero elements
    min_dist_indices = np.argwhere(dist_to_zeros == dist_to_zeros.min()).flatten()  # find indices of zero elements with minimum distance to center
    chosen_index = np.random.choice(min_dist_indices)  # randomly choose one of the zero elements with minimum distance to center
    return tuple(zero_indices[chosen_index])  # return index tuple

def compute_command_state_based(state, obstacles, desiredVel, rl_policy=None, keyboard=False, keyboard_input=''):
    # print("Computing command based on obstacle information!")
    # print("Obstacles: ", obstacles)

    """
    # Example of SRT command
    command_mode = 0
    command = AgileCommand(command_mode)
    command.t = state.t
    command.rotor_thrusts = [1.0, 1.0, 1.0, 1.0]

    # Example of CTBR command
    command_mode = 1
    command = AgileCommand(command_mode)
    command.t = state.t
    command.collective_thrust = 10.0
    command.bodyrates = [0.0, 0.0, 0.0]
    """

    # LINVEL command (velocity is expressed in world frame)
    command_mode = 2
    command = AgileCommand(command_mode)
    command.t = state.t
    command.yawrate = 0.0

    obst_dist_threshold = 8
    obst_inflate_factor = 0.6 #0.4#0.6
    method_id = 1 # 0 = old spiral method, 1 = new re-factored, 2 = constant
    if keyboard:
        import select
        method_id = 3

    # calculate an obstacle-free waypoint
    x_displacement = 8 #5
    grid_center_offset = 8
    grid_displacement = 0.5
    y_vals = np.arange(-grid_center_offset, grid_center_offset + grid_displacement, grid_displacement)
    num_wpts = y_vals.size

    start = time.time()

    # old expert
    if method_id == 0:

        wpts_2d = np.zeros((num_wpts, num_wpts, 2))
        for xi, x in enumerate(np.arange(grid_center_offset, -grid_center_offset-grid_displacement, -grid_displacement)):
            for yi, y in enumerate(np.arange(grid_center_offset, -grid_center_offset-grid_displacement, -grid_displacement)):
                wpts_2d[yi, xi] = [x, y]

        # the first layer of wpts_2d is actually the world y axis, the second is z axis
        # the third, the x axis, should all be +5m forward
        x_slice = x_displacement * np.ones((num_wpts, num_wpts))
        wpts_2d = np.concatenate((x_slice[:, :, None], wpts_2d), axis=2)

        # try spiraling outward again but just using bounds instead, and selecting blocks
        idx_midpt = num_wpts // 2
        curr_x = idx_midpt
        curr_y = idx_midpt
        x_bound = 1
        y_bound = -1
        wpt_idxs_2d = []
        count = 0
        while curr_x < num_wpts:
            if count % 4 == 0:
                x_bound = count / 4 + 1
            if (count - 1) % 4 == 0:
                y_bound = -((count - 1) / 4 + 1)

            if not count % 2:  # x-dir vector
                xvals = np.arange(
                    curr_x, idx_midpt + x_bound, -1 if x_bound < 0 else 1, dtype=int
                )
                wpt_idxs_2d += [
                    pair for pair in zip(np.repeat(int(curr_y), xvals.size), xvals)
                ]
                curr_x = idx_midpt + x_bound
                x_bound *= -1
            else:  # y-dir vector
                yvals = np.arange(
                    curr_y, idx_midpt + y_bound, -1 if y_bound < 0 else 1, dtype=int
                )
                wpt_idxs_2d += [
                    pair for pair in zip(yvals, np.repeat(int(curr_x), yvals.size))
                ]
                curr_y = idx_midpt + y_bound
                y_bound *= -1

            count += 1

        # iterate through waypoints, spiraling outwards from center
        for wpt_idx in wpt_idxs_2d:
            found_valid_pt = True
            # check if the current wpt is valid for all obstacles ahead of our current position
            for obst in [obst for obst in obstacles.obstacles if obst.position.x > 0 and obst.position.x < obst_dist_threshold]:
                if check_collision(((0, 0, 0), (wpts_2d[wpt_idx])), ((obst.position.x, obst.position.y, obst.position.z), obst.scale+obst_inflate_factor)):
                    found_valid_pt = False
                    break
            if found_valid_pt:
                break

        # CHECK AGAIN WITH OBSTACLE SCALE REDUCED TO .17
        if not found_valid_pt:
            print("[EXPERT] Didn't find a feasible path, Searching again with less inflation!")
            for wpt_idx in wpt_idxs_2d:
                found_valid_pt = True
                # check if the current wpt is valid for all obstacles ahead of our current position
                for obst in [obst for obst in obstacles.obstacles if obst.position.x > 0 and obst.position.x < obst_dist_threshold]:
                    if check_collision(((0, 0, 0), (wpts_2d[wpt_idx])), ((obst.position.x, obst.position.y, obst.position.z), obst.scale+0.17)):
                        found_valid_pt = False
                        break
                if found_valid_pt:
                    break
        
        # simplest controller: waypoint --PID--> linear velocity command
        yvel = 1.25 * (wpts_2d[wpt_idx][1])
        # x_scale_down_factor = (grid_center_offset - np.abs(yvel))/grid_center_offset
        xvel = max(desiredVel, 1 * (wpts_2d[wpt_idx][0]))
        zvel = 1.25 * wpts_2d[wpt_idx][2]

    # new expert
    elif method_id == 1:

        wpts_2d = np.zeros((num_wpts, num_wpts, 3))
        collisions = np.zeros((num_wpts, num_wpts))
        for xi, x in enumerate(np.arange(grid_center_offset, -grid_center_offset-grid_displacement, -grid_displacement)):
            for yi, y in enumerate(np.arange(grid_center_offset, -grid_center_offset-grid_displacement, -grid_displacement)):
                wpts_2d[yi, xi] = [x_displacement, x, y]
                for obst in [obst for obst in obstacles.obstacles if obst.position.x > 0 and obst.position.x < obst_dist_threshold]:
                    # print(f'wpt: {wpts_2d[yi, xi]} \t obst: {obst.position.x, obst.position.y, obst.position.z, obst.scale+obst_inflate_factor}')
                    if check_collision(((0, 0, 0), (wpts_2d[yi, xi])), ((obst.position.x, obst.position.y, obst.position.z), obst.scale+obst_inflate_factor)):
                        collisions[yi, xi] = 1
                        break


        if collisions.sum() == collisions.size:
            print(f'[EXPERT] No collision-free path found')
            xvel = 0.5
            yvel = 0
            zvel = 0.25            
        else:
            wpt_idx = find_closest_zero_index(collisions)
            wpt = wpts_2d[wpt_idx[0], wpt_idx[1]]

            # make the desired velocity vector of magnitude desiredVel
            wpt = (wpt / np.linalg.norm(wpt)) * desiredVel
            xvel = wpt[0]
            yvel = wpt[1]
            zvel = wpt[2]

    # just fly forward
    elif method_id == 2:

        xvel, yvel, zvel = (4.0, 0., 0.)

    elif method_id == 3:

        xvel, yvel, zvel = (2., 0., 0.)

        # print(f'[EXPERT] Keyboard input: {keyboard_input}')

        # Check if there is any keypress
        if keyboard_input == 'w':
            zvel = 1.0
        elif keyboard_input == 's':
            zvel = -1.0
        elif keyboard_input == 'a':
            yvel = 1.0
        elif keyboard_input == 'd':
            yvel = -1.0

        # norm the command vector up to desiredVel
        scaler = desiredVel/np.linalg.norm([xvel, yvel, zvel])
        xvel, yvel, zvel = (xvel*scaler, yvel*scaler, zvel*scaler)


    if time.time() - int(time.time()) < 0.1: # print this as infrequently as possible
        print(f'[EXPERT] Expert method {method_id} took {time.time() - start:.3f} seconds')

    command.velocity = [xvel, yvel, zvel]

    # recover altitude if too low
    if state.pos[2] < 2:
        command.velocity[2] = (2 - state.pos[2]) * 2

    # manual speedup
    min_xvel_cmd = 1.0
    hardcoded_ctl_threshold = 2.0
    if state.pos[0] < hardcoded_ctl_threshold:
        command.velocity[0] = max(min_xvel_cmd, (state.pos[0]/hardcoded_ctl_threshold)*desiredVel)
        

    ################################################
    # !!! End !!!
    ###############################################

    return command
