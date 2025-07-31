#!/usr/bin/python3
import argparse

import rospy
from dodgeros_msgs.msg import Command
from dodgeros_msgs.msg import QuadState
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
from geometry_msgs.msg import TwistStamped
from sensor_msgs.msg import Image
from std_msgs.msg import Empty, Float32
import tf.transformations as tf_trans

# from rl_example import load_rl_policy
from user_code import compute_command_vision_based, compute_command_state_based
from utils import AgileCommandMode, AgileQuadState

import time
import numpy as np
import pandas as pd
import os, sys
from os.path import join as opj
from copy import deepcopy
import cv2
import torch

sys.path.append(opj(os.path.dirname(os.path.abspath(__file__)), '../../models'))
from model import *

class AgilePilotNode:
    def __init__(self, vision_based=False, model_type=None, model_path=None, desVel=None, keyboard=False):
        print("[RUN_COMPETITION] Initializing agile_pilot_airsim_node...")
        rospy.init_node("agile_pilot_airsim_node", anonymous=False)

        self.vision_based = vision_based
        self.rl_policy = None
        self.publish_commands = False
        self.cv_bridge = CvBridge()
        self.state = None
        self.keyboard = keyboard

        quad_name = "Drone1"

        self.init = 0
        self.col = None
        self.t1 = 0 #Time flag
        self.timestamp = 0 #Time stamp initial
        self.last_valid_img = None #Image that will be logged
        data_log_format = {'timestamp':[],
                           'desired_vel':[],
                           'quat_1':[],
                           'quat_2':[],
                           'quat_3':[],
                           'quat_4':[],
                           'pos_x':[],
                           'pos_y':[],
                           'pos_z':[],
                           'vel_x':[],
                           'vel_y':[],
                           'vel_z':[],
                           'velcmd_x':[],
                           'velcmd_y':[],
                           'velcmd_z':[],
                           'ct_cmd':[],
                           'br_cmd_x':[],
                           'br_cmd_y':[],
                           'br_cmd_z':[],
                           'is_collide': [],
        } 
        self.data_log = pd.DataFrame(data_log_format) # store in the data frame
        self.count = 0 # counter for the csv
        
        # @NOTE: Dont log too fast, I have not tested that
        self.time_interval = .03 #Time interval for logging

        self.data_collection_xrange = [2, 60]

        # make the folder for the epoch
        self.folder = f"train_set/{int(time.time()*100)}" 
        os.mkdir(self.folder)

        self.desiredVel = desVel #self.readVel("velocity.txt") #np.random.uniform(low=2.0, high=3.0)
        print()
        print(f"[RUN_COMPETITION] Desired velocity = {self.desiredVel}")
        print()

        # load trained model here (copied over from user_code.py)
        if model_path is not None:
            print(f"[RUN_COMPETITION] Model loading from {model_path} ...")
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if model_type == 'LSTMNet':
                self.model = LSTMNet().to(self.device).float()
            elif model_type == 'UNetLSTM':
                self.model = UNetConvLSTMNet().to(self.device).float()
            elif model_type == 'ConvNet':
                self.model = ConvNet().to(self.device).float()                
            elif model_type == 'ViT':
                self.model = ViT().to(self.device).float()
            elif model_type == 'ViTLSTM':
                self.model = LSTMNetVIT().to(self.device).float()                
            else:
                print(f'[RUN_COMPETITION] Invalid model_type {model_type}. Exiting.')
                exit()

            # Give full path if possible since the bash script runs from outside the folder
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()

            # Initialize hidden state
            self.model_hidden_state = None

            print(f"[RUN_COMPETITION] Model loaded")
            time.sleep(2)

        self.start_time = 0
        self.logged_time_flag = 0
        self.depth_im_threshold = 100

        self.curr_cmd = None

        # Logic subscribers
        self.start_sub = rospy.Subscriber(
            "/vitfly/" + quad_name + "/start_navigation",
            Empty,
            self.start_callback,
            queue_size=1,
            tcp_nodelay=True,
        )

        # Observation subscribers
        # self.odom_sub = rospy.Subscriber(
        #     "/" + quad_name + "/dodgeros_pilot/state",
        #     QuadState,
        #     self.state_callback,
        #     queue_size=1,
        #     tcp_nodelay=True,
        # )
        self.observation_sub = rospy.Subscriber(
            "/vitfly/"+quad_name+"/v_pref",
            Float32,
            self.observation_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.odom_sub = rospy.Subscriber(
            "/airsim_node/Drone1/odom_local_ned",
            Odometry,
            self.state_callback,
            queue_size=1,
            tcp_nodelay=True,
        )
        self.img_sub = rospy.Subscriber(
            "/airsim_node/Drone1/Camera_front/DepthPlanar",
            Image,
            self.img_callback,
            queue_size=1,
            tcp_nodelay=True,
        )

        self.rgb_img_sub = rospy.Subscriber(
            "/airsim_node/Drone1/Camera_front/Scene",
            Image,
            self.rgb_callback,
            queue_size=1,
            tcp_nodelay=True,
        )


        # Command publishers
        self.linvel_pub = rospy.Publisher(
            "/vitfly/" + quad_name + "/velocity_command",
            TwistStamped,
            queue_size=1,
        )
        self.debug_img1_pub = rospy.Publisher(
            "/debug_img1",
            Image,
            queue_size=1,
        )
        self.debug_img2_pub = rospy.Publisher(
            "/debug_img2",
            Image,
            queue_size=1,
        )
        print("[RUN_COMPETITION] Initialization completed!")

        self.ctr = 0

        self.keyboard_input = ''
        self.got_keypress = 0.0
        self.rgb_img = None

    def rgb_callback(self, img):
        self.rgb_img = self.cv_bridge.imgmsg_to_cv2(img, desired_encoding="passthrough")




    def readVel(self,file):
        with open(file,"r") as f:
            x = f.readlines()
            for i in range(len(x)):
                if i == 0:
                    return float(x[i].split("\n")[0])

    def img_callback(self, img_data):
        self.ctr += 1
        self.prevImg = deepcopy(self.last_valid_img)
        img = self.cv_bridge.imgmsg_to_cv2(img_data, desired_encoding="passthrough")
        img = np.clip(img/self.depth_im_threshold, 0, 1)
                
        if self.prevImg is None:
            self.prevImg = img

        self.last_valid_img = deepcopy(img) if img.min() > 0.0 else self.last_valid_img
        
    def observation_callback(self, obs):
        # rospy.loginfo(f"[RUN_COMPETITION] Get observation from airsim")
        self.desiredVel = obs.data
        if not self.publish_commands:
            return
        if not self.vision_based:
            return
        
        if self.state is None:
            return
        
        # print('[RUN_COMPETITION] calling compute_command_vision_based')
        start_compute_time = time.time()

        command, (debug_img1, debug_img2), self.model_hidden_state = compute_command_vision_based(self.state, self.last_valid_img, self.prevImg,self.desiredVel, self.model, self.model_hidden_state)

        # publish debug images
        self.debug_img1_pub.publish(self.cv_bridge.cv2_to_imgmsg(debug_img1, encoding="passthrough"))
        self.debug_img2_pub.publish(self.cv_bridge.cv2_to_imgmsg(debug_img2, encoding="passthrough"))

        if self.ctr % 30 == 0:
            print(f'[RUN_COMPETITION] compute_command_vision_based took {time.time() - start_compute_time} seconds')

        self.publish_command(command)
        print(f'[RUN_COMPETITION] output: {command.velocity}')

        if self.state.pos[0] < 0.1:
            self.start_time = command.t

        if self.state.pos[0] >= 60 and self.logged_time_flag == 0:
            file = "timeTaken.dat"
            with open(file, "a") as file:
                file.write(str(float(command.t - self.start_time))+"\n")
            self.logged_time_flag = 1
        
        #if we exceed the time interval then save the data
        if (self.state.t - self.t1 > self.time_interval or self.t1==0) and self.state.pos[0] < 63:
            #reset the time flag
            self.t1 = self.state.t

            # Get the current time stamp - instant
            timestamp = round(
                self.state.t, 3
            )  # If you need more hz, you might need to modify this round

            # Save the image by the name of that instant
            cv2.imwrite(f"{self.folder}/{str(timestamp)}.png", (self.last_valid_img*255).astype(np.uint8))

            # Get the collision flag
            if self.col is None:
                self.col = 0
            # Append the data frame
            # @TODO: This needs to be managed better if the number of datapoints exceeds 10,000
            self.data_log.loc[len(self.data_log)] = [
                timestamp,
                self.desiredVel,
                self.state.att[0],
                self.state.att[1],
                self.state.att[2],
                self.state.att[3],
                self.state.pos[0],
                self.state.pos[1],
                self.state.pos[2],
                self.state.vel[0],
                self.state.vel[1],
                self.state.vel[2],
                command.velocity[0],
                command.velocity[1],
                command.velocity[2],
                0.0,
                0.0,
                0.0,
                0.0,
                self.col,
            ]

            # Counter flag for saving the data frame
            self.count += 1
        return
        # Save once every 10 instances - writing every instance can be expensive
        if self.count % 5 == 0:
            self.data_log.to_csv(self.folder + "/data.csv")

    def state_callback(self, odom_data):
        # 如果self.state还未初始化，则初始化为全零
        if self.state is None:
            self.state = AgileQuadState()
        # 时间戳
        self.state.t = odom_data.header.stamp.to_sec()
        # 位置
        self.state.pos[0] = odom_data.pose.pose.position.x
        self.state.pos[1] = odom_data.pose.pose.position.y
        self.state.pos[2] = odom_data.pose.pose.position.z
        # 姿态（四元数，顺序为wxyz）
        self.state.att[0] = odom_data.pose.pose.orientation.w
        self.state.att[1] = odom_data.pose.pose.orientation.x
        self.state.att[2] = odom_data.pose.pose.orientation.y
        self.state.att[3] = odom_data.pose.pose.orientation.z
        # 线速度
        self.state.vel[0] = odom_data.twist.twist.linear.x
        self.state.vel[1] = odom_data.twist.twist.linear.y
        self.state.vel[2] = odom_data.twist.twist.linear.z
        # 角速度
        self.state.omega[0] = odom_data.twist.twist.angular.x
        self.state.omega[1] = odom_data.twist.twist.angular.y
        self.state.omega[2] = odom_data.twist.twist.angular.z


    def publish_command(self, command):
        if command.mode == AgileCommandMode.LINVEL:
            # Transform body frame velocity to world frame
            if self.state is not None:
                # Get quaternion from current state (w, x, y, z)
                quat = [self.state.att[1], self.state.att[2], self.state.att[3], self.state.att[0]]  # (x, y, z, w)
                
                # Get rotation matrix from quaternion
                # rotation_matrix = tf_trans.quaternion_matrix(quat)
                roll, pitch, yaw = tf_trans.euler_from_quaternion(quat)
                
                # Body frame velocity vector
                vel_body = np.array([command.velocity[0], command.velocity[1], command.velocity[2], 1.0])
                
                # Transform to world frame
                #vel_world = rotation_matrix.dot(vel_body)
                vel_world = np.array([
                    command.velocity[0] * np.cos(yaw) - command.velocity[1] * np.sin(yaw),
                    command.velocity[0] * np.sin(yaw) + command.velocity[1] * np.cos(yaw),
                    command.velocity[2]
                ])
                
                # Create TwistStamped message with world frame velocity
                vel_msg = TwistStamped()
                vel_msg.header.stamp = rospy.Time(command.t)
                vel_msg.twist.linear.x = vel_world[0]
                vel_msg.twist.linear.y = vel_world[1]
                vel_msg.twist.linear.z = vel_world[2]
                vel_msg.twist.angular.x = 0.0
                vel_msg.twist.angular.y = 0.0
                vel_msg.twist.angular.z = command.yawrate
                
                # rospy.loginfo(f"[RUN_COMPETITION] Body frame: [{command.velocity[0]:.3f}, {command.velocity[1]:.3f}, {command.velocity[2]:.3f}] -> World frame: [{vel_world[0]:.3f}, {vel_world[1]:.3f}, {vel_world[2]:.3f}]")
                
                if self.publish_commands:
                    self.linvel_pub.publish(vel_msg)
                    return
            else:
                rospy.logwarn("[RUN_COMPETITION] State not available, cannot transform velocity")
        else:
            assert False, "Unknown command mode specified"

    def start_callback(self, data):
        print("[RUN_COMPETITION] Start publishing commands!")
        self.publish_commands = True
        self.model_hidden_state = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Agile Pilot.")
    parser.add_argument("--vision_based", help="Fly vision-based", required=False, dest="vision_based", action="store_true")
    parser.add_argument('--model_type', type=str, default='LSTMNet', help='string matching model name in lstmArch.py')
    parser.add_argument('--model_path', type=str, default=None, help='absolute path to model checkpoint')
    parser.add_argument('--des_vel', type=float, default=None, help='desired velocity for quadrotor')
    parser.add_argument("--keyboard", help="Fly state-based mode but take velocity commands from keyboard WASD", required=False, dest="keyboard", action="store_true")

    args = parser.parse_args()
    agile_pilot_node = AgilePilotNode(vision_based=args.vision_based, model_type=args.model_type, model_path=args.model_path, desVel=args.des_vel, keyboard=args.keyboard)
    rospy.spin()
