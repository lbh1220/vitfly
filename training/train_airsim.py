"""
@authors: Modified for AirSim dataset training with GAT models
@organization: GRASP Lab, University of Pennsylvania
@date: ...
@license: ...

@brief: This module contains the training routine for AirSim dataset with GAT-based models
"""

import os, sys
from os.path import join as opj
import numpy as np
import torch
from datetime import datetime
import time
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.nn.functional as F

from dataloading_airsim import dataloader_airsim, preload
sys.path.append(opj(os.path.dirname(os.path.abspath(__file__)), '../models'))
import model_airsim as model_library

# NOTE this suppresses tensorflow warnings and info
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import getpass
uname = getpass.getuser()

class TRAINER_AIRSIM:
    def __init__(self, args=None):
        self.args = args
        if self.args is not None:
            self.device = args.device
            self.basedir = args.basedir
            self.logdir = args.logdir
            self.datadir = args.datadir
            self.ws_suffix = args.ws_suffix
            self.dataset_name = args.dataset
            self.short = args.short

            self.model_type = args.model_type
            self.use_traffic = args.use_traffic
            self.val_split = args.val_split
            self.seed = args.seed
            self.load_checkpoint = args.load_checkpoint
            self.checkpoint_path = args.checkpoint_path
            self.lr = args.lr
            self.N_eps = args.N_eps
            self.lr_warmup_epochs = args.lr_warmup_epochs
            self.lr_decay = args.lr_decay
            self.save_model_freq = args.save_model_freq
            self.val_freq = args.val_freq
        else:
            raise Exception("Args are not provided")

        assert self.dataset_name is not None, 'Dataset name not provided'

        ###############
        ## Workspace ##
        ###############
        traffic_suffix = "_with_traffic" if self.use_traffic else "_no_traffic"
        
        # Add model parameters to workspace name
        if hasattr(self.args, 'embed_dim'):
            model_params_suffix = "_embed{}_gat{}".format(
                getattr(self.args, 'embed_dim', 64),
                getattr(self.args, 'gat_hidden_dim', 128)
            )
            traffic_suffix += model_params_suffix
        
        expname = datetime.now().strftime('d%m_%d_t%H_%M') + traffic_suffix
        self.workspace = opj(self.basedir, self.logdir, expname)
        wkspc_ctr = 2
        while os.path.exists(self.workspace):
            self.workspace = opj(self.basedir, self.logdir, expname+'_{}'.format(str(wkspc_ctr)))
            wkspc_ctr += 1
        self.workspace = self.workspace + self.ws_suffix
        os.makedirs(self.workspace)
        self.writer = SummaryWriter(self.workspace)

        # save ordered args, config, and a logfile to write stdout to
        if self.args is not None:
            f = opj(self.workspace, 'args.txt')
            with open(f, 'w') as file:
                for arg in sorted(vars(self.args)):
                    attr = getattr(self.args, arg)
                    file.write('{} = {}\n'.format(arg, attr))
                if hasattr(self.args, 'config'):
                    f = opj(self.workspace, 'config.txt')
                    with open(f, 'w') as file:
                        file.write(open(self.args.config, 'r').read())
        f = opj(self.workspace, 'log.txt')
        self.logfile = open(f, 'w')

        self.mylogger('[TRAINER_AIRSIM init] Making workspace {}'.format(self.workspace))
        self.mylogger('[TRAINER_AIRSIM init] Using traffic data: {}'.format(self.use_traffic))

        self.dataset_dir = opj(self.datadir, self.dataset_name)

        #################
        ## Dataloading ##
        #################
        if self.load_checkpoint:
            print('[TRAINER_AIRSIM init] Loading train_val_dirs from checkpoint')
            try:
                train_val_dirs = tuple(np.load(opj(os.path.dirname(self.checkpoint_path), 'train_val_dirs.npy'), allow_pickle=True))
            except:
                print('[TRAINER_AIRSIM init] Could not load train_val_dirs from checkpoint, dataloading from scratch')
                train_val_dirs = None
        else:    
            train_val_dirs = None

        self.dataloader(val_split=self.val_split, short=self.short, seed=self.seed, train_val_dirs=train_val_dirs)

        # TODO hardcoding num_training_steps to be the number of trajectories instead of number of images
        self.num_training_steps = self.train_traj_lengths.shape[0]
        self.num_val_steps = self.val_traj_lengths.shape[0]
        self.lr_warmup_iters = self.lr_warmup_epochs * self.num_training_steps

        ##################################
        ## Define network and optimizer ##
        ##################################
        self.mylogger('[SETUP] Establishing model and optimizer.')
        if self.model_type == 'LSTMNetVIT_GAT':
            # GAT model parameters
            embed_dim = getattr(self.args, 'embed_dim', 64)
            gat_hidden_dim = getattr(self.args, 'gat_hidden_dim', 128)
            gat_num_heads = getattr(self.args, 'gat_num_heads', 4)
            gat_num_layers = getattr(self.args, 'gat_num_layers', 2)
            
            self.model = model_library.LSTMNetVIT_GAT(
                robot_state_dim=4,
                traffic_state_dim=5,
                num_traffic_agents=5,
                embed_dim=embed_dim,
                gat_hidden_dim=gat_hidden_dim,
                gat_num_heads=gat_num_heads,
                gat_num_layers=gat_num_layers
            ).to(self.device).float()
            
            self.mylogger('[SETUP] Using GAT model with embed_dim: {}, gat_hidden_dim: {}, num_heads: {}, num_layers: {}'.format(
                embed_dim, gat_hidden_dim, gat_num_heads, gat_num_layers))
        elif self.model_type == 'LSTMNetVIT_NoTraffic_2D':
            embed_dim = getattr(self.args, 'embed_dim', 64)
            self.model = model_library.LSTMNetVIT_NoTraffic_2D(
                robot_state_dim=4, 
                embed_dim=embed_dim
            ).to(self.device).float()
            self.mylogger('[SETUP] Using no-traffic 2D model with embed_dim: {}'.format(embed_dim))
        else:
            self.mylogger('[SETUP] Invalid model_type {}. Available: LSTMNetVIT_GAT, LSTMNetVIT_NoTraffic_2D'.format(self.model_type))
            exit()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        self.num_eps_trained = 0
        if self.load_checkpoint:
            self.load_from_checkpoint(self.checkpoint_path)

        self.total_its = self.num_eps_trained * self.num_training_steps

    def mylogger(self, msg):
        print(msg)
        self.logfile.write(msg+'\n')

    def load_from_checkpoint(self, checkpoint_path):
        try:
            self.num_eps_trained = int(checkpoint_path[-10:-4])
        except:
            self.num_eps_trained = 0
            self.mylogger('[SETUP] Could not parse number of epochs trained from checkpoint path {}, using 0'.format(checkpoint_path))
        self.mylogger('[SETUP] Loading checkpoint from {}, already trained for {} epochs'.format(checkpoint_path, self.num_eps_trained))
        self.model.load_state_dict(torch.load(checkpoint_path, map_location=self.device))

    def dataloader(self, val_split, short=0, seed=None, train_val_dirs=None):
        self.mylogger('[DATALOADER] Loading from {}'.format(self.dataset_dir))
        train_data, val_data, is_png, (self.train_dirs, self.val_dirs), norm_params = dataloader_airsim(
            opj(self.basedir, self.dataset_dir), 
            val_split=val_split, 
            short=short, 
            seed=seed, 
            train_val_dirs=train_val_dirs,
            use_traffic=self.use_traffic
        )
        
        # Save normalization parameters
        if norm_params is not None:
            import pickle
            norm_params_path = opj(self.workspace, 'norm_params.pkl')
            with open(norm_params_path, 'wb') as f:
                pickle.dump(norm_params, f)
            self.mylogger('[DATALOADER] Normalization parameters saved to {}'.format(norm_params_path))
        
        # Unpack data (new format)
        self.train_robot_states, self.train_traffic_states, self.train_images, self.train_actions, self.train_traj_lengths = train_data
        self.val_robot_states, self.val_traffic_states, self.val_images, self.val_actions, self.val_traj_lengths = val_data
        
        self.mylogger('[DATALOADER] Dataloading done | train images {}, val images {}'.format(self.train_images.shape, self.val_images.shape))
        self.mylogger('[DATALOADER] Robot states | train {}, val {}'.format(self.train_robot_states.shape, self.val_robot_states.shape))
        if self.use_traffic:
            self.mylogger('[DATALOADER] Traffic states | train {}, val {}'.format(self.train_traffic_states.shape, self.val_traffic_states.shape))

        # Preload data to device
        data_to_preload = [self.train_robot_states, self.train_traffic_states, self.train_images, self.train_actions]
        self.train_robot_states, self.train_traffic_states, self.train_images, self.train_actions = preload(data_to_preload, self.device)
        
        data_to_preload = [self.val_robot_states, self.val_traffic_states, self.val_images, self.val_actions]
        self.val_robot_states, self.val_traffic_states, self.val_images, self.val_actions = preload(data_to_preload, self.device)
        
        self.mylogger('[DATALOADER] Preloading into device {} done'.format(self.device))

        assert self.train_images.max() <= 1.0 and self.train_images.min() >= 0.0, 'Images not normalized (values outside [0.0, 1.0])'

        # save train and val dirs in workspace for later use
        np.save(opj(self.workspace, 'train_val_dirs.npy'), np.array((self.train_dirs, self.val_dirs), dtype=object))

    def lr_scheduler(self, it):
        if it < self.lr_warmup_iters:
            lr = (0.9*self.lr)/self.lr_warmup_iters * it + 0.1*self.lr
        else:
            if self.lr_decay:
                lr = self.lr * (0.1 ** ((it-self.lr_warmup_iters) / (self.N_eps*self.num_training_steps)))
            else:
                lr = self.lr
        return lr

    def save_model(self, ep):
        self.mylogger('[SAVE] Saving model at epoch {}'.format(ep))
        path = self.workspace
        torch.save(self.model.state_dict(), opj(path, 'model_{}.pth'.format(str(ep).zfill(6))))
        self.mylogger('[SAVE] Model saved at {}'.format(path))

    def train(self):
        self.mylogger('[TRAIN] Training for {} epochs'.format(self.N_eps))
        train_start = time.time()

        # starting indices of trajectories in dataset
        self.train_traj_starts = np.cumsum(self.train_traj_lengths) - self.train_traj_lengths
        train_traj_lengths = self.train_traj_lengths

        for ep in range(self.num_eps_trained, self.num_eps_trained + self.N_eps):

            # periodically save model checkpoint
            if ep % self.save_model_freq == 0 and ep - self.num_eps_trained > 0:
                self.save_model(ep)

            # periodically evaluate on validation set
            if ep % self.val_freq == 0:
                self.validation(ep)

            ep_loss = 0
            gradnorm = 0

            # shuffling order of training data trajectories here
            shuffled_traj_indices = np.random.permutation(len(self.train_traj_starts))
            train_traj_starts = self.train_traj_starts[shuffled_traj_indices]
            train_traj_lengths = self.train_traj_lengths[shuffled_traj_indices]

            ### Training loop ###
            self.model.train()
            for it in range(self.num_training_steps):
                self.optimizer.zero_grad()
                
                # Extract trajectory data
                traj_start = train_traj_starts[it]
                traj_length = train_traj_lengths[it]
                traj_end = traj_start + traj_length
                
                # Get trajectory data
                traj_images = self.train_images[traj_start+1:traj_end].unsqueeze(1)  # (T-1, 1, H, W)
                traj_robot_states = self.train_robot_states[traj_start+1:traj_end]    # (T-1, 4)
                traj_actions = self.train_actions[traj_start+1:traj_end]             # (T-1, 2)
                
                # Prepare model inputs
                if self.model_type == 'LSTMNetVIT_GAT':
                    traj_traffic_states = self.train_traffic_states[traj_start+1:traj_end]  # (T-1, 5, 5)
                    model_inputs = [traj_images, traj_robot_states, traj_traffic_states]
                else:
                    model_inputs = [traj_images, traj_robot_states]
                
                # Forward pass
                pred, _ = self.model(model_inputs)
                
                # Actions are already normalized by desired_vel during data loading
                # Calculate loss (MSE between predicted and normalized target actions)
                # loss = F.mse_loss(pred, traj_actions)
                beta = 10
                # 1. 分别计算前进和转向分量的loss
                loss_forward = F.mse_loss(pred[:, 0], traj_actions[:, 0])
                loss_angular = F.mse_loss(pred[:, 1], traj_actions[:, 1])
                
                # 2. 组合成最终loss
                loss = loss_forward + beta * loss_angular
                
                ep_loss += loss
                loss.backward()
                gradnorm += torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=torch.inf)
                self.optimizer.step()
                
                new_lr = self.lr_scheduler(self.total_its-self.num_eps_trained*self.num_training_steps)
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                
                self.total_its += 1

            ep_loss /= self.num_training_steps
            gradnorm /= self.num_training_steps

            self.mylogger('[TRAIN] Completed epoch {}/{}, ep_loss = {:.6f}, time = {:.2f}s, time/epoch = {:.2f}s'.format(
                ep + 1, self.num_eps_trained + self.N_eps, ep_loss, 
                time.time() - train_start, (time.time() - train_start)/(ep + 1 - self.num_eps_trained)))

            self.writer.add_scalar('train/loss', ep_loss, ep)
            self.writer.add_scalar('train/gradnorm', gradnorm, ep)
            self.writer.add_scalar('train/lr', new_lr, self.total_its)
            self.writer.flush()

        self.mylogger('[TRAIN] Training complete, total time = {:.2f}s'.format(time.time() - train_start))
        self.save_model(ep)

    def validation(self, ep):
        self.mylogger('[VAL] Validating for val set of size {} images'.format(self.val_images.shape[0]))

        val_start = time.time()

        with torch.no_grad():
            ep_loss = 0

            # starting index of trajectories in dataset
            val_traj_starts = np.cumsum(self.val_traj_lengths) - self.val_traj_lengths

            ### Validation loop ###
            self.model.eval()

            for it in range(self.num_val_steps):
                # Extract trajectory data
                traj_start = val_traj_starts[it]
                traj_length = self.val_traj_lengths[it]
                traj_end = traj_start + traj_length
                
                # Get trajectory data
                traj_images = self.val_images[traj_start+1:traj_end].unsqueeze(1)  # (T-1, 1, H, W)
                traj_robot_states = self.val_robot_states[traj_start+1:traj_end]    # (T-1, 4)
                traj_actions = self.val_actions[traj_start+1:traj_end]             # (T-1, 2)
                
                # Prepare model inputs
                if self.model_type == 'LSTMNetVIT_GAT':
                    traj_traffic_states = self.val_traffic_states[traj_start+1:traj_end]  # (T-1, 5, 5)
                    model_inputs = [traj_images, traj_robot_states, traj_traffic_states]
                else:
                    model_inputs = [traj_images, traj_robot_states]
                
                # Forward pass
                pred, _ = self.model(model_inputs)
                
                # Actions are already normalized by desired_vel during data loading
                # Calculate loss
                loss = F.mse_loss(pred, traj_actions)
                ep_loss += loss

            ep_loss /= self.num_val_steps

            self.mylogger('[VAL] Completed validation, val_loss = {:.6f}, time taken = {:.2f} s'.format(ep_loss, time.time() - val_start))
            self.writer.add_scalar('val/loss', ep_loss, ep)

def argparsing():
    import configargparse
    parser = configargparse.ArgumentParser()

    # general params
    parser.add_argument('--config', is_config_file=True, help='config file relative path')
    parser.add_argument('--basedir', type=str, default='/home/{}/vitfly_ws/src/vitfly'.format(uname), help='path to repo')
    parser.add_argument('--logdir', type=str, default='training/logs_airsim', help='path to relative logging directory')
    parser.add_argument('--datadir', type=str, default='/home/{}/vitfly_ws/data'.format(uname), help='path to relative dataset directory')
    
    # experiment-level and learner params
    parser.add_argument('--ws_suffix', type=str, default='', help='suffix if any to workspace name')
    parser.add_argument('--model_type', type=str, default='LSTMNetVIT_GAT', help='model type: LSTMNetVIT_GAT or LSTMNetVIT_NoTraffic_2D')
    parser.add_argument('--use_traffic', action='store_true', default=False, help='whether to use traffic data')
    parser.add_argument('--dataset', type=str, default='Drone1', help='name of dataset folder')
    parser.add_argument('--short', type=int, default=0, help='if nonzero, how many trajectory folders to load')
    parser.add_argument('--val_split', type=float, default=0.2, help='fraction of dataset to use for validation')
    parser.add_argument('--seed', type=int, default=None, help='random seed to use for python random, numpy, and torch')
    parser.add_argument('--device', type=str, default='cuda', help='generic cuda device; specific GPU should be specified in CUDA_VISIBLE_DEVICES')
    parser.add_argument('--load_checkpoint', action='store_true', default=False, help='whether to load from a model checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default='', help='absolute path to model checkpoint')
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--N_eps', type=int, default=100, help='number of epochs to train for')
    parser.add_argument('--lr_warmup_epochs', type=int, default=5, help='number of epochs to warmup learning rate for')
    parser.add_argument('--lr_decay', action='store_true', default=False, help='whether to use lr_decay')
    parser.add_argument('--save_model_freq', type=int, default=25, help='frequency with which to save model checkpoints')
    parser.add_argument('--val_freq', type=int, default=10, help='frequency with which to evaluate on validation set')
    
    # GAT model parameters
    parser.add_argument('--embed_dim', type=int, default=64, help='embedding dimension for robot and traffic states')
    parser.add_argument('--gat_hidden_dim', type=int, default=128, help='hidden dimension for GAT layers')
    parser.add_argument('--gat_num_heads', type=int, default=4, help='number of attention heads in GAT')
    parser.add_argument('--gat_num_layers', type=int, default=2, help='number of GAT layers')

    args = parser.parse_args()
    
    # Automatically set model type and use_traffic based on model_type
    if args.model_type == 'LSTMNetVIT_GAT':
        args.use_traffic = True
    elif args.model_type == 'LSTMNetVIT_NoTraffic_2D':
        args.use_traffic = False
    
    print('[CONFIGARGPARSE] Using model type: {}'.format(args.model_type))
    print('[CONFIGARGPARSE] Using traffic data: {}'.format(args.use_traffic))
    if hasattr(args, 'config') and args.config:
        print('[CONFIGARGPARSE] Parsing args from config file {}'.format(args.config))

    return args

if __name__ == '__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    args = argparsing()
    print(args)

    learner = TRAINER_AIRSIM(args)
    learner.train() 