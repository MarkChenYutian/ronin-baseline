import torch
import numpy as np
import pypose as pp
import torch.utils.data as Data
from scipy.spatial.transform import Rotation as T

import matplotlib.pyplot as plt
import os, pickle
from os import path as osp

#from utils import qinterp, qnorm, lookAt, slerp
# from seqdatasets import SeqDataset, SeqInfDataset
from data_utils import CompiledSequence

class NavGSequence(CompiledSequence):
    """
    Output:
    acce: the accelaration in **world frame**
    """
    feature_dim = 6
    target_dim = 3
    aux_dim = 8
    
    def __init__(self, data_path, **kwargs):
        super().__init__(**kwargs)
        self.ts, self.features, self.targets, self.orientations, self.gt_pos = None, None, None, None, None
        self.w = kwargs.get('interval', 1)
        self.info = {}
        
        self.camera_ext_R = pp.identity_SO3()
        self.camera_ext_t = torch.tensor(np.array([0., 0., 0.,]))
        self.vicon_ext_R =  pp.identity_SO3()
        self.vicon_ext_t =  torch.tensor(np.array([0., 0., 0.]))
        self.ext_T = pp.SE3(torch.cat((self.camera_ext_t, self.camera_ext_R)))

        
        if data_path is not None:
            self.load(data_path)
    
    def load(self, path):
        if path[-1] == '/':
            path = path[:-1]

        self.info['path'] = osp.split(path)[-1]
        self.info['ori_source'] = 'game_rv'

        
        self.load_imu(path)
        self.load_gt(path)
        print("data_path:",path)
    
    
        # get the index for the data
        t_start = np.max([self.info['gt_time'][0], self.info['time'][0]])
        t_end = np.min([self.info['gt_time'][-1], self.info['time'][-1]])

        idx_start_imu = np.searchsorted(self.info['time'], t_start)
        idx_start_gt = np.searchsorted(self.info['gt_time'], t_start)

        idx_end_imu = np.searchsorted(self.info['time'], t_end, 'right')
        idx_end_gt = np.searchsorted(self.info['gt_time'], t_end, 'right')

        for k in ['gt_time', 'gt_orientation', 'gt_translation', 'angular_velocity', 'velocity']:
            self.info[k] = self.info[k][idx_start_gt:idx_end_gt]

        # ## imu data
        for k in ['time', 'acc', 'gyro']:
            self.info[k] = self.info[k][idx_start_imu:idx_end_imu]

        # move the time to torch
        # self.info["time"] = torch.tensor(self.info["time"])
        # self.info["gt_time"] = torch.tensor(self.info["gt_time"])
        self.info['dt'] = (self.info["time"][1:] - self.info["time"][:-1])[:,None]
        # self.info["mask"] = torch.ones(self.info["time"].shape[0], dtype=torch.bool)

        # self.info["gyro"] = torch.tensor(self.info["gyro"])
        # self.info["acc"] = torch.tensor(self.info["acc"])
        #self.info["velocity"] = torch.tensor(self.info["velocity"])


        ######
        ts =self.info["time"]


        ##if global
        # gyro_glob = self.info['gyro']
        # acce_glob = self.info['acc']
        print(pp.SO3(torch.tensor(self.info["gt_orientation"])).shape)
        print(pp.so3(self.info['gyro']).Exp().shape)
        gyro_g = pp.SO3(torch.tensor(self.info["gt_orientation"])) * torch.tensor(self.info['gyro'])
        acce_g = pp.SO3(torch.tensor(self.info["gt_orientation"])) * torch.tensor(self.info['acc'])
        
        
        
        
        gyro_glob = np.array(gyro_g)
        acce_glob = np.array(acce_g)
        
        gt_pos =  self.info["gt_translation"]
        #self.ts = ts
        self.ts =ts.reshape((-1,1))
        t_reshape = ts.reshape((-1,1))
        self.features = np.concatenate([gyro_glob, acce_glob], axis=1)
        t = (t_reshape[self.w:] - t_reshape[:-self.w])
        
        self.targets = ((gt_pos[self.w:, :3] - gt_pos[:-self.w, :3]) / t)
        #self.targets = self.info["velocity"]
        self.gt_pos = gt_pos
        self.orientations = self.info["gt_orientation"]
        print(self.ts.shape, self.features.shape, self.targets.shape, self.gt_pos.shape, self.orientations.shape,
              self.w)
        
        
        
        

    def load_imu(self, file_path):
        with open(os.path.join(file_path, "imu.pkl"), "rb") as handle:
            imu_data = pickle.load(handle)
        
        self.info["time"] = imu_data[:,0] 
        self.info["acc"] = imu_data[:,1:4]# acc a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]
        self.info["gyro"] = imu_data[:,4:] # w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1]

    def load_gt(self, file_path):
        with open(os.path.join(file_path, "pose.pkl"), "rb") as handle:
            pose = pickle.load(handle)

        with open(os.path.join(file_path, "twist.pkl"), "rb") as handle:
            twist = pickle.load(handle)

        self.info["gt_time"] = pose[:,0] 
        self.info["gt_orientation"] = pp.SO3(pose[:,4:]).double()# xyzw
        #self.info['gt_translation'] = torch.tensor(pose[:,1:4])
        self.info['gt_translation'] = pose[:,1:4]
        self.info["velocity"] = np.array(self.info["gt_orientation"] @ torch.tensor(twist[:,1:4]))
        self.info["angular_velocity"] = self.info["gt_orientation"] @ torch.tensor(twist[:,4:])
        
        
    def get_length(self):
        return self.info['time'].shape[0]


#####################
    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_aux(self):
        return np.concatenate([self.ts, self.orientations, self.gt_pos], axis=1)

    def get_meta(self):
        return '{}: orientation {}'.format(self.info['path'], self.info['ori_source'])
