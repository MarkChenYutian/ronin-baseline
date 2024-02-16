from os import path as osp
import sys

import pandas

import torch
import torch
import pypose as pp
import numpy as np
import pykitti
from datetime import datetime
#import quaternion

from data_utils import CompiledSequence


class KITTISequence(CompiledSequence):
    """
    Dataset :- RIDI (can be downloaded from https://wustl.app.box.com/s/6lzfkaw00w76f8dmu0axax7441xcrzd9)
    Features :- raw angular rate and acceleration (includes gravity).
    """
    feature_dim = 6
    target_dim = 3
    aux_dim = 8

    def __init__(self, data_path, **kwargs):
        super().__init__(**kwargs)
        self.ts, self.features, self.targets, self.orientations, self.gt_pos = None, None, None, None, None
        self.w = kwargs.get('interval', 1)
        self.info = {}

        if data_path is not None:
            self.load(data_path)

    def load(self, path):
        if path[-1] == '/':
            path = path[:-1]

        self.info['path'] = osp.split(path)[-1]
        # self.info['ori_source'] = 'game_rv'
        self.info['ori_source'] = 'gyro_integration'

        
        self.load_imu(path)
        self.load_gt(path)
        # self.interpotation()  # No need for this since KITTI is aligned

        ##elseif
        # gyro_glob = self.info["gyro"]
        # acce_glob = self.info["acc"]

        ts =self.info["time"]


        ##if global
        gyro_glob = self.info['gyro']
        acce_glob = self.info['acc']
        
        ori = pp.SO3(self.info['gt_orientation']).double()
        # nz = np.zeros(ts.shape).reshape((-1,1))[1:]
        gyro_glob = (ori @ torch.tensor(self.info['gyro'])).numpy()
        acce_glob = (ori @ torch.tensor(self.info['acc'])).numpy()


        gt_pos =  self.info["gt_translation"]
        #self.ts = ts
        self.ts =ts.reshape((-1,1))[1:]
        t_reshape = ts.reshape((-1,1))
        self.features = np.concatenate([gyro_glob, acce_glob], axis=1)
        t = (t_reshape[self.w:] - t_reshape[:-self.w])[1:]
        self.targets = ((gt_pos[self.w:, :3] - gt_pos[:-self.w, :3]) / t)
        self.gt_pos = gt_pos
        self.orientations = self.info["gt_orientation"]
        print(self.ts.shape, self.features.shape, self.targets.shape, self.gt_pos.shape, self.orientations.shape,
              self.w)
        
    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_aux(self):
        return np.concatenate([self.ts, self.orientations, self.gt_pos], axis=1)

    def get_meta(self):
        return '{}: orientation {}'.format(self.info['path'], self.info['ori_source'])

    def load_imu(self, folder):
        tokens = folder.split("/")
        root_dir = "/".join(tokens[:-2])
        data_date = tokens[-2]
        drive = tokens[-1]
        
        raw_data = pykitti.raw(root_dir, data_date, drive)
        raw_len  = len(raw_data.timestamps) - 1
        self.info["time"] = np.array(
            [datetime.timestamp(raw_data.timestamps[i]) for i in range(raw_len + 1)],
            dtype=np.double
        )
        self.info["gyro"] = np.array(
            [   
                [raw_data.oxts[i].packet.wx,
                 raw_data.oxts[i].packet.wy,
                 raw_data.oxts[i].packet.wz]
                for i in range(raw_len)
            ],
            dtype=np.double
        )
        self.info["acc"]  = np.array(
            [
                [raw_data.oxts[i].packet.ax,
                raw_data.oxts[i].packet.ay,
                raw_data.oxts[i].packet.az]
                for i in range(raw_len)
            ],
            dtype=np.double
        )
    
    def load_gt(self, folder):
        tokens = folder.split("/")
        root_dir = "/".join(tokens[:-2])
        data_date = tokens[-2]
        drive = tokens[-1]
        
        raw_data = pykitti.raw(root_dir, data_date, drive)
        raw_len  = len(raw_data.timestamps) - 1
        
        # gt_data = np.loadtxt(osp.join(folder, "mav0/state_groundtruth_estimate0/data.csv"), dtype=float, delimiter=',')
        self.info["gt_time"] = self.info["time"]
        self.info["pos"] = np.array(
            [raw_data.oxts[i].T_w_imu[0:3, 3]
             for i in range(raw_len)],
            dtype=np.double
        )
        self.info['quat'] = pp.euler2SO3(torch.tensor(
            [
                [raw_data.oxts[i].packet.roll,
                raw_data.oxts[i].packet.pitch,
                raw_data.oxts[i].packet.yaw]
                for i in range(raw_len)
            ]
        )).double()
        # self.info["b_acc"] = gt_data[:,-3:]
        # self.info["b_gyro"] = gt_data[:,-6:-3]
        raw_velocity = torch.tensor(
            [[raw_data.oxts[i].packet.vf,
              raw_data.oxts[i].packet.vl,
              raw_data.oxts[i].packet.vu]
            for i in range(raw_len)],
            dtype=torch.double
        )
        print(self.info["quat"].shape)
        print(raw_velocity.shape)
        print(self.info["time"].shape)
        self.info["velocity"] = (self.info["quat"] @ raw_velocity).numpy()
        self.info["quat"] = self.info["quat"].numpy()
        self.info["gt_orientation"] = self.info["quat"]
        self.info["gt_translation"] = self.info["pos"]
        
    def interpotation(self):
        t_start = np.max([self.info['gt_time'][0], self.info['time'][0]])
        t_end = np.min([self.info['gt_time'][-1], self.info['time'][-1]])

        idx_start_imu = np.searchsorted(self.info['time'], t_start)
        idx_start_gt = np.searchsorted(self.info['gt_time'], t_start)

        idx_end_imu = np.searchsorted(self.info['time'], t_end, 'right')
        idx_end_gt = np.searchsorted(self.info['gt_time'], t_end, 'right')

        ## GT data
        for k in ['gt_time', 'pos', 'quat', 'velocity']:
            print(k, self.info[k].shape)
            self.info[k] = self.info[k][idx_start_gt:idx_end_gt]

        # ## imu data
        for k in ['time', 'acc', 'gyro']:
            print(k, self.info[k].shape)
            self.info[k] = self.info[k][idx_start_imu:idx_end_imu]

        ## start interpotation
        self.info["gt_orientation"] = self.interp_rot(self.info['time'], self.info['gt_time'], self.info['quat'])
        self.info["gt_translation"] = self.interp_xyz(self.info['time'], self.info['gt_time'], self.info['pos'])

    def slerp(self, q0, q1, tau, DOT_THRESHOLD = 0.9995):
        """Spherical linear interpolation."""

        dot = (q0*q1).sum(dim=1)
        q1[dot < 0] = -q1[dot < 0]
        dot[dot < 0] = -dot[dot < 0]

        q = torch.zeros_like(q0)
        tmp = q0 + tau.unsqueeze(1) * (q1 - q0)
        tmp = tmp[dot > DOT_THRESHOLD]
        q[dot > DOT_THRESHOLD] = tmp / tmp.norm(dim=1, keepdim=True)

        theta_0 = dot.acos()
        sin_theta_0 = theta_0.sin()
        theta = theta_0 * tau
        sin_theta = theta.sin()
        s0 = (theta.cos() - dot * sin_theta / sin_theta_0).unsqueeze(1)
        s1 = (sin_theta / sin_theta_0).unsqueeze(1)
        q[dot < DOT_THRESHOLD] = ((s0 * q0) + (s1 * q1))[dot < DOT_THRESHOLD]
        return q / q.norm(dim=1, keepdim=True)

    def qinterp(self, qs, t, t_int):
        idxs = np.searchsorted(t, t_int)
        idxs0 = idxs-1
        idxs0[idxs0 < 0] = 0
        idxs1 = idxs
        idxs1[idxs1 == t.shape[0]] = t.shape[0] - 1
        q0 = qs[idxs0]
        q1 = qs[idxs1]
        tau = torch.zeros_like(t_int)
        dt = (t[idxs1]-t[idxs0])[idxs0 != idxs1]
        tau[idxs0 != idxs1] = (t_int-t[idxs0])[idxs0 != idxs1]/dt
        return self.slerp(q0, q1, tau)

    def interp_rot(self, time, opt_time, quat):
        # interpolation in the log space
        imu_dt = torch.Tensor(time - opt_time[0])
        gt_dt = torch.Tensor(opt_time - opt_time[0])

        # quat = qnorm(torch.tensor(quat))
        quat = torch.tensor(quat)
        quat = self.qinterp(quat, gt_dt, imu_dt).double()
        self.info['rot_wxyz'] = quat
        rot = torch.zeros_like(quat)
        rot[:,3] = quat[:,0]
        rot[:,:3] = quat[:,1:]
        
        return rot
        #return pp.SO3(rot)

    def interp_xyz(self, time, opt_time, xyz):
        
        intep_x = np.interp(time, xp=opt_time, fp = xyz[:,0])
        intep_y = np.interp(time, xp=opt_time, fp = xyz[:,1])
        intep_z = np.interp(time, xp=opt_time, fp = xyz[:,2])

        inte_xyz = np.stack([intep_x, intep_y, intep_z]).transpose()
        return inte_xyz
        #return torch.tensor(inte_xyz)

