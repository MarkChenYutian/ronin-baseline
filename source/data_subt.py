import os
import numpy as np
import torch
import numpy as np
import pypose as pp
import os
# from utils import qinterp, qnorm, lookAt, slerp
import pickle
from data_utils import CompiledSequence

class SubTSequence(CompiledSequence):
   
    """
    Output:
    acce: the accelaration in **imu frame**
    gyro: the gyro in **imu frame**
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
        self.load_imu(path)
        self.load_gt(path)
        self.load_bias(path)
        print("data_path:",path)
    
    
        
        # get the index for the data
        t_start = np.max([self.info['gt_time'][0], self.info['time'][0]])
        t_end = np.min([self.info['gt_time'][-1], self.info['time'][-1]])

        idx_start_imu = np.searchsorted(self.info['time'], t_start)
        idx_start_gt = np.searchsorted(self.info['gt_time'], t_start)

        idx_end_imu = np.searchsorted(self.info['time'], t_end, 'right')
        idx_end_gt = np.searchsorted(self.info['gt_time'], t_end, 'right')
    
        for k in ['gt_time', 'pos', 'quat','gyro_bias','acc_bias','vel']:
            self.info[k] = self.info[k][idx_start_gt:idx_end_gt]

        # ## imu data
        for k in ['time', 'acc', 'gyro','rot_imu']:
            self.info[k] = self.info[k][idx_start_imu:idx_end_imu]
        
        #inteporlate the ground truth pose
        self.info['gt_translation'] = self.interp_xyz(self.info['time'], self.info['gt_time'], self.info['pos'])
        self.info['g_b'] = self.interp_rot(self.info['time'], self.info['gt_time'], pp.so3(self.info['gyro_bias']).Exp()).Log()
        self.info['a_b'] = self.interp_xyz(self.info['time'], self.info['gt_time'], self.info['acc_bias'])
        self.info['velocity'] = self.interp_xyz(self.info['time'], self.info['gt_time'], self.info['vel'])
        self.info['gt_orientation'] = self.interp_rot(self.info['time'], self.info['gt_time'], self.info['quat'])
        
        # move to torch
        self.info["time"] = torch.tensor(self.info["time"]).double()
        self.info["gt_time"] = torch.tensor(self.info["gt_time"]).double()
        self.info['dt'] = (self.info["time"][1:] - self.info["time"][:-1])[:,None].double()

        gyro_g = pp.SO3(torch.tensor(self.info["gt_orientation"])) * torch.tensor(self.info['gyro'])
        acce_g = pp.SO3(torch.tensor(self.info["gt_orientation"])) * torch.tensor(self.info['acc'])
             
        gyro = np.array(gyro_g)
        acc = np.array(acce_g)
        ts =self.info["time"]
        gt_pos =  np.array(self.info["gt_translation"])
        self.ts =ts.reshape((-1,1))
        t_reshape = ts.reshape((-1,1))
        self.features = np.concatenate([gyro, acc], axis=1)
        t = (t_reshape[self.w:] - t_reshape[:-self.w])
        
        self.targets = np.array(((gt_pos[self.w:, :3] - gt_pos[:-self.w, :3]) / t))
        self.gt_pos = gt_pos
        self.orientations = self.info["gt_orientation"]
        print(type(self.features))
        print(type(self.targets))
        print(self.ts.shape, self.features.shape, self.targets.shape, self.gt_pos.shape, self.orientations.shape,
              self.w)
       
   
    def load_imu(self, folder):
        imu_data = np.loadtxt(os.path.join(folder, "imu_data/imu_data.csv"),dtype = float, delimiter = ',', skiprows=1)
    
        self.info["time"] = imu_data[:,0]/1e9
        self.info['quat_imu'] = imu_data[:,1:5] #xyzw
        self.info['rot_imu'] = pp.SO3(self.info['quat_imu'])
        self.info["gyro"] = torch.tensor(imu_data[:,5:8]) # w_RS_S_x [rad s^-1],w_RS_S_y [rad s^-1],w_RS_S_z [rad s^-1]
        self.info["acc"] = torch.tensor(imu_data[:,8:11])# acc a_RS_S_x [m s^-2],a_RS_S_y [m s^-2],a_RS_S_z [m s^-2]

    def load_gt(self,folder):
        gt_data = np.loadtxt(os.path.join(folder, "ground_truth/ground_truth_imu.csv"), dtype=float, delimiter=',',skiprows=1)
        self.info["gt_time"] = gt_data[:,0] / 1e9
        self.info['pos'] = gt_data[:,1:4]
        self.info['quat'] = gt_data[:,4:8] # xyzw
        self.info['vel'] = gt_data[:,8:11]
        self.info['transform'] = gt_data[:,1:8]
    
    def load_bias(self,folder):
        gt_data = np.loadtxt(os.path.join(folder, "ground_truth/ground_truth_imu.csv"), dtype=float, delimiter=',',skiprows=1)
        self.info["gyro_bias"] =torch.tensor(gt_data[:,11:14])
        self.info["acc_bias"] = torch.tensor(gt_data[:,14:17])
        
    def interp_xyz(self,time, opt_time, xyz):
        intep_x = np.interp(time, xp=opt_time, fp = xyz[:,0])
        intep_y = np.interp(time, xp=opt_time, fp = xyz[:,1])
        intep_z = np.interp(time, xp=opt_time, fp = xyz[:,2])
        inte_xyz = np.stack([intep_x, intep_y, intep_z]).transpose()
        return torch.tensor(inte_xyz)

    def interp_rot(self,time, opt_time, quat):
        quat_wxyz = np.zeros_like(quat)
        quat_wxyz[:,0] = quat[:,3]
        quat_wxyz[:,1:] = quat[:,:3]
        quat_wxyz = torch.tensor(quat_wxyz)
        imu_dt = torch.Tensor(time - opt_time[0])
        gt_dt = torch.Tensor(opt_time - opt_time[0])
        quat = self.qinterp(quat_wxyz, gt_dt, imu_dt).double()
        quat_xyzw = torch.zeros_like(quat)
        quat_xyzw[:,3] = quat[:,0]
        quat_xyzw[:,:3] = quat[:,1:]
        return pp.SO3(quat_xyzw)
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
    
    def get_length(self):
        return self.info['time'].shape[0]
    
    def get_feature(self):
        return self.features

    def get_target(self):
        return self.targets

    def get_aux(self):
        return np.concatenate([self.ts, self.orientations, self.gt_pos], axis=1)

    # def get_meta(self):
    #     return '{}: orientation {}'.format(self.info['path'], self.info['ori_source'])

