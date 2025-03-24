import torch
from torch.utils.data import Dataset
import glob
from scipy.io import loadmat
import numpy as np
import math
import os



class AudioPoseDataset(Dataset):
    def __init__(self,root_dir, max_frames=100, device='cpu'):
        self.root_dir = root_dir
        file_list = glob.glob(self.root_dir+ '/**/ff.mat', recursive=True)
        self.data = []
        self.max_len = max_frames
        self.device = device
        self.UNIT_VALUE = 0.04  # 0.033333 #0.029931972789115413
        for file in file_list:
            aud_file = file.replace('ff.mat', 'audio.npy')
            if os.path.isfile(aud_file):
                self.data.append([file, aud_file])


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pose_path, aud_path = self.data[idx]

        dmm = loadmat(pose_path)
        dmm = dmm['coeff']
        wlm = np.load(aud_path)

        nb_gen = math.ceil(wlm.shape[0]/ self.max_len)
        wlm = np.reshape(wlm, (wlm.shape[0], wlm.shape[1] * wlm.shape[2]))
        pose = np.transpose(dmm)

        pose_tensor = torch.tensor(pose, dtype=torch.float32)
        wlm_tensor = torch.tensor(wlm, dtype=torch.float32)
        return pose_tensor, wlm_tensor, pose_path, nb_gen, wlm.shape[0]
