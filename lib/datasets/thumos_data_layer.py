import os.path as osp
import numpy as np

import torch
import torch.utils.data as data

class TRNTHUMOSDataLayer(data.Dataset):
    def __init__(self, args, phase='train'):
        self.data_root = args.data_root
        self.camera_feature = args.camera_feature
        self.motion_feature = args.motion_feature       # optical flow will not be used in our case
        self.sessions = getattr(args, phase+'_session_set')
        self.enc_steps = args.enc_steps
        self.dec_steps = args.dec_steps
        self.training = phase=='train'
        self.args_inputs = args.inputs

        self.inputs = []
        for session in self.sessions:
            target = np.load(osp.join(self.data_root, 'target_startend' if args.num_classes == 44 else 'target', session+'.npy'))
            seed = np.random.randint(self.enc_steps) if self.training else 0
            for start, end in zip(range(seed, target.shape[0] - self.dec_steps, self.enc_steps),
                                  range(seed + self.enc_steps, target.shape[0] - self.dec_steps, self.enc_steps)):
                if args.downsample_backgr and self.training:
                    background_vect = np.zeros_like(target[start:end])
                    background_vect[:, 0] = 1
                    if (target[start:end] == background_vect).all():
                        continue

                enc_target = target[start:end]
                dec_target = self.get_dec_target(target[start:end + self.dec_steps])
                self.inputs.append([
                    session, start, end, enc_target, dec_target,
                ])

    def get_dec_target(self, target_vector):
        target_matrix = np.zeros((self.enc_steps, self.dec_steps, target_vector.shape[-1]))
        for i in range(self.enc_steps):
            for j in range(self.dec_steps):
                # 0 -> [1, 2, 3]
                # target_matrix[i,j] = target_vector[i+j+1,:]
                # 0 -> [0, 1, 2]
                target_matrix[i,j] = target_vector[i+j,:]
        return target_matrix

    def __getitem__(self, index):
        session, start, end, enc_target, dec_target = self.inputs[index]

        camera_inputs = np.load(osp.join(self.data_root, self.camera_feature, session+'.npy'), mmap_mode='r')
        camera_inputs = camera_inputs[start:end]
        camera_inputs = torch.as_tensor(camera_inputs.astype(np.float32))

        motion_inputs = np.load(osp.join(self.data_root, self.motion_feature, session+'.npy'), mmap_mode='r')
        motion_inputs = motion_inputs[start:end]
        motion_inputs = torch.as_tensor(motion_inputs.astype(np.float32))

        enc_target = torch.as_tensor(enc_target.astype(np.float32))
        dec_target = torch.as_tensor(dec_target.astype(np.float32))
        dec_target = dec_target.view(-1, enc_target.shape[-1])

        return camera_inputs, motion_inputs, enc_target, dec_target

    def __len__(self):
        return len(self.inputs)