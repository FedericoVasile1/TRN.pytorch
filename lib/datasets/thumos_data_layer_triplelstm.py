import os.path as osp
import numpy as np

import torch
import torch.utils.data as data

class TRNTHUMOSDataLayerTripleLSTM(data.Dataset):
    def __init__(self, args, phase='train'):
        self.data_root = args.data_root
        self.camera_feature = args.camera_feature
        self.sessions = getattr(args, phase+'_session_set')
        self.enc_steps = args.enc_steps
        self.dec_steps = 1
        self.training = phase=='train'

        self.inputs = []
        if not self.training:
            # validate only on a subset
            self.sessions = self.sessions[0:50]
            pass
        for session in self.sessions:
            target_acts = np.load(osp.join(self.data_root, 'target', session+'.npy'))
            target_startend = np.load(osp.join(self.data_root, 'target_startend', session + '.npy'))
            seed = np.random.randint(self.enc_steps) if self.training else 0
            for start, end in zip(
                range(seed, target_acts.shape[0] - self.dec_steps, self.enc_steps),
                range(seed + self.enc_steps, target_acts.shape[0] - self.dec_steps, self.enc_steps)):
                if args.downsample_backgr and self.training:
                    background_vect = np.zeros_like(target_acts[start:end])
                    background_vect[:, 0] = 1
                    if (target_acts[start:end] == background_vect).all():
                        continue

                enc_target_acts = target_acts[start:end]
                enc_target_startend = target_startend[start:end]
                self.inputs.append([
                    session, start, end, enc_target_acts, enc_target_startend,
                ])

    def __getitem__(self, index):
        session, start, end, enc_target_acts, enc_target_startend = self.inputs[index]

        feature_vectors = np.load(
            osp.join(self.data_root, self.camera_feature, session+'.npy'), mmap_mode='r')
        camera_inputs = feature_vectors[start:end]
        camera_inputs = torch.as_tensor(camera_inputs.astype(np.float32))
        motion_inputs = np.zeros((self.enc_steps, 1))     # zeros because optical flow will not be used
        enc_target_acts = torch.as_tensor(enc_target_acts.astype(np.float32))
        enc_target_startend = torch.as_tensor(enc_target_startend.astype(np.float32))

        return camera_inputs, motion_inputs, enc_target_acts, enc_target_startend

    def __len__(self):
        return len(self.inputs)