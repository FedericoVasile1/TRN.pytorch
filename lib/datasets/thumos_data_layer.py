import os.path as osp

import torch
import torch.utils.data as data
from PIL import Image

import numpy as np

class TRNTHUMOSDataLayer(data.Dataset):
    def __init__(self, args, transform, phase='train'):
        self.data_root = args.data_root
        self.camera_frames_root = args.camera_frames_root   # base dir where the frames are
        self.sessions = getattr(args, phase+'_session_set')   # contains the names of all videos, both validation and test
        self.enc_steps = args.enc_steps     # in the paper is called 'le' (input sequence length)
        self.dec_steps = args.dec_steps     # in the paper is called 'ld' (timesteps in the future)
        self.training = phase=='train'
        self.data_aug = args.data_aug
        self.transform = transform

        # Only for debug purpose; train only on a minibatch of
        #  samples to check if everything works correctly
        if args.mini_batch != 0:
            self.sessions = self.sessions[0:args.mini_batch]

        self.inputs = []
        for session in self.sessions:
            target = np.load(osp.join(self.data_root, 'target_frames_24fps', session+'.npy'))   # shape:(num_frames, one_hot_vector)
            # data augmentation
            seed = np.random.randint(self.enc_steps) if self.training and self.data_aug else 0
            for start, end in zip(
                range(seed, target.shape[0] - self.dec_steps, self.enc_steps),
                range(seed + self.enc_steps, target.shape[0] - self.dec_steps, self.enc_steps)):
                enc_target = target[start:end]
                dec_target = self.get_dec_target(target[start:end + self.dec_steps])
                # session: name of the video
                # start: index of start frame for this "mini-sample" of the video
                # end: index of end frame for this "mini-sample" of the video
                # enc_target: target labels (each one as one_hot_vector) for the encoder.  shape:(start-end, len(one_hot_vect))
                # dec_target: target labels for the decoder
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

        # get the mini-sample of the video (i.e. start to end frames of the video)
        frames = None
        for idx_frame in range(start, end):
            # sum 1 to idx_frame because the index of frame images start from 1 while target array starts from 0
            frame = Image.open(osp.join(self.data_root, self.camera_frames_root, session, str(idx_frame+1)+'.jpg'))
            # TODO: do we need to apply some transformation to each frame?? (e.g. normalization)
            frame = self.transform(frame)
            if frames is None:
                # frames.shape:(num_frames, 3, H, W)
                frames = torch.zeros(end-start, frame.shape[0], frame.shape[1], frame.shape[2], dtype=torch.float32)
            frames[idx_frame-start] = frame.to(torch.float32)

        # also self.training? makes sense?
        if self.training and self.data_aug:
            pass

        frames = torch.FloatTensor(frames)
        enc_target = torch.as_tensor(enc_target.astype(np.float32))
        dec_target = torch.as_tensor(dec_target.astype(np.float32))

        return frames, enc_target, dec_target.view(-1, enc_target.shape[-1])

    def __len__(self):
        return len(self.inputs)
