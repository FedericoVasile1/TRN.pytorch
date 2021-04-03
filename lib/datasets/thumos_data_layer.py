import os.path as osp
import numpy as np

import torch
import torch.utils.data as data

class THUMOSDataLayer(data.Dataset):
    def __init__(self, args, phase='train'):
        self.data_root = args.data_root
        self.model_input = args.model_input
        self.steps = args.steps
        self.training = phase=='train'
        self.sessions = getattr(args, phase+'_session_set')
        self.chunk_size = args.chunk_size

        if args.model == 'TRN':
            self.steps = args.enc_steps
            self.enc_steps = args.enc_steps
            self.dec_steps = args.dec_steps
        else:
            self.enc_steps = 0
            self.dec_steps = 0

        self.inputs = []
        for session in self.sessions:
            target = np.load(osp.join(self.data_root,
                                      args.model_target,
                                      session+'.npy'))
            # round to multiple of chunk_size
            num_frames = target.shape[0]
            num_frames = num_frames - (num_frames % args.chunk_size)
            target = target[:num_frames]
            # For each chunk, the central frame label is the label of the entire chunk
            target = target[args.chunk_size // 2::args.chunk_size]

            seed = np.random.randint(self.steps) if self.training else 0
            for start, end in zip(range(seed, target.shape[0] - self.dec_steps, self.steps),
                                  range(seed + self.steps, target.shape[0] - self.dec_steps, self.steps)):

                step_target = target[start:end]

                if args.model == 'TRN':
                    dec_step_target = self.get_dec_target(target[start:end + args.dec_steps])
                else:
                    dec_step_target = None
                self.inputs.append([
                    session, (step_target, dec_step_target), start, end
                ])

    def __getitem__(self, index):
        session, step_target, start, end = self.inputs[index]

        feature_vectors = np.load(osp.join(self.data_root,
                                           self.model_input,
                                           session+'.npy'),
                                  mmap_mode='r')

        feature_vectors = feature_vectors[start:end]

        feature_vectors = torch.as_tensor(feature_vectors.astype(np.float32))
        if step_target[1] is None:
            step_target = torch.as_tensor(step_target[0].astype(np.float32))
        else:
            step_target = (torch.as_tensor(step_target[0].astype(np.float32)),
                           torch.as_tensor(step_target[1].astype(np.float32)))

        return feature_vectors, step_target

    def __len__(self):
        return len(self.inputs)

    def get_dec_target(self, target_vector):
        target_matrix = np.zeros((self.enc_steps, self.dec_steps, target_vector.shape[-1]))
        for i in range(self.enc_steps):
            for j in range(self.dec_steps):
                # 0 -> [1, 2, 3]
                # target_matrix[i,j] = target_vector[i+j+1,:]
                # 0 -> [0, 1, 2]
                target_matrix[i, j] = target_vector[i + j, :]
        return target_matrix