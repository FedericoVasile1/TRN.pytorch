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
        self.chunk_size = args.chunk_size
        self.sessions = getattr(args, phase+'_session_set')

        self.inputs = []
        for session in self.sessions:
            target = np.load(osp.join(self.data_root, 'target_frames_24fps', session+'.npy'))
            # round to multiple of CHUNK_SIZE
            num_frames = target.shape[0]
            num_frames = num_frames - (num_frames % args.chunk_size)
            target = target[:num_frames]
            # For each chunk, the central frame label is the label of the entire chunk
            target = target[args.chunk_size // 2::args.chunk_size]

            seed = np.random.randint(self.steps) if self.training else 0
            for start, end in zip(range(seed, target.shape[0], self.steps),
                                  range(seed + self.steps, target.shape[0], self.steps)):

                step_target = target[start:end]
                self.inputs.append([
                    session, start, end, step_target,
                ])

    def __getitem__(self, index):
        session, start, end, step_target = self.inputs[index]

        feature_vectors = np.load(osp.join(self.data_root, self.model_input, session+'.npy'), mmap_mode='r')
        feature_vectors = feature_vectors[start:end]
        feature_vectors = torch.as_tensor(feature_vectors.astype(np.float32))

        step_target = torch.as_tensor(step_target.astype(np.float32))

        return feature_vectors, step_target

    def __len__(self):
        return len(self.inputs)