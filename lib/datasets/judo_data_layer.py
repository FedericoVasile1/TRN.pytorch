import os.path as osp
import numpy as np
import random

import torch
import torch.utils.data as data

def get_dec_target(self, target_vector):
    target_matrix = np.zeros((self.enc_steps, self.dec_steps, target_vector.shape[-1]))
    for i in range(self.enc_steps):
        for j in range(self.dec_steps):
            # 0 -> [1, 2, 3]
            # target_matrix[i,j] = target_vector[i+j+1,:]
            # 0 -> [0, 1, 2]
            target_matrix[i, j] = target_vector[i + j, :]
    return target_matrix

class JUDODataLayer(data.Dataset):
    def __init__(self, args, phase='train'):
        self.data_root = args.data_root
        self.model_input = args.model_input
        self.steps = args.steps
        self.training = phase=='train'
        self.sessions = getattr(args, phase+'_session_set')
        self.chunk_size = args.chunk_size

        self.downsampling = args.downsampling > 0 and self.training
        if self.downsampling:
            # WE ARE TAKING INTO ACCOUNT ONLY ACTION CLASSES, I.E. BACKGROUND CLASS IS NOT DOWNSAMPLED!
            # MODIFY ALSO LINE 101 IF YOU WANT TO INCLUDE ALSO BACKGROUND CLASS
            self.class_to_count = {idx_class+1: 0 for idx_class in range(args.num_classes-1)}

        self.inputs = []
        if self.downsampling:
            # in order to do not downsample always the same samples!
            random.shuffle(self.sessions)
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
            for start, end in zip(range(seed, target.shape[0], self.steps),
                                  range(seed + self.steps, target.shape[0] + 1, self.steps)):

                step_target = target[start:end]

                flag = True
                if self.downsampling:
                    unique, counts = np.unique(step_target.argmax(axis=1), return_counts=True)

                    # drop if action samples are greater than threshold
                    for action_idx, num_samples in self.class_to_count.items():
                        if num_samples < args.downsampling:
                            continue
                        if action_idx in step_target.argmax(axis=1):
                            flag = False
                    # count actions labels
                    for i, action_idx in enumerate(unique):
                        if action_idx == 0:
                            # ignore background class
                            continue
                        self.class_to_count[action_idx] += counts[i]

                if flag:
                    if args.model == 'TRN':
                        dec_step_target = get_dec_target(target[start:end + args.dec_steps])
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
        step_target = torch.as_tensor(step_target.astype(np.float32))

        return feature_vectors, step_target[0] if step_target[1] is None else step_target

    def __len__(self):
        return len(self.inputs)