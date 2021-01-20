import os
import os.path as osp
import numpy as np
import random

import torch
import torch.utils.data as data


class Candidates_JUDODataLayer_v2(data.Dataset):
    def __init__(self, args, phase='train'):
        self.data_root = args.data_root
        self.model_input = args.model_input
        self.training = phase=='train'
        self.sessions = getattr(args, phase+'_session_set')

        self.downsampling = args.downsampling > 0 and self.training
        if self.downsampling:
            # WE ARE TAKING INTO ACCOUNT ALSO BACKGROUND CLASS!!
            self.class_to_count = {idx_class: 0 for idx_class in range(args.num_classes)}

        self.inputs = []
        files = os.listdir(osp.join(args.data_root, args.model_target))
        if self.downsampling > 0:
            # in order to do not downsample always the same samples
            random.shuffle(files)
        for filename in files:
            if filename.split('___')[1][:-4] not in self.sessions:
                continue

            target = np.load(osp.join(self.data_root, args.model_target, filename))
            num_frames = target.shape[0]
            num_frames = num_frames - (num_frames % args.chunk_size)
            target = target[:num_frames]
            target = target[args.chunk_size // 2::args.chunk_size]

            for idx_chunk in range(target.shape[0]):
                flag = True
                if self.downsampling:
                    cls_idx = target[idx_chunk].argmax().item()
                    self.class_to_count[cls_idx] += 1
                    if self.class_to_count[cls_idx] > args.downsampling:
                        flag = False

                if flag:
                    self.inputs.append([
                        filename, target[idx_chunk], idx_chunk
                    ])

    def __getitem__(self, index):
        filename, target, idx_chunk = self.inputs[index]

        feature_vectors = np.load(osp.join(self.data_root,
                                           self.model_input,
                                           filename),
                                  mmap_mode='r')
        feature_vector = feature_vectors[idx_chunk]

        feature_vector = torch.as_tensor(feature_vector.astype(np.float32))
        target = torch.as_tensor(target.astype(np.float32))

        return feature_vector, target

    def __len__(self):
        return len(self.inputs)