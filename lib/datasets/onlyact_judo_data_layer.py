import os
import os.path as osp
import numpy as np
import random

import torch
import torch.utils.data as data

class OnlyAct_JUDODataLayer(data.Dataset):
    def __init__(self, args, phase='train'):
        if 'onlyact' not in args.model_input or 'onlyact' not in args.model_target:
            raise Exception('Wrong --model_input or --model_target option')
        self.data_root = args.data_root
        self.model_input = args.model_input
        self.training = phase=='train'
        self.sessions = getattr(args, phase+'_session_set')

        self.downsampling = args.downsampling > 0 and self.training
        if self.downsampling:
            self.class_to_count = {idx_class: 0 for idx_class in range(args.num_classes)}

        self.inputs = []
        if self.downsampling:
            # in order to do not downsample always the same samples!
            random.shuffle(self.sessions)
        files = os.listdir(osp.join(args.data_root, args.model_target))
        for filename in files:
            if filename.split('___')[1][:-4] not in self.sessions:
                continue

            target = np.load(osp.join(self.data_root,
                                      args.model_target,
                                      filename))

            flag = True
            if self.downsampling:
                cls = target.argmax()
                self.class_to_count[cls] += 1
                if self.class_to_count[cls] > self.downsampling:
                    flag = False

            if flag:
                self.inputs.append([
                    filename, target
                ])

    def __getitem__(self, index):
        filename, target = self.inputs[index]

        feature_vector = np.load(osp.join(self.data_root,
                                           self.model_input,
                                           filename),
                                  mmap_mode='r').squeeze(axis=0)

        feature_vectors = torch.as_tensor(feature_vector.astype(np.float32))
        target = torch.as_tensor(target.astype(np.float32))

        return feature_vector, target

    def __len__(self):
        return len(self.inputs)