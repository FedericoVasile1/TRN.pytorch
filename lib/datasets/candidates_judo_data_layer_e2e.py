import os
import os.path as osp
import numpy as np
import random

import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image

from lib.models.i3d.i3d import I3DNormalization

class Candidates_JUDODataLayerE2E(data.Dataset):
    def __init__(self, args, phase='train'):
        self.data_root = args.data_root
        self.model_input = args.model_input
        self.training = phase=='train'
        self.sessions = getattr(args, phase+'_session_set')
        self.chunk_size = args.chunk_size
        self.is_3D = args.is_3D

        self.downsampling = args.downsampling > 0 and self.training
        if self.downsampling:
            # WE ARE TAKING INTO ACCOUNT BOTH BACKGROUND AND ACTION CLASSES
            self.class_to_count = {idx_class: 0 for idx_class in range(args.num_classes)}

        if args.is_3D:
            if args.feature_extractor == 'I3D':
                self.transform = transforms.Compose([
                    transforms.Resize((224, 320)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    I3DNormalization(),
                ])
            else:
                raise Exception('Wrong --feature_extractor option. ' + args.feature_extractor + ' unknown')
        else:
            if args.feature_extractor == 'RESNEXT101':
                self.transform = transforms.Compose([
                    transforms.Resize((224, 320)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                raise Exception('Wrong --feature_extractor option. ' + args.feature_extractor + ' unknown')

        self.inputs = []
        files = os.listdir(osp.join(args.data_root, args.model_target))
        if self.downsampling:
            random.shuffle(files)
        for filename in files:
            if filename.split('___')[1][:-4] not in self.sessions:
                continue

            target = np.load(osp.join(self.data_root, args.model_target, filename))
            # temporal data augmentation
            shift = np.random.randint(self.chunk_size) if self.training else 0
            target = target[shift:]
            # round to multiple of chunk_size
            num_frames = target.shape[0]
            num_frames = num_frames - (num_frames % args.chunk_size)
            target = target[:num_frames]
            # For each chunk, the central frame label is the label of the entire chunk
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
                        filename, target[idx_chunk], idx_chunk, shift
                    ])

    def __getitem__(self, index):
        if self.is_3D:
            return self.getitem_3D(index)
        else:
            return self.getitem_2D(index)

    def getitem_3D(self, index):
        filename, target, idx_chunk, shift = self.inputs[index]

        start_idx = int(filename.split('___')[0]) + shift
        start_idx = start_idx + idx_chunk * self.chunk_size
        raw_frames = []
        for i in range(self.chunk_size):
            idx_cur_frame = start_idx + i
            frame = Image.open(osp.join(self.data_root,
                                        self.model_input,
                                        filename.split('___')[1][:-4],
                                        str(idx_cur_frame)+'.jpg')).convert('RGB')
            frame = self.transform(frame).to(dtype=torch.float32)
            raw_frames.append(frame)

        raw_frames = torch.stack(raw_frames)
        raw_frames = raw_frames.permute(1, 0, 2, 3)    # channel before

        target = torch.as_tensor(target.astype(np.float32))

        return raw_frames, target

    def getitem_2D(self, index):
        filename, target, idx_chunk, shift = self.inputs[index]

        start_idx = int(filename.split('___')[0]) + shift
        start_idx = start_idx + idx_chunk * self.chunk_size
        idx_cur_frame = start_idx + self.chunk_size // 2

        frame = Image.open(osp.join(self.data_root,
                                    self.model_input,
                                    filename.split('___')[1][:-4],
                                    str(idx_cur_frame)+'.jpg')).convert('RGB')
        frame = self.transform(frame).to(dtype=torch.float32)

        target = torch.as_tensor(target.astype(np.float32))

        return frame, target


    def __len__(self):
        return len(self.inputs)