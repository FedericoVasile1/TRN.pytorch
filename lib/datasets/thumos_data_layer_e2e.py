import os.path as osp
import numpy as np
import warnings

import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image

from lib.models.i3d.i3d import I3DNormalization

class THUMOSDataLayerE2E(data.Dataset):
    def __init__(self, args, phase='train'):
        self.data_root = args.data_root
        self.model_input = args.model_input
        self.steps = args.steps
        self.training = phase=='train'
        self.chunk_size = args.chunk_size
        self.sessions = getattr(args, phase+'_session_set')

        if args.is_3D:
            if args.feature_extractor == 'I3D':
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    I3DNormalization(),
                ])
            elif args.feature_extractor == 'RESNET2+1D':
                self.transform = transforms.Compose([
                    transforms.Resize((112, 112)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])
                ])
            else:
                raise Exception('Wrong --feature_extractor option. ' + args.feature_extractor + ' unknown')
            self.is_3D = True
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            self.is_3D = False

        # In case of 3D this means that a chunk of CHUNK_SIZE consecutive frames will be fed to the 3D model that will
        #  give us the corresponding feature vector as output. Furthermore, the label associated to the feature vector
        #  is the label of the central frame of the chunk.
        # In case of 2D this means that for each chunk only the central frame of the chunk is taken and fed into
        #  the 2D model to generate the feature vector.
        self.chunk_size = args.chunk_size

        self.inputs = []
        for session in self.sessions:
            target = np.load(osp.join(self.data_root, 'target_frames_24fps', session+'.npy'))
            # round to multiple of CHUNK_SIZE
            num_frames = target.shape[0]
            num_frames = num_frames - (num_frames % self.chunk_size)
            target = target[:num_frames]
            # For each chunk, the central frame label is the label of the entire chunk
            target = target[self.chunk_size // 2::self.chunk_size]

            seed = np.random.randint(self.steps) if self.training else 0
            for start, end in zip(range(seed, target.shape[0], self.steps),
                                  range(seed + self.steps, target.shape[0], self.steps)):

                step_target = target[start:end]
                self.inputs.append([
                    session, start, end, step_target,
                ])

    def __getitem__(self, index):
        if self.is_3D:
            return self.getitem_3D(index)
        else:
            return self.getitem_2D(index)

    def getitem_2D(self, index):
        session, start, end, step_target = self.inputs[index]

        raw_frames = None
        for count in range(start, end):
            idx_frame = count * self.chunk_size + (self.chunk_size // 2)
            frame = Image.open(osp.join(self.data_root,
                                        self.model_input,
                                        session,
                                        str(idx_frame + 1) + '.jpg')).convert('RGB')
            frame = self.transform(frame).to(dtype=torch.float32)
            if raw_frames is None:
                raw_frames = torch.zeros((end - start, frame.shape[0], frame.shape[1], frame.shape[2]),
                                         dtype=torch.float32)
            raw_frames[count - start] = frame

        step_target = torch.as_tensor(step_target.astype(np.float32))

        return raw_frames, step_target

    def getitem_3D(self, index):
        session, start, end, step_target = self.inputs[index]

        raw_frames = None
        for count in range(start, end):
            # retrieve the index of the central frame of the chunk
            idx_central_frame = count * self.chunk_size + (self.chunk_size // 2)
            # now load all of the frames of the chunk
            start_f = idx_central_frame - self.chunk_size // 2
            end_f = idx_central_frame + self.chunk_size // 2
            for idx_frame in range(start_f, end_f):
                frame = Image.open(osp.join(self.data_root,
                                            self.model_input,
                                            session,
                                            str(idx_frame + 1) + '.jpg')).convert('RGB')
                frame = self.transform(frame).to(dtype=torch.float32)
                if raw_frames is None:
                    raw_frames = torch.zeros((end - start, self.chunk_size, frame.shape[0], frame.shape[1], frame.shape[2]),
                                             dtype=torch.float32)
                raw_frames[count - start, idx_frame - start_f] = frame

        # switch channel with chunk_size (3d models want input in this way)
        raw_frames = raw_frames.permute(0, 2, 1, 3, 4)

        step_target = torch.as_tensor(step_target.astype(np.float32))

        return raw_frames, step_target

    def __len__(self):
        return len(self.inputs)

