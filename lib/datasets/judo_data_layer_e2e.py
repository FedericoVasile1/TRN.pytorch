import os.path as osp
import numpy as np

import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image

from lib.models.i3d.i3d import I3DNormalization

class JUDODataLayerE2E(data.Dataset):
    def __init__(self, args, phase='train'):
        if args.eval_on_untrimmed:
            if phase == 'train':
                self.datalayer = _PerType_JUDODataLayerE2E(args, 'UNTRIMMED', phase)
                self.appo1 = _PerType_JUDODataLayerE2E(args, 'TRIMMED', 'train')
                self.appo2 = _PerType_JUDODataLayerE2E(args, 'TRIMMED', 'val')
                self.appo3 = _PerType_JUDODataLayerE2E(args, 'TRIMMED', 'test')
                self.datalayer.inputs.extend(self.appo1.inputs)
                self.datalayer.inputs.extend(self.appo2.inputs)
                self.datalayer.inputs.extend(self.appo3.inputs)
                del self.appo1
                del self.appo2
                del self.appo3
            else:
                self.datalayer = _PerType_JUDODataLayerE2E(args, 'UNTRIMMED', phase)
        else:
            if args.use_trimmed == args.use_untrimmed == True:
                self.datalayer = _PerType_JUDODataLayerE2E(args, 'UNTRIMMED', phase)
                self.appo = _PerType_JUDODataLayerE2E(args, 'TRIMMED', phase)
                self.datalayer.inputs.extend(self.appo.inputs)
                del self.appo
            elif args.use_trimmed:
                self.datalayer = _PerType_JUDODataLayerE2E(args, 'TRIMMED', phase)
            elif args.use_untrimmed:
                self.datalayer = _PerType_JUDODataLayerE2E(args, 'UNTRIMMED', phase)

    def __getitem__(self, index):
        return self.datalayer.__getitem__(index)

    def __len__(self):
        return self.datalayer.__len__()

class _PerType_JUDODataLayerE2E(data.Dataset):
    def __init__(self, args, dataset_type, phase='train'):
        self.data_root = args.data_root
        self.model_input = args.model_input
        self.steps = args.steps
        self.training = phase=='train'
        self.chunk_size = args.chunk_size
        self.sessions = getattr(args, phase+'_session_set')[dataset_type]

        if args.is_3D:
            if args.feature_extractor == 'I3D':
                self.transform = transforms.Compose([
                    transforms.Resize((224, 320)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    I3DNormalization(),
                ])
            elif args.feature_extractor == 'RESNET2+1D':
                self.transform = transforms.Compose([
                    transforms.Resize((224, 320)),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])
                ])
            else:
                raise Exception('Wrong --feature_extractor option. ' + args.feature_extractor + ' unknown')
            self.is_3D = True
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 320)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            self.is_3D = False

        self.inputs = []
        for session in self.sessions:
            if not osp.isfile(osp.join(self.data_root, dataset_type, args.model_target, session + '.npy')):
                # skip videos in which the pose model does not detect any fall(i.e. fall==-1  in fall_detections.csv).
                # TODO: fix these videos later on, in order to incluso also them
                continue

            target = np.load(osp.join(self.data_root, dataset_type, args.model_target, session+'.npy'))
            # round to multiple of CHUNK_SIZE
            num_frames = target.shape[0]
            num_frames = num_frames - (num_frames  % self.chunk_size)
            target = target[:num_frames]
            # For each chunk, the central frame label is the label of the entire chunk
            target = target[self.chunk_size//2::self.chunk_size]

            seed = np.random.randint(self.steps) if self.training else 0
            for start, end in zip(range(seed, target.shape[0], self.steps),
                                  range(seed + self.steps, target.shape[0], self.steps)):

                step_target = target[start:end]
                self.inputs.append([
                    dataset_type, session, start, end, step_target,
                ])

    def __getitem__(self, index):
        if self.is_3D:
            return self.getitem_3D(index)
        else:
            return self.getitem_2D(index)

    def getitem_2D(self, index):
        dataset_type, session, start, end, step_target = self.inputs[index]

        raw_frames = None
        for count in range(start, end):
            idx_frame = count * self.chunk_size + (self.chunk_size // 2)
            frame = Image.open(osp.join(self.data_root,
                                        dataset_type,
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
        dataset_type, session, start, end, step_target = self.inputs[index]

        raw_frames = None
        for count in range(start, end):
            # retrieve the index of the central frame of the chunk
            idx_central_frame = count * self.chunk_size + (self.chunk_size // 2)
            # now load all of the frames of the chunk
            start_f = idx_central_frame - self.chunk_size // 2
            end_f = idx_central_frame + self.chunk_size // 2
            for idx_frame in range(start_f, end_f):
                frame = Image.open(osp.join(self.data_root,
                                            dataset_type,
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

