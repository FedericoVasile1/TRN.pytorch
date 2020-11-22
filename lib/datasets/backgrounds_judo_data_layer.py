import os.path as osp
import numpy as np

import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image

class Backgrounds_JUDODataLayer(data.Dataset):
    def __init__(self, args, phase='train'):
        if args.eval_on_untrimmed:
            if phase == 'train':
                self.datalayer = _Backgrounds_JUDODataLayer(args, 'UNTRIMMED', phase)
                self.appo1 = _Backgrounds_JUDODataLayer(args, 'TRIMMED', 'train')
                self.appo2 = _Backgrounds_JUDODataLayer(args, 'TRIMMED', 'val')
                self.appo3 = _Backgrounds_JUDODataLayer(args, 'TRIMMED', 'test')
                self.datalayer.inputs.extend(self.appo1.inputs)
                self.datalayer.inputs.extend(self.appo2.inputs)
                self.datalayer.inputs.extend(self.appo3.inputs)
                del self.appo1
                del self.appo2
                del self.appo3
            else:
                self.datalayer = _Backgrounds_JUDODataLayer(args, 'UNTRIMMED', phase)
        else:
            if args.use_trimmed == args.use_untrimmed == True:
                self.datalayer = _Backgrounds_JUDODataLayer(args, 'UNTRIMMED', phase)
                self.appo = _Backgrounds_JUDODataLayer(args, 'TRIMMED', phase)
                self.datalayer.inputs.extend(self.appo.inputs)
                del self.appo
            elif args.use_trimmed:
                self.datalayer = _Backgrounds_JUDODataLayer(args, 'TRIMMED', phase)
            elif args.use_untrimmed:
                self.datalayer = _Backgrounds_JUDODataLayer(args, 'UNTRIMMED', phase)

    def __getitem__(self, index):
        return self.datalayer.__getitem__(index)

    def __len__(self):
        return self.datalayer.__len__()

class _Backgrounds_JUDODataLayer(data.Dataset):
    def __init__(self, args, dataset_type, phase='train'):
        self.data_root = args.data_root
        self.model_input = args.model_input     # video_frames_25fps
        self.steps = args.steps
        if self.steps != 50:
            raise Exception('Mini clips two seconds long')
        self.training = phase=='train'
        self.sessions = getattr(args, phase+'_session_set')[dataset_type]

        self.chunk_size = args.chunk_size

        self.transform = transforms.Compose([
            transforms.Resize((227, 324)),
            transforms.CenterCrop(227),
            transforms.ToTensor(),
            #transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])       # TODO ????
        ])

        self.inputs = []
        for session in self.sessions:
            target = np.load(osp.join(self.data_root, 'UNTRIMMED', '10s_target_frames_25fps', session+'.npy'))
            #seed = np.random.randint(self.steps) if self.training else 0
            target = target[::self.chunk_size]
            count = 0
            for idx in range(len(target)):
                if target[idx, 0] == 0:
                    count = 0
                    continue

                count += 1
                if count == self.steps:
                    self.inputs.append([session, idx+1-count])
                    count = 0

    def __getitem__(self, index):
        session, start_idx = self.inputs[index]
        frames = []
        for step in range(self.steps):
            raw_frame = Image.open(osp.join(self.data_root,
                                            'UNTRIMMED',
                                            self.model_input,
                                            session,
                                            str((start_idx+step)*self.chunk_size+1)+'.jpg')).convert('RGB')
            raw_frame = self.transform(raw_frame)
            frames.append(raw_frame)

        frames = torch.stack(frames)
        return frames

    def __len__(self):
        return len(self.inputs)