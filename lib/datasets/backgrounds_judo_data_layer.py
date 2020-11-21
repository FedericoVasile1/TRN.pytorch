import os.path as osp
import numpy as np

import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image

class Backgrounds_JUDODataLayer(data.Dataset):
    def __init__(self, args, dataset_type, phase='train'):
        self.data_root = args.data_root
        self.model_input = args.model_input     # video_frames_25fps
        self.steps = args.steps
        if self.steps != 50:
            raise Exception('Mini clips two seconds long')
        self.training = phase=='train'
        self.sessions = getattr(args, phase+'_session_set')[dataset_type]

        self.transform = transforms.Compose([
            transforms.Resize((224, 320)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            #transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])       # TODO ????
        ])

        self.inputs = []
        for session in self.sessions:
            target = np.load(osp.join(self.data_root, 'UNTRIMMED', '10s_target_frames_25fps', session+'.npy'))
            #seed = np.random.randint(self.steps) if self.training else 0
            appo = []
            for idx in len(target):
                if target[idx, 0] == 0:
                    appo = []
                    continue

                appo.append(target[idx])
                if len(appo) == self.steps:
                    self.inputs.append([session, idx+1-args.steps])

    def __getitem__(self, index):
        session, start_idx = self.inputs[index]
        frames = []
        for step in self.steps:
            raw_frame = Image.open(osp.join(self.data_root,
                                            'UNTRIMMED',
                                            self.model_input,
                                            session,
                                            str(step+1)+'.jpg')).convert('RGB')
            raw_frame = self.transform(raw_frame)
            frames.append(raw_frame)

        frames = torch.stack(frames)
        return frames

    def __len__(self):
        return len(self.inputs)