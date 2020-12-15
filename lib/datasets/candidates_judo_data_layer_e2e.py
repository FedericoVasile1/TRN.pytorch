import os
import os.path as osp
import numpy as np

import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image

from lib.models.i3d.i3d import I3DNormalization

class Candidates_JUDODataLayerE2E(data.Dataset):
    def __init__(self, args, phase='train'):
        if args.eval_on_untrimmed:
            raise Exception('In EndToEnd mode only --use_untrimmed option is supported, so drop --eval_on_untrimmed')
        else:
            if args.use_trimmed == args.use_untrimmed == True:
                raise Exception('In EndToEnd mode only --use_untrimmed option is supported, so drop --use_trimmed')
            elif args.use_trimmed:
                raise Exception('In EndToEnd mode only --use_untrimmed option is supported, so drop --use_trimmed')
            elif args.use_untrimmed:
                self.datalayer = Candidates_PerType_JUDODataLayerE2E(args, 'UNTRIMMED', phase)

    def __getitem__(self, index):
        return self.datalayer.__getitem__(index)

    def __len__(self):
        return self.datalayer.__len__()

class Candidates_PerType_JUDODataLayerE2E(data.Dataset):
    def __init__(self, args, dataset_type, phase='train'):
        if args.model != 'CNN3D':
            raise Exception('This class is intended to be used only for CNN3D model since it does not take into '
                            'account the --steps arguments, i.e. each chunk is treated independently so one chunk'
                            'is returned instead a sequence of subsequent chunks "--steps" long.')

        self.data_root = args.data_root
        self.model_input = args.model_input
        self.training = phase=='train'
        self.sessions = getattr(args, phase+'_session_set')[dataset_type]

        self.chunk_size = args.chunk_size

        if args.feature_extractor == 'I3D':
            self.transform = transforms.Compose([
                transforms.Resize((224, 320)),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                I3DNormalization(),
            ])
        else:
            raise Exception('Wrong --feature_extractor option. ' + args.feature_extractor + ' unknown')

        self.inputs = []
        for filename in os.listdir(osp.join(args.data_root, dataset_type, args.model_target)):
            if filename.split('___')[1][:-4] not in self.sessions:
                continue

            target = np.load(osp.join(self.data_root, dataset_type, args.model_target, filename))
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
                self.inputs.append([
                    dataset_type, filename, target[idx_chunk], idx_chunk, shift
                ])

    def __getitem__(self, index):
        dataset_type, filename, target, idx_chunk, shift = self.inputs[index]

        start_idx = int(filename.split('___')[0]) + shift
        start_idx = start_idx + idx_chunk * self.chunk_size
        raw_frames = []
        for i in range(self.chunk_size):
            idx_cur_frame = start_idx + i
            frame = Image.open(osp.join(self.data_root,
                                        dataset_type,
                                        self.model_input,
                                        filename[:-4],
                                        str(idx_cur_frame)+'.jpg')).convert('RGB')
            frame = self.transform(frame).to(dtype=torch.float32)
            raw_frames.append(frame)

        raw_frames = torch.stack(raw_frames)
        raw_frames = raw_frames.permute(1, 0, 2, 3)    # channel before

        target = torch.as_tensor(target.astype(np.float32))

        return raw_frames, target

    def __len__(self):
        return len(self.inputs)