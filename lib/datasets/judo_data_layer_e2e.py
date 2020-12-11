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
        self.training = phase=='train'
        self.sessions = getattr(args, phase+'_session_set')[dataset_type]

        self.chunk_size = args.chunk_size

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
            raise Exception('Wrong --feature_extractor option. CNN2D feature extractors are no longer supported.')

        self.inputs = []
        for session in self.sessions:
            if dataset_type == 'TRIMMED':
                if not osp.isfile(osp.join(self.data_root, dataset_type, '2s_target_frames_25fps', session+'.npy')):
                    # skip videos in which the pose model does not detect any fall(i.e. fall==-1  in fall_detections.csv).
                    # TODO: fix these videos later on, in order to incluso also them
                    continue

            target = np.load(osp.join(self.data_root,
                                      dataset_type,
                                      args.model_target if dataset_type=='UNTRIMMED' else '2s_target_frames_25fps',
                                      session+'.npy'))
            # temporal data augmentation
            shift = np.random.randint(self.chunk_size) if self.training else 0
            target = target[shift:]
            # round to multiple of chunk_size
            num_frames = target.shape[0]
            num_frames = num_frames - (num_frames  % self.chunk_size)
            target = target[:num_frames]
            # For each chunk, the central frame label is the label of the entire chunk
            target = target[self.chunk_size//2::self.chunk_size]

            for idx_chunk in range(target.shape[0]):
                self.inputs.append([
                    dataset_type, session, target[idx_chunk], idx_chunk, shift,
                ])

    def __getitem__(self, index):
        dataset_type, session, target, idx_chunk, shift = self.inputs[index]

        start_idx = shift + idx_chunk * self.chunk_size
        raw_frames = []
        for i in range(self.chunk_size):
            idx_cur_frame = start_idx + i
            frame = Image.open(osp.join(self.data_root,
                                        dataset_type,
                                        self.model_input if dataset_type=='UNTRIMMED' else 'i3d_224x224_chunk9',
                                        session,
                                        str(idx_cur_frame)+'.jpg')).convert('RGB')
            frame = transforms(frame).to(dtype=torch.float32)
            raw_frames.append(frame)

        raw_frames = torch.stack(raw_frames)
        raw_frames = raw_frames.permute(1, 0, 2, 3)     # channel before

        target = torch.as_tensor(target.astype(np.float32))

        return raw_frames, target

    def __len__(self):
        return len(self.inputs)

