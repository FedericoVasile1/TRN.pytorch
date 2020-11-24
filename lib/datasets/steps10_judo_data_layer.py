import os.path as osp
import numpy as np

import torch
import torch.utils.data as data

class Steps10JUDODataLayer(data.Dataset):
    def __init__(self, args, phase='train'):
        if args.eval_on_untrimmed:
            if phase == 'train':
                self.datalayer = _PerType_JUDODataLayer(args, 'UNTRIMMED', phase)
                self.appo1 = _PerType_JUDODataLayer(args, 'TRIMMED', 'train')
                self.appo2 = _PerType_JUDODataLayer(args, 'TRIMMED', 'val')
                self.appo3 = _PerType_JUDODataLayer(args, 'TRIMMED', 'test')
                self.datalayer.inputs.extend(self.appo1.inputs)
                self.datalayer.inputs.extend(self.appo2.inputs)
                self.datalayer.inputs.extend(self.appo3.inputs)
                del self.appo1
                del self.appo2
                del self.appo3
            else:
                self.datalayer = _PerType_JUDODataLayer(args, 'UNTRIMMED', phase)
        else:
            if args.use_trimmed == args.use_untrimmed == True:
                self.datalayer = _PerType_JUDODataLayer(args, 'UNTRIMMED', phase)
                self.appo = _PerType_JUDODataLayer(args, 'TRIMMED', phase)
                self.datalayer.inputs.extend(self.appo.inputs)
                del self.appo
            elif args.use_trimmed:
                self.datalayer = _PerType_JUDODataLayer(args, 'TRIMMED', phase)
            elif args.use_untrimmed:
                self.datalayer = _PerType_JUDODataLayer(args, 'UNTRIMMED', phase)

    def __getitem__(self, index):
        return self.datalayer.__getitem__(index)

    def __len__(self):
        return self.datalayer.__len__()

class _PerType_JUDODataLayer(data.Dataset):
    def __init__(self, args, dataset_type, phase='train'):
        self.data_root = args.data_root
        self.model_input = args.model_input
        self.steps = 240 // 9       # TODO
        self.training = phase=='train'
        self.sessions = getattr(args, phase+'_session_set')[dataset_type]

        self.inputs = []
        for session in self.sessions:
            target = np.load(osp.join(self.data_root, dataset_type, '10s_target_frames_25fps', session+'.npy'))
            # round to multiple of chunk_size
            num_frames = target.shape[0]
            num_frames = num_frames - (num_frames % args.chunk_size)
            target = target[:num_frames]
            # For each chunk, the central frame label is the label of the entire chunk
            target = target[args.chunk_size // 2::args.chunk_size]

            count = 1
            for t in range(len(target)):
                if t == 0:
                    continue

                if np.argmax(target[t-1]) != np.argmax(target[t]):
                    count = 1
                else:
                    count += 1
                    if count == self.steps:
                        self.inputs.append([dataset_type, session, t+1-self.steps, t+1, target[t+1-self.steps:t+1]])

                    count = 0

    def __getitem__(self, index):
        dataset_type, session, start, end, step_target = self.inputs[index]

        feature_vectors = np.load(osp.join(self.data_root, dataset_type, self.model_input, session+'.npy'),
                                  mmap_mode='r')
        feature_vectors = feature_vectors[start:end]
        feature_vectors = torch.as_tensor(feature_vectors.astype(np.float32))

        step_target = torch.as_tensor(step_target.astype(np.float32))

        return feature_vectors, step_target

    def __len__(self):
        return len(self.inputs)