import os.path as osp
import numpy as np

import torch
import torch.utils.data as data

class JUDODataLayer(data.Dataset):
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
        self.steps = args.steps
        self.training = phase=='train'
        self.sessions = getattr(args, phase+'_session_set')[dataset_type]
        self.use_heatmaps = args.use_heatmaps

        self.inputs = []
        for session in self.sessions:
            if not osp.isfile(osp.join(self.data_root, dataset_type, args.model_target, session+'.npy')):
                # skip videos in which the pose model does not detect any fall(i.e. fall==-1  in fall_detections.csv).
                # TODO: fix these videos later on, in order to include also them
                continue

            target = np.load(osp.join(self.data_root, dataset_type, args.model_target, session+'.npy'))
            # round to multiple of chunk_size
            num_frames = target.shape[0]
            num_frames = num_frames - (num_frames % args.chunk_size)
            target = target[:num_frames]
            # For each chunk, the central frame label is the label of the entire chunk
            target = target[args.chunk_size // 2::args.chunk_size]

            seed = np.random.randint(self.steps) if self.training else 0
            for start, end in zip(range(seed, target.shape[0], self.steps),
                                  range(seed + self.steps, target.shape[0], self.steps)):

                if args.downsample_backgr and self.training:
                    background_vect = np.zeros_like(target[start:end])
                    background_vect[:, 0] = 1
                    if (target[start:end] == background_vect).all():
                        continue

                step_target = target[start:end]
                self.inputs.append([
                    dataset_type, session, start, end, step_target,
                ])

    def __getitem__(self, index):
        dataset_type, session, start, end, step_target = self.inputs[index]

        feature_vectors = np.load(osp.join(self.data_root, dataset_type, self.model_input, session+'.npy'),
                                  mmap_mode='r')
        feature_vectors = feature_vectors[start:end]
        feature_vectors = torch.as_tensor(feature_vectors.astype(np.float32))

        if dataset_type == 'UNTRIMMED' and self.use_heatmaps:
            heatmaps_feature_vectors = np.load(osp.join(self.data_root, dataset_type, 'heatmaps_i3d_224x224_chunk9', session + '.npy'),
                                               mmap_mode='r')
            heatmaps_feature_vectors = heatmaps_feature_vectors[start:end]
            heatmaps_feature_vectors = torch.as_tensor(heatmaps_feature_vectors.astype(np.float32))
        else:
            heatmaps_feature_vectors = torch.zeros_like(feature_vectors)

        step_target = torch.as_tensor(step_target.astype(np.float32))

        return feature_vectors, heatmaps_feature_vectors, step_target

    def __len__(self):
        return len(self.inputs)