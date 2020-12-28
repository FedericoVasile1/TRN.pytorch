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
        self.use_heatmaps = args.use_heatmaps and dataset_type=='UNTRIMMED'
        self.use_untrimmed = args.use_untrimmed
        self.use_trimmed = args.use_trimmed
        self.class_to_count = {idx_class+1: 0 for idx_class in range(args.num_classes-1)}

        self.inputs = []
        for session in self.sessions:
            if not osp.isfile(osp.join(self.data_root, dataset_type, args.model_target, session+'.npy')):
                # skip trimmed videos in which the pose model does
                #  not detect any fall(i.e. fall==-1 in fall_detections.csv).
                # TODO: fix these videos later on, in order to include also them
                continue

            target = np.load(osp.join(self.data_root,
                                      dataset_type,
                                      args.model_target if dataset_type=='UNTRIMMED' else '2s_target_frames_25fps',
                                      session+'.npy'))
            # round to multiple of chunk_size
            num_frames = target.shape[0]
            num_frames = num_frames - (num_frames % args.chunk_size)
            target = target[:num_frames]
            # For each chunk, the central frame label is the label of the entire chunk
            target = target[args.chunk_size // 2::args.chunk_size]

            if dataset_type == 'UNTRIMMED':
                seed = np.random.randint(self.steps) if self.training else 0
            else:
                seed = np.random.randint(28+1 - self.steps) if self.training else 0
            for start, end in zip(range(seed, target.shape[0], self.steps),
                                  range(seed + self.steps, target.shape[0] + 1, self.steps)):
                if args.downsample_backgr and self.training:
                    background_vect = np.zeros_like(target[start:end])
                    background_vect[:, 0] = 1
                    if (target[start:end] == background_vect).all():
                        continue

                step_target = target[start:end]

                unique, counts = np.unique(step_target.argmax(axis=1), return_counts=True)

                # drop if action samples are greater than threshold
                for action_idx, num_samples in self.class_to_count.items():
                    if num_samples < 2500:
                        continue
                    if action_idx in step_target.argmax(axis=1):
                        continue

                # count actions labels
                for i, action_idx in enumerate(unique):
                    if action_idx == 0:
                        # ignore background class
                        continue
                    self.class_to_count[action_idx] += counts[i]

                self.inputs.append([
                    dataset_type, session, step_target, start, end
                ])

    def __getitem__(self, index):
        dataset_type, session, step_target, start, end = self.inputs[index]

        MODEL_INPUT = None
        if self.use_trimmed and not self.use_untrimmed:
            MODEL_INPUT = self.model_input
        elif self.use_trimmed and self.use_untrimmed:
            if dataset_type == 'UNTRIMMED':
                MODEL_INPUT = self.model_input
            elif dataset_type == 'TRIMMED':
                MODEL_INPUT = 'i3d_224x224_chunk9'
        elif self.use_untrimmed and not self.use_trimmed:
            MODEL_INPUT = self.model_input

        feature_vectors = np.load(osp.join(self.data_root,
                                           dataset_type,
                                           MODEL_INPUT,
                                           session+'.npy'),
                                  mmap_mode='r')
        feature_vectors = feature_vectors[start:end]

        if self.use_heatmaps:
            heatmaps_feature_vectors = np.load(osp.join(self.data_root,
                                                        dataset_type,
                                                        'heatmaps_'+self.model_input,
                                                        session + '.npy'),
                                               mmap_mode='r')
            heatmaps_feature_vectors = heatmaps_feature_vectors[start:end]
        else:
            heatmaps_feature_vectors = np.zeros_like(feature_vectors)

        feature_vectors = torch.as_tensor(feature_vectors.astype(np.float32))
        heatmaps_feature_vectors = torch.as_tensor(heatmaps_feature_vectors.astype(np.float32))
        step_target = torch.as_tensor(step_target.astype(np.float32))

        return feature_vectors, heatmaps_feature_vectors, step_target

    def __len__(self):
        return len(self.inputs)