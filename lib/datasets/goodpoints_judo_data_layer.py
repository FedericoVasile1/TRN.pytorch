import os
import os.path as osp
import numpy as np

import torch
import torch.utils.data as data

class Goodpoints_JUDODataLayer(data.Dataset):
    def __init__(self, args, phase='train'):
        if args.eval_on_untrimmed:
            if phase == 'train':
                self.datalayer = Goodpoints_PerType_JUDODataLayer(args, 'UNTRIMMED', phase)
                self.appo1 = Goodpoints_PerType_JUDODataLayer(args, 'TRIMMED', 'train')
                self.appo2 = Goodpoints_PerType_JUDODataLayer(args, 'TRIMMED', 'val')
                self.appo3 = Goodpoints_PerType_JUDODataLayer(args, 'TRIMMED', 'test')
                self.datalayer.inputs.extend(self.appo1.inputs)
                self.datalayer.inputs.extend(self.appo2.inputs)
                self.datalayer.inputs.extend(self.appo3.inputs)
                del self.appo1
                del self.appo2
                del self.appo3
            else:
                self.datalayer = Goodpoints_PerType_JUDODataLayer(args, 'UNTRIMMED', phase)
        else:
            if args.use_trimmed == args.use_untrimmed == True:
                self.datalayer = Goodpoints_PerType_JUDODataLayer(args, 'UNTRIMMED', phase)
                self.appo = Goodpoints_PerType_JUDODataLayer(args, 'TRIMMED', phase)
                self.datalayer.inputs.extend(self.appo.inputs)
                del self.appo
            elif args.use_trimmed:
                self.datalayer = Goodpoints_PerType_JUDODataLayer(args, 'TRIMMED', phase)
            elif args.use_untrimmed:
                self.datalayer = Goodpoints_PerType_JUDODataLayer(args, 'UNTRIMMED', phase)

    def __getitem__(self, index):
        return self.datalayer.__getitem__(index)

    def __len__(self):
        return self.datalayer.__len__()

class Goodpoints_PerType_JUDODataLayer(data.Dataset):
    def __init__(self, args, dataset_type, phase='train'):
        self.data_root = args.data_root
        self.model_input = args.model_input
        self.steps = args.steps
        self.training = phase=='train'
        self.sessions = getattr(args, phase+'_session_set')[dataset_type]


        self.inputs = []
        if dataset_type == 'UNTRIMMED':
            for filename in os.listdir(osp.join(args.data_root, dataset_type, args.model_target)):
                if filename.split('___')[1][:-4] not in self.sessions:
                    continue

                target = np.load(osp.join(self.data_root, dataset_type, args.model_target, filename))
                # round to multiple of chunk_size
                num_frames = target.shape[0]
                num_frames = num_frames - (num_frames % args.chunk_size)
                target = target[:num_frames]
                # For each chunk, the central frame label is the label of the entire chunk
                target = target[args.chunk_size // 2::args.chunk_size]

                seed = np.random.randint(self.steps) if self.training else 0
                for start, end in zip(range(seed, target.shape[0], self.steps),
                                      range(seed + self.steps, target.shape[0], self.steps)):

                    step_target = target[start:end]
                    self.inputs.append([
                        dataset_type, filename, start, end, step_target,
                    ])
        elif dataset_type == 'TRIMMED':
            for filename in self.sessions:
                if not osp.isfile(osp.join(self.data_root, dataset_type, '2s_target_frames_25fps', filename+'.npy')):
                    # skip videos in which the pose model does not detect any fall(i.e. fall==-1  in fall_detections.csv).
                    # TODO: fix these videos later on, in order to include also them
                    continue

                target = np.load(osp.join(self.data_root, dataset_type, '2s_target_frames_25fps', filename+'.npy'))
                # round to multiple of chunk_size
                num_frames = target.shape[0]
                num_frames = num_frames - (num_frames % args.chunk_size)
                target = target[:num_frames]
                # For each chunk, the central frame label is the label of the entire chunk
                target = target[args.chunk_size // 2::args.chunk_size]

                seed = np.random.randint(self.steps) if self.training else 0
                for start, end in zip(range(seed, target.shape[0], self.steps),
                                      range(seed + self.steps, target.shape[0], self.steps)):

                    step_target = target[start:end]
                    self.inputs.append([
                        dataset_type, filename+'.npy', step_target, start, end
                    ])
        else:
            raise Exception('Unknown dataset')

    def __getitem__(self, index):
        dataset_type, filename, start, end, step_target = self.inputs[index]

        feature_vectors = np.load(osp.join(self.data_root,
                                           dataset_type,
                                           self.model_input if dataset_type=='UNTRIMMED' else self.model_input[3:],
                                           filename),
                                  mmap_mode='r')
        feature_vectors = feature_vectors[start:end]
        feature_vectors = torch.as_tensor(feature_vectors.astype(np.float32))

        step_target = torch.as_tensor(step_target.astype(np.float32))

        return feature_vectors, step_target

    def __len__(self):
        return len(self.inputs)