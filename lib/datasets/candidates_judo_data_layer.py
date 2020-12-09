import os
import os.path as osp
import numpy as np

import torch
import torch.utils.data as data

class Candidates_JUDODataLayer(data.Dataset):
    def __init__(self, args, phase='train'):
        if args.eval_on_untrimmed:
            if phase == 'train':
                self.datalayer = Candidates_PerType_JUDODataLayer(args, 'UNTRIMMED', phase)
                self.appo1 = Candidates_PerType_JUDODataLayer(args, 'TRIMMED', 'train')
                self.appo2 = Candidates_PerType_JUDODataLayer(args, 'TRIMMED', 'val')
                self.appo3 = Candidates_PerType_JUDODataLayer(args, 'TRIMMED', 'test')
                self.datalayer.inputs.extend(self.appo1.inputs)
                self.datalayer.inputs.extend(self.appo2.inputs)
                self.datalayer.inputs.extend(self.appo3.inputs)
                del self.appo1
                del self.appo2
                del self.appo3
            else:
                self.datalayer = Candidates_PerType_JUDODataLayer(args, 'UNTRIMMED', phase)
        else:
            if args.use_trimmed == args.use_untrimmed == True:
                self.datalayer = Candidates_PerType_JUDODataLayer(args, 'UNTRIMMED', phase)
                self.appo = Candidates_PerType_JUDODataLayer(args, 'TRIMMED', phase)
                self.datalayer.inputs.extend(self.appo.inputs)
                del self.appo
            elif args.use_trimmed:
                self.datalayer = Candidates_PerType_JUDODataLayer(args, 'TRIMMED', phase)
            elif args.use_untrimmed:
                self.datalayer = Candidates_PerType_JUDODataLayer(args, 'UNTRIMMED', phase)

    def __getitem__(self, index):
        return self.datalayer.__getitem__(index)

    def __len__(self):
        return self.datalayer.__len__()

class Candidates_PerType_JUDODataLayer(data.Dataset):
    def __init__(self, args, dataset_type, phase='train'):
        self.data_root = args.data_root
        self.model_input = args.model_input
        self.training = phase=='train'
        self.sessions = getattr(args, phase+'_session_set')[dataset_type]

        if dataset_type == 'TRIMMED':
            args.model_target = 'target_frames_25fps'

        self.inputs = []
        for filename in os.listdir(osp.join(args.data_root, dataset_type, args.model_target)):
            if dataset_type == 'UNTRIMMED':
                if filename.split('___')[1][:-4] not in self.sessions:
                    continue

            target = np.load(osp.join(self.data_root, dataset_type, args.model_target, filename))

            # TODO: to decide whether or not to add data augmentation along the temporal dimension

            self.inputs.append([
                dataset_type, filename, target
            ])

    def __getitem__(self, index):
        dataset_type, filename, target = self.inputs[index]

        feature_vectors = np.load(osp.join(self.data_root, dataset_type, self.model_input, filename),
                                  mmap_mode='r')
        feature_vectors = torch.as_tensor(feature_vectors.astype(np.float32))

        target = torch.as_tensor(target.astype(np.float32))

        return feature_vectors, target

    def __len__(self):
        return len(self.inputs)