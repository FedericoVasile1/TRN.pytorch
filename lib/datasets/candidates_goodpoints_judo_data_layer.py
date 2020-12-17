import os
import os.path as osp
import numpy as np

import torch
import torch.utils.data as data

class CandidatesGoodpoints_JUDODataLayer(data.Dataset):
    def __init__(self, args, phase='train'):
        if args.eval_on_untrimmed:
            if phase == 'train':
                # load candidates and trimmed
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
                # load goodpoints
                self.datalayer2 = Goodpoints_PerType_JUDODataLayer(args, 'UNTRIMMED', phase)
                self.datalayer.inputs.extend(self.datalayer2.inputs)
                del self.datalayer2
            else:
                # load candidates
                self.datalayer = Candidates_PerType_JUDODataLayer(args, 'UNTRIMMED', phase)
                # load goodpoints
                self.datalayer2 = Goodpoints_PerType_JUDODataLayer(args, 'UNTRIMMED', phase)
                self.datalayer.inputs.extend(self.datalayer2.inputs)
                del self.datalayer2
        else:
            if args.use_trimmed == args.use_untrimmed == True:
                # load candidates and trimmed
                self.datalayer = Candidates_PerType_JUDODataLayer(args, 'UNTRIMMED', phase)
                self.appo = Candidates_PerType_JUDODataLayer(args, 'TRIMMED', phase)
                self.datalayer.inputs.extend(self.appo.inputs)
                del self.appo
                # load goodpoints
                self.datalayer2 = Goodpoints_PerType_JUDODataLayer(args, 'UNTRIMMED', phase)
                self.datalayer.inputs.extend(self.datalayer2.inputs)
                del self.datalayer2
            elif args.use_trimmed:
                self.datalayer = Candidates_PerType_JUDODataLayer(args, 'TRIMMED', phase)
            elif args.use_untrimmed:
                # laod candidates
                self.datalayer = Candidates_PerType_JUDODataLayer(args, 'UNTRIMMED', phase)
                # load goodpoints
                self.datalayer2 = Goodpoints_PerType_JUDODataLayer(args, 'UNTRIMMED', phase)
                self.datalayer.inputs.extend(self.datalayer2.inputs)
                del self.datalayer2

    def __getitem__(self, index):
        dataset_type, filename, step_target, start, end, untrimmed_type = self.datalayer.inputs[index]

        if dataset_type == 'TRIMMED':
            MODEL_INPUT = 'i3d_224x224_chunk9'
        else:
            if untrimmed_type == 'goodpoints':
                MODEL_INPUT = 'goodpoints_i3d_224x224_chunk9'
            elif untrimmed_type == 'candidates':
                MODEL_INPUT = self.datalayer.model_input    # i.e. 'candidates_i3d_224x224_chunk9' or 'candidatesV2_i3d_224x224_chunk9'
            else:
                raise Exception('Unknown model input')

        feature_vectors = np.load(osp.join(self.datalayer.data_root, dataset_type, MODEL_INPUT, filename),
                                  mmap_mode='r')
        feature_vectors = feature_vectors[start:end]

        heatmaps_feature_vectors = np.zeros_like(feature_vectors)

        feature_vectors = torch.as_tensor(feature_vectors.astype(np.float32))
        heatmaps_feature_vectors = torch.as_tensor(heatmaps_feature_vectors.astype(np.float32))
        step_target = torch.as_tensor(step_target.astype(np.float32))

        return feature_vectors, heatmaps_feature_vectors, step_target

    def __len__(self):
        return self.datalayer.__len__()

class Candidates_PerType_JUDODataLayer():
    def __init__(self, args, dataset_type, phase='train'):
        self.data_root = args.data_root
        self.model_input = args.model_input
        self.training = phase=='train'
        self.sessions = getattr(args, phase+'_session_set')[dataset_type]

        self.steps = args.steps
        if args.model_input.split('_')[0] == 'candidatesV2' and args.steps > 22:
            raise Exception('For '+args.model_input+' we supports only steps<=22 since these clips are 22 steps long.')
        if args.model_input.split('_')[0] == 'candidates' and args.steps > 11:
            raise Exception('For '+args.model_input+' we supports only steps==11 since these clips are 11 steps long.')

        self.inputs = []
        if dataset_type == 'UNTRIMMED':
            for filename in os.listdir(osp.join(args.data_root, dataset_type, args.model_target)):
                if filename.split('___')[1][:-4] not in self.sessions:
                    continue

                target = np.load(osp.join(self.data_root, dataset_type, args.model_target, filename))
                num_frames = target.shape[0]
                num_frames = num_frames - (num_frames % args.chunk_size)
                target = target[:num_frames]
                target = target[args.chunk_size // 2::args.chunk_size]

                if args.model_input.split('_')[0] == 'candidates':
                    seed = np.random.randint(11+1 - self.steps) if self.training else 0
                elif args.model_input.split('_')[0] == 'candidatesV2':
                    seed = np.random.randint(22+1 - self.steps) if self.training else 0
                else:
                    raise Exception('Unknown --model_input')
                for start, end in zip(range(seed, target.shape[0], self.steps),
                                      range(seed + self.steps, target.shape[0] + 1, self.steps)):
                    step_target = target[start:end]
                    self.inputs.append([
                        dataset_type, filename, step_target, start, end, 'candidates'
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

                seed = np.random.randint(28+1 - self.steps) if self.training else 0
                for start, end in zip(range(seed, target.shape[0], self.steps),
                                      range(seed + self.steps, target.shape[0] + 1, self.steps)):
                    step_target = target[start:end]
                    self.inputs.append([
                        dataset_type, filename+'.npy', step_target, start, end, ''
                    ])

            else:
                raise Exception('Unknown dataset')

class Goodpoints_PerType_JUDODataLayer(data.Dataset):
    def __init__(self, args, dataset_type, phase='train'):
        self.data_root = args.data_root
        self.training = phase=='train'
        self.sessions = getattr(args, phase+'_session_set')[dataset_type]

        # HARD-CODED
        MODEL_TARGET = 'goodpoints_4s_target_frames_25fps'

        self.steps = args.steps
        if self.steps > 28:
            raise Exception('The clips are 28 steps long, so --steps can not be greater than 28.')

        self.inputs = []
        if dataset_type == 'UNTRIMMED':
            for filename in os.listdir(osp.join(args.data_root, dataset_type, MODEL_TARGET)):
                if filename.split('___')[1][:-4] not in self.sessions:
                    continue

                target = np.load(osp.join(self.data_root, dataset_type, MODEL_TARGET, filename))
                # round to multiple of chunk_size
                num_frames = target.shape[0]
                num_frames = num_frames - (num_frames % args.chunk_size)
                target = target[:num_frames]
                # For each chunk, the central frame label is the label of the entire chunk
                target = target[args.chunk_size // 2::args.chunk_size]

                seed = np.random.randint(28+1 - self.steps) if self.training else 0
                for start, end in zip(range(seed, target.shape[0], self.steps),
                                      range(seed + self.steps, target.shape[0] + 1, self.steps)):
                    step_target = target[start:end]
                    self.inputs.append([
                        dataset_type, filename, step_target, start, end, 'goodpoints'
                    ])
        else:
            raise Exception('Unknown dataset')