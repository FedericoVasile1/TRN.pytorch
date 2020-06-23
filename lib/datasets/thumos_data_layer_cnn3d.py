import os.path as osp

import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import numpy as np

'''
WARNING: this class supports now only 3d feature extractor, change self.transform to support others
'''
class CNN3DTHUMOSDataLayer(data.Dataset):
    def __init__(self, args, phase='train'):
        self.data_root = args.data_root
        self.camera_feature = args.camera_feature
        self.sessions = getattr(args, phase+'_session_set')
        self.training = phase=='train'
        self.transform = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])
        ])

        self.CHUNK_SIZE = 6

        self.inputs = []
        if not self.training:
            self.sessions = self.sessions[0:50]
            pass
        for session in self.sessions:
            target = np.load(osp.join(self.data_root, 'target_frames_24fps', session+'.npy'))
            # round to multiple of CHUNK_SIZE
            num_frames = target.shape[0]
            num_frames = num_frames - (num_frames  % self.CHUNK_SIZE)
            target = target[:num_frames]

            # For each chunk, take only the central frame, the label of the central frame will be the
            #  label of the entire chunk
            target = target[self.CHUNK_SIZE//2::self.CHUNK_SIZE]

            for idx in range(len(target)):
                self.inputs.append([session, idx, target[idx]])

    def __getitem__(self, index):
        session, idx, label = self.inputs[index]

        camera_inputs = None
        # retrieve the index of the central frame of the chunk
        idx_central_frame = idx * self.CHUNK_SIZE + (self.CHUNK_SIZE // 2)
        # now load all of the frames of the chunk
        start_f = idx_central_frame - self.CHUNK_SIZE // 2
        end_f = idx_central_frame + self.CHUNK_SIZE // 2
        for idx_frame in range(start_f, end_f):
            frame = Image.open(osp.join(self.data_root, 'video_frames_24fps', session, str(idx_frame+1)+'.jpg')).convert('RGB')
            frame = self.transform(frame).to(dtype=torch.float32)
            if camera_inputs is None:
                camera_inputs = torch.zeros((self.CHUNK_SIZE, frame.shape[0], frame.shape[1], frame.shape[2]),
                                            dtype=torch.float32)
            camera_inputs[idx_frame - start_f] = frame

        # switch channel with chunk_size (3d models want input in this way)
        camera_inputs = camera_inputs.permute(1, 0, 2, 3)

        return camera_inputs, label

    def __len__(self):
        return len(self.inputs)