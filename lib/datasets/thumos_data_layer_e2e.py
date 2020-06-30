import os.path as osp

import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import numpy as np

'''
WARNING: this class supports now only 3d feature extractor, change self.transform to support others
'''
class TRNTHUMOSDataLayerE2E(data.Dataset):
    def __init__(self, args, phase='train'):
        self.data_root = args.data_root
        self.camera_feature = args.camera_feature
        self.motion_feature = args.motion_feature
        self.sessions = getattr(args, phase+'_session_set')
        self.enc_steps = args.enc_steps
        self.dec_steps = args.dec_steps
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

            seed = np.random.randint(self.enc_steps) if self.training else 0
            for start, end in zip(
                range(seed, target.shape[0] - self.dec_steps, self.enc_steps),
                range(seed + self.enc_steps, target.shape[0] - self.dec_steps, self.enc_steps)):
                if args.downsample_backgr and self.training:
                    background_vect = np.zeros_like(target[start:end])
                    background_vect[:, 0] = 1
                    if (target[start:end] == background_vect).all():
                        continue

                enc_target = target[start:end]
                dec_target = self.get_dec_target(target[start:end + self.dec_steps])
                self.inputs.append([
                    session, start, end, enc_target, dec_target,
                ])

    def get_dec_target(self, target_vector):
        target_matrix = np.zeros((self.enc_steps, self.dec_steps, target_vector.shape[-1]))
        for i in range(self.enc_steps):
            for j in range(self.dec_steps):
                # 0 -> [1, 2, 3]
                # target_matrix[i,j] = target_vector[i+j+1,:]
                # 0 -> [0, 1, 2]
                target_matrix[i,j] = target_vector[i+j,:]
        return target_matrix

    def __getitem__(self, index):
        session, start, end, enc_target, dec_target = self.inputs[index]

        camera_inputs = None
        for count in range(start, end):
            # retrieve the index of the central frame of the chunk
            idx_central_frame = count * self.CHUNK_SIZE + (self.CHUNK_SIZE // 2)
            # now load all of the frames of the chunk
            start_f = idx_central_frame - self.CHUNK_SIZE // 2
            end_f = idx_central_frame + self.CHUNK_SIZE // 2
            for idx_frame in range(start_f, end_f):
                frame = Image.open(osp.join(self.data_root, self.camera_feature, session, str(idx_frame+1) + '.jpg')).convert('RGB')
                frame = self.transform(frame).to(dtype=torch.float32)
                if camera_inputs is None:
                    camera_inputs = torch.zeros((end - start, self.CHUNK_SIZE, frame.shape[0], frame.shape[1], frame.shape[2]),
                                                dtype=torch.float32)
                camera_inputs[count - start, idx_frame - start_f] = frame

        motion_inputs = np.zeros((end - start, self.enc_steps))  # optical flow will not be used
        motion_inputs = torch.as_tensor(motion_inputs.astype(np.float32))
        enc_target = torch.as_tensor(enc_target.astype(np.float32))
        dec_target = torch.as_tensor(dec_target.astype(np.float32))

        # switch channel with chunk_size (3d models want input in this way)
        camera_inputs = camera_inputs.permute(0, 2, 1, 3, 4)

        return camera_inputs, motion_inputs, enc_target, dec_target.view(-1, enc_target.shape[-1])

    def __len__(self):
        return len(self.inputs)

class TRNTHUMOSDataLayerVGG(data.Dataset):
    def __init__(self, args, phase='train'):
        self.data_root = args.data_root
        self.camera_feature = args.camera_feature
        self.motion_feature = args.motion_feature
        self.sessions = getattr(args, phase+'_session_set')
        self.enc_steps = args.enc_steps
        self.dec_steps = args.dec_steps
        self.training = phase=='train'
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        self.SAMPLE_FRAMES = 6

        self.inputs = []
        if not self.training:
            self.sessions = self.sessions[0:50]
            pass
        for session in self.sessions:
            target = np.load(osp.join(self.data_root, 'target_frames_24fps', session+'.npy'))
            num_frames = target.shape[0]
            num_frames = num_frames - (num_frames  % self.SAMPLE_FRAMES)
            target = target[:num_frames]

            target = target[self.SAMPLE_FRAMES//2::self.SAMPLE_FRAMES]

            seed = np.random.randint(self.enc_steps) if self.training else 0
            for start, end in zip(
                range(seed, target.shape[0] - self.dec_steps, self.enc_steps),
                range(seed + self.enc_steps, target.shape[0] - self.dec_steps, self.enc_steps)):
                if args.downsample_backgr and self.training:
                    background_vect = np.zeros_like(target[start:end])
                    background_vect[:, 0] = 1
                    if (target[start:end] == background_vect).all():
                        continue

                enc_target = target[start:end]
                dec_target = self.get_dec_target(target[start:end + self.dec_steps])
                self.inputs.append([
                    session, start, end, enc_target, dec_target,
                ])

    def get_dec_target(self, target_vector):
        target_matrix = np.zeros((self.enc_steps, self.dec_steps, target_vector.shape[-1]))
        for i in range(self.enc_steps):
            for j in range(self.dec_steps):
                # 0 -> [1, 2, 3]
                # target_matrix[i,j] = target_vector[i+j+1,:]
                # 0 -> [0, 1, 2]
                target_matrix[i,j] = target_vector[i+j,:]
        return target_matrix

    def __getitem__(self, index):
        session, start, end, enc_target, dec_target = self.inputs[index]

        camera_inputs = None
        for count in range(start, end):
            idx_frame = count * self.SAMPLE_FRAMES + (self.SAMPLE_FRAMES // 2)
            frame = Image.open(osp.join(self.data_root, self.camera_feature, session, str(idx_frame+1) + '.jpg')).convert('RGB')
            frame = self.transform(frame).to(dtype=torch.float32)
            if camera_inputs is None:
                camera_inputs = torch.zeros((end - start, frame.shape[0], frame.shape[1], frame.shape[2]),
                                            dtype=torch.float32)
            camera_inputs[count - start] = frame

        motion_inputs = np.zeros((end - start, self.enc_steps))  # optical flow will not be used
        motion_inputs = torch.as_tensor(motion_inputs.astype(np.float32))
        enc_target = torch.as_tensor(enc_target.astype(np.float32))
        dec_target = torch.as_tensor(dec_target.astype(np.float32))

        return camera_inputs, motion_inputs, enc_target, dec_target.view(-1, enc_target.shape[-1])

    def __len__(self):
        return len(self.inputs)