import os.path as osp
import numpy as np
import warnings

import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image

class LateFusion_TRNJUDODataLayerE2E(data.Dataset):
    def __init__(self, args, phase='train'):
        if args.model != 'CNNLATEFUSION':
            raise Exception('This dataset layer can only work with --model == CNNLATEFUSION')

        self.data_root = args.data_root
        self.camera_feature = args.camera_feature
        self.motion_feature = args.motion_feature
        self.sessions = getattr(args, phase+'_session_set')
        self.enc_steps = args.enc_steps
        self.dec_steps = args.dec_steps
        self.training = phase=='train'

        self.transform = transforms.Compose([
            transforms.Resize((224, 320)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        # This means that for each chunk only the central frame label is taken and this will be the label of the
        #  entire chunk. Regarding the frames, all the frames of the chunk are fed one by one into
        #  the CNNLATEFUSION model, then predictions are averaged, this will be the prediction for the chunk.
        self.CHUNK_SIZE = args.chunk_size

        self.inputs = []
        for session in self.sessions:
            target = np.load(osp.join(self.data_root, 'target_frames_25fps', session+'.npy'))
            # round to multiple of CHUNK_SIZE
            num_frames = target.shape[0]
            num_frames = num_frames - (num_frames  % self.CHUNK_SIZE)
            target = target[:num_frames]
            # For each chunk, take only the central frame
            target = target[self.CHUNK_SIZE//2::self.CHUNK_SIZE]

            seed = np.random.randint(self.enc_steps) if self.training else 0
            for start, end in zip(range(seed, target.shape[0] - self.dec_steps, self.enc_steps),
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
                frame = Image.open(osp.join(self.data_root,
                                            self.camera_feature,
                                            session,
                                            str(idx_frame + 1) + '.jpg')).convert('RGB')
                frame = self.transform(frame).to(dtype=torch.float32)
                if camera_inputs is None:
                    camera_inputs = torch.zeros((end - start, self.CHUNK_SIZE, frame.shape[0], frame.shape[1], frame.shape[2]),
                                                dtype=torch.float32)
                camera_inputs[count - start, idx_frame - start_f] = frame

        if self.motion_feature == '':
            motion_inputs = np.zeros((end - start, self.enc_steps))
        else:
            # TODO: we have still not considered the possibility to insert optical flow in judo dataset, so
            #       this part below can be deleted
            if self.chunk_size == 6:
                motion_inputs = np.load(osp.join(self.data_root, self.motion_feature, session + '.npy'), mmap_mode='r')
                motion_inputs = motion_inputs[start:end]
            else:
                warnings.warn('Actually, we only offer optical flow images for args.chunk_size==6'
                              'Hence change this argument to 6 if you want ot use optical flow, otherwise will be discarded')
                motion_inputs = np.zeros((end - start, self.enc_steps))
        motion_inputs = torch.as_tensor(motion_inputs.astype(np.float32))

        enc_target = torch.as_tensor(enc_target.astype(np.float32))
        dec_target = torch.as_tensor(dec_target.astype(np.float32))

        return camera_inputs, motion_inputs, enc_target, dec_target

    def __len__(self):
        return len(self.inputs)

