import _init_paths
import utils as utl

import os
import numpy as np

from torchvision import models, transforms
import torch
import torch.nn as nn

from PIL import Image

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.video.r2plus1d_18(pretrained=True)
    FEAT_VECT_DIM = model.fc.in_features
    model.fc = nn.Identity()

    model = model.to(device)
    model.train(False)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])
    ])

    SAMPLE_FRAMES = 6     # generate a feature vector every SAMPLE_FRAMES frames

    DATA_ROOT = 'data/THUMOS'
    VIDEO_FRAMES = 'video_frames_24fps'   # base folder where the video folders (containing the frames) are
    VIDEO_FEATURES = 'resnet3d_180x320'

    with torch.set_grad_enabled(False):
        videos_dir = os.listdir(os.path.join(DATA_ROOT, VIDEO_FRAMES))
        videos_dir = [dir for dir in videos_dir if 'video' in dir]
        for dir in videos_dir:
            num_frames = len(os.listdir(os.path.join(DATA_ROOT, VIDEO_FRAMES, dir)))
            num_frames = num_frames - (num_frames % SAMPLE_FRAMES)

            feat_vects_video = torch.zeros(num_frames//SAMPLE_FRAMES, FEAT_VECT_DIM, dtype=torch.float32)
            sample = torch.zeros(SAMPLE_FRAMES, 3, 180, 320, dtype=torch.float32)
            for idx_frame in range(0, num_frames):
                # idx_frame+1 because frames start from 1.  e.g. 1.jpg
                frame = Image.open(os.path.join(DATA_ROOT, VIDEO_FRAMES, dir, str(idx_frame+1)+'.jpg')).convert('RGB')
                frame = transform(frame).to(dtype=torch.float32)
                sample[idx_frame%SAMPLE_FRAMES] = frame
                if idx_frame%SAMPLE_FRAMES == SAMPLE_FRAMES-1:
                    sample = sample.permute(1, 0, 2, 3).unsqueeze(0).to(device)
                    # forward pass
                    feat_vect = model(sample)     # TODO: load a batch instead of a single sample
                    feat_vects_video[idx_frame//SAMPLE_FRAMES] = feat_vect.squeeze(0)
                    sample = torch.zeros(SAMPLE_FRAMES, 3, 180, 320, dtype=torch.float32)

            np.save(os.path.join(DATA_ROOT, VIDEO_FEATURES, str(dir)+'.npy'), feat_vects_video.numpy())

if __name__ == '__main__':
    main()