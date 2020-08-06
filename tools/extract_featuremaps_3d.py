import _init_paths
import utils as utl

import os
import numpy as np

from torchvision import models, transforms
import torch
import torch.nn as nn

from PIL import Image

class Squeeze(nn.Module):
    def __init__(self):
        super(Squeeze, self).__init__()

    def forward(self, x):
        return x.squeeze(2)

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.video.r2plus1d_18(pretrained=True)
    model =  nn.Sequential(
        *list(model.children())[:-2],
        Squeeze(),
    )
    FEAT_MAP_DIM = (512, 7, 7)

    model = model.to(device)
    model.train(False)

    transform = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize([0.43216, 0.394666, 0.37645], [0.22803, 0.22145, 0.216989])
    ])

    CHUNK_SIZE = 6

    DATA_ROOT = 'data/THUMOS'
    VIDEO_FRAMES = 'video_frames_24fps'   # base folder where the video folders (containing the frames) are
    VIDEO_FEATURES = 'resnet3d_featuremaps'

    with torch.set_grad_enabled(False):
        videos_dir = os.listdir(os.path.join(DATA_ROOT, VIDEO_FRAMES))
        videos_dir = [dir for dir in videos_dir if 'video' in dir]
        for dir in videos_dir:
            if str(dir)+'.npy' in os.listdir(os.path.join(DATA_ROOT, VIDEO_FEATURES)):
                continue

            feat_maps_video = torch.zeros(num_frames//CHUNK_SIZE, FEAT_MAP_DIM[0], FEAT_MAP_DIM[1], FEAT_MAP_DIM[2], dtype=torch.float32)
            num_frames = len(os.listdir(os.path.join(DATA_ROOT, VIDEO_FRAMES, dir)))
            num_frames = num_frames - (num_frames % CHUNK_SIZE)
            sample = torch.zeros(CHUNK_SIZE, 3, 112, 112, dtype=torch.float32)
            for idx_frame in range(0, num_frames):
                # idx_frame+1 because frames start from 1.  e.g. 1.jpg
                frame = Image.open(os.path.join(DATA_ROOT, VIDEO_FRAMES, dir, str(idx_frame+1)+'.jpg')).convert('RGB')
                frame = transform(frame).to(dtype=torch.float32)
                sample[idx_frame % CHUNK_SIZE] = frame
                if idx_frame % CHUNK_SIZE == CHUNK_SIZE - 1:
                    sample = sample.permute(1, 0, 2, 3).unsqueeze(0).to(device)
                    # forward pass
                    feat_map = model(sample)
                    feat_maps_video[idx_frame//CHUNK_SIZE] = feat_map.squeeze(0)
                    sample = torch.zeros(CHUNK_SIZE, 3, 112, 112, dtype=torch.float32)

            np.save(os.path.join(DATA_ROOT, VIDEO_FEATURES, str(dir)+'.npy'), feat_maps_video.numpy())
            print('Processed video ' + str(dir))

if __name__ == '__main__':
    main()