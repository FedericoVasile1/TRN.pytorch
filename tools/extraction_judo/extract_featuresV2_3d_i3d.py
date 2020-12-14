import os
import sys
import numpy as np

from torchvision import transforms
import torch
import torch.nn as nn
from PIL import Image

sys.path.append(os.getcwd())
from lib.models.i3d.i3d import InceptionI3d
from lib.datasets.judo_data_layer_e2e import I3DNormalization

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = InceptionI3d()
    model.load_state_dict(torch.load('lib/models/i3d/rgb_imagenet.pt'))
    model.dropout = nn.Identity()
    model.logits = nn.Identity()
    model.avg_pool = nn.AvgPool3d(kernel_size=[2, 4, 4], stride=(1, 1, 1))      # 4x4 instead of 7x7

    FEAT_VECT_DIM = 1024

    model = model.to(device)
    model.train(False)

    transform = transforms.Compose([
        transforms.Resize((112, 160)),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        I3DNormalization(),
    ])

    SAMPLE_FRAMES = 9     # generate a feature vector every SAMPLE_FRAMES frames

    DATA_ROOT = 'data/JUDO/UNTRIMMED'
    VIDEO_FRAMES = 'video_frames_25fps'   # base folder where the video folders (containing the frames) are
    VIDEO_FEATURES = 'i3d_112x112_chunk9'

    with torch.set_grad_enabled(False):
        videos_dir = os.listdir(os.path.join(DATA_ROOT, VIDEO_FRAMES))
        videos_dir = [dir for dir in videos_dir if '.mp4' in dir]
        for dir in videos_dir:
            num_frames = len(os.listdir(os.path.join(DATA_ROOT, VIDEO_FRAMES, dir)))
            num_frames = num_frames - (num_frames % SAMPLE_FRAMES)

            feat_vects_video = torch.zeros(num_frames//SAMPLE_FRAMES, FEAT_VECT_DIM, dtype=torch.float32)
            sample = torch.zeros(SAMPLE_FRAMES, 3, 112, 112, dtype=torch.float32)
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
                    sample = torch.zeros(SAMPLE_FRAMES, 3, 112, 112, dtype=torch.float32)

            np.save(os.path.join(DATA_ROOT, VIDEO_FEATURES, str(dir)+'.npy'), feat_vects_video.numpy())

if __name__ == '__main__':
    main()