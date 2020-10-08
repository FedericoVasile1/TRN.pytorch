import os
import numpy as np
import sys

sys.path.append(os.getcwd())
import _init_paths
from lib.models.i3d import InceptionI3d
from lib.datasets.judo_data_layer_e2e import I3DNormalization


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
    os.environ['CUDA_VISIBLE_DEVICES'] = str(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = InceptionI3d()
    model.load_state_dict(torch.load('rgb_imagenet.pt'))
    model.replace_logits(22)  # == num_classes
    model.dropout = nn.Identity()
    model.logits = nn.Identity()

    FEAT_VECT_DIM = 1024

    model = model.to(device)
    model.train(False)

    transform = transforms.Compose([
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        I3DNormalization(),
    ])

    #SAMPLE_FRAMES = 9     # generate a feature vector every SAMPLE_FRAMES frames
    SAMPLE_FRAMES = 6  # generate a feature vector every SAMPLE_FRAMES frames

    DATA_ROOT = 'data/THUMOS'
    VIDEO_FRAMES = 'video_frames_24fps'   # base folder where the video folders (containing the frames) are
    #VIDEO_FEATURES = 'i3d_224x224_chunk9'
    VIDEO_FEATURES = 'i3d_224x224_chunk6'

    with torch.set_grad_enabled(False):
        videos_dir = os.listdir(os.path.join(DATA_ROOT, VIDEO_FRAMES))
        videos_dir = [dir for dir in videos_dir if 'video' in dir]
        for dir in videos_dir:
            if str(dir) + '.npy' in os.listdir(os.path.join(DATA_ROOT, VIDEO_FEATURES)):
                continue

            num_frames = len(os.listdir(os.path.join(DATA_ROOT, VIDEO_FRAMES, dir)))
            num_frames = num_frames - (num_frames % SAMPLE_FRAMES)

            feat_vects_video = torch.zeros(num_frames//SAMPLE_FRAMES, FEAT_VECT_DIM, dtype=torch.float32)
            sample = torch.zeros(SAMPLE_FRAMES, 3, 224, 224, dtype=torch.float32)
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
                    sample = torch.zeros(SAMPLE_FRAMES, 3, 224, 224, dtype=torch.float32)

            print('Processed:')
            np.save(os.path.join(DATA_ROOT, VIDEO_FEATURES, str(dir)+'.npy'), feat_vects_video.numpy())
            print(str(dir))

if __name__ == '__main__':
    main()