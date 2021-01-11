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

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = str(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = InceptionI3d()
    model.load_state_dict(torch.load('rgb_imagenet.pt'))

    # EXTRACT FEATURES AT MIXED4F LAYER
    model.VALID_ENDPOINTS = model.VALID_ENDPOINTS[:-5]

    model.avg_pool = nn.Identity()
    model.dropout = nn.Identity()
    model.logits = nn.Identity()

    FEAT_MAP_DIM = (832, 3, 14, 14)     # C x T x H x W

    model = model.to(device)
    model.train(False)

    transform = transforms.Compose([
        transforms.Resize((224, 320)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        I3DNormalization(),
    ])

    CHUNK_SIZE = 9     # generate a feature vector every CHUNK_SIZE frames
    BATCH_SIZE = 64

    DATA_ROOT = 'data/JUDO/UNTRIMMED'
    VIDEO_FRAMES = 'video_frames_25fps'   # base folder where the video folders (containing the frames) are
    VIDEO_FEATURES = 'i3d_featuremaps_mixed4f_chunk'+str(CHUNK_SIZE)

    with torch.set_grad_enabled(False):
        videos_dir = os.listdir(os.path.join(DATA_ROOT, VIDEO_FRAMES))
        videos_dir = [dir for dir in videos_dir if '.mp4' in dir]
        for dir in videos_dir:
            num_frames = len(os.listdir(os.path.join(DATA_ROOT, VIDEO_FRAMES, dir)))
            num_frames = num_frames - (num_frames % CHUNK_SIZE)

            feat_vects_video = []
            batch = []
            for i in range(0, num_frames, CHUNK_SIZE):
                sample = []
                for idx_frame in range(i, i+CHUNK_SIZE):
                    # idx_frame+1 because frames start from 1.  e.g. 1.jpg
                    frame = Image.open(os.path.join(DATA_ROOT,
                                                    VIDEO_FRAMES,
                                                    dir,
                                                    str(idx_frame+1)+'.jpg')).convert('RGB')
                    frame = transform(frame).to(dtype=torch.float32)
                    sample.append(frame)

                sample = torch.stack(sample)
                batch.append(sample.permute(1, 0, 2, 3))
                if len(batch) == BATCH_SIZE:
                    batch = torch.stack(batch).to(device)
                    # forward pass
                    feat_vect = model(batch)
                    feat_vects_video.append(feat_vect)
                    batch = []

            if len(batch) != 0:
                batch = torch.stack(batch).to(device)
                feat_vect = model(batch)
                feat_vects_video.append(feat_vect)

            feat_vects_video = torch.cat(feat_vects_video).cpu()
            np.save(os.path.join(DATA_ROOT,
                                 VIDEO_FEATURES,
                                 str(dir)+'.npy'), feat_vects_video.numpy())

if __name__ == '__main__':
    main()